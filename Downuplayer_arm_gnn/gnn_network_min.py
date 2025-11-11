# masked_tree_autoencoder.py
from dataclasses import dataclass
from typing import Optional, Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, AttentionalAggregation
from torch_scatter import scatter_mean
import numpy as np
from urdf_to_graph_utilis import debug_edge_index, print_step_header, print_step_footer
def _vprint(enabled: bool, *args, **kwargs):
    if enabled:
        print(*args, **kwargs)


# =========================================================
# ヘルパ：MLP（Linear -> [LN -> ReLU -> Dropout] -> Linear ...）
# =========================================================
def mlp(sizes: List[int], dropout: float = 0.1, verbose: bool = False) -> nn.Sequential:
    """
    sizes=[in, mid, out] のような形で MLP を作る。
    中間層のみに LayerNorm/ReLU/Dropout を入れ、最終 Linear 後は入れない。
    """
    if verbose:
        print("[mlp] layer sizes:", sizes)
    layers: List[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i+1], bias=False))  # LN を使うので bias は省略
        if i < len(sizes) - 2:  # 中間層のときだけ入れる
            layers += [nn.LayerNorm(sizes[i+1]), nn.ReLU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers)

#massやorigin_xyzにノイズを加え,データをかさ増しする関数
def apply_noise_augmentation(data, noise_level=0.05):
    """
    data.x の特定の列に学習時のみノイズを加える。
    対象: mass(2), origin_xyz(10,11,12) とする例
    """
    # ノイズ対象の列インデックス
    target_cols = [2, 10, 11, 12] # mass, origin_x, origin_y, origin_z

    # データの複製（元のデータを壊さないため。必要に応じて）
    x_aug = data.x.clone()

    for c in target_cols:
        # [-noise_level, +noise_level] の一様乱数を生成
        noise = (torch.rand_like(x_aug[:, c]) * 2 - 1) * noise_level
        # 乗法ノイズ: value * (1 + noise)
        x_aug[:, c] = x_aug[:, c] * (1.0 + noise)

    # dataの書き換え (shallow copyしてxだけ差し替えると安全)
    data_aug = data.clone()
    data_aug.x = x_aug
    return data_aug

# =========================================================
# マスクノード選択（必要に応じて verbose 出力）
# =========================================================
def choose_mask_idx(
    data: Data,
    device,
    strategy: str,
    verbose: bool = False,
) -> Optional[torch.Tensor]:
    """
    バッチ内の各グラフから 1 ノードずつランダムにマスク対象を選ぶ。
    strategy='none' のときは None を返す。
    返り値: LongTensor[ num_graphs_in_batch ] or None
    """
    if strategy is None or strategy.lower() == "none":
        if verbose:
            print("[mask] strategy='none' → マスクしません（返り値=None）")
        return None

    batch = getattr(data, "batch", torch.zeros(data.num_nodes, dtype=torch.long, device=device))
    uniq = torch.unique(batch)
    chosen: List[int] = []

    if verbose:
        print(f"[mask] strategy='{strategy}', graphs_in_batch={uniq.numel()}, num_nodes={data.num_nodes}")

    for b in uniq.tolist():
        idx = (batch == b).nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            if verbose:
                print(f"  - graph_id={b}: ノードなし → スキップ")
            continue
        sel = idx[torch.randint(0, idx.numel(), (1,), device=idx.device)].item()
        if verbose:
            print(f"  - graph_id={b}: ノード数={idx.numel()} → マスク対象 node_idx={sel}")
        chosen.append(sel)

    if len(chosen) == 0:
        if verbose:
            print("[mask] 選択できるノードがありません → 返り値=None")
        return None

    out = torch.tensor(chosen, device=device, dtype=torch.long)
    if verbose:
        print(f"[mask] 返り値: {out.tolist()}")
    return out

# =========================================================
# DownUpLayer：木構造に沿って GINConv を down/up で往復
# =========================================================
class DownUpLayer(nn.Module):
    def __init__(self, hidden: int, bottleneck_dim: int, dropout: float = 0.1):
        super().__init__()
        # GINConv の更新関数 φ は MLP。down/up で別パラメータ。
        self.down = GINConv(mlp([hidden, bottleneck_dim, hidden], dropout=dropout), train_eps=True)
        self.up   = GINConv(mlp([hidden, bottleneck_dim, hidden], dropout=dropout), train_eps=True)
        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.dir_emb = nn.Parameter(torch.randn(2, hidden) * 0.02)  # [down, up] これは方向埋め込み

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # 親→子
        #GINon
        x = self.ln1(F.relu(self.down(x, edge_index) + self.dir_emb[0]))
        # 子→親
        rev = edge_index[[1, 0]]
        #シンプルに
        x = self.ln2(F.relu(self.up(x, rev) + self.dir_emb[1]))
        return x

# =========================================================
# root プーリング（必要なら）
# =========================================================
def root_pool(x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, root_index: Optional[torch.Tensor]) -> torch.Tensor:
    if root_index is not None:
        return x[root_index]  # [B,H]

    N, E = x.size(0), edge_index.size(1)
    indeg = torch.zeros(N, device=x.device, dtype=torch.long)
    indeg.scatter_add_(0, edge_index[1], torch.ones(E, device=x.device, dtype=torch.long))
    is_root = (indeg == 0).float()

    if (is_root.sum() == 0) or (scatter_mean(is_root, batch, dim=0) == 0).any():
        attn = AttentionalAggregation(gate_nn=mlp([x.size(1), x.size(1)//2, 1]))
        return attn(x, batch)
    x_root = scatter_mean(x * is_root.unsqueeze(-1), batch, dim=0) / (scatter_mean(is_root, batch, dim=0).unsqueeze(-1) + 1e-6)
    return x_root

# =========================================================
# Encoder：各ノード独立の in_proj → DownUpLayer×K
# =========================================================
class TreeEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, num_rounds: int = 2, dropout: float = 0.1, bottleneck_dim: int = 128):
        super().__init__()
        self.in_proj = mlp([in_dim, bottleneck_dim, hidden], dropout=dropout)
        self.layers = nn.ModuleList([
            DownUpLayer(hidden, bottleneck_dim=bottleneck_dim, dropout=dropout)
            for _ in range(num_rounds)
        ])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)  # [N, hidden]
        for layer in self.layers:
            h = layer(h, edge_index)
        return h  # [N, hidden]

# =========================================================
# Decoder：anchor/mask_flag/node_context を結合 → DownUpLayer×K → out_proj
# =========================================================
class TreeDecoder(nn.Module):
    def __init__(self, hidden: int = 128, num_rounds: int = 2,
                 dropout: float = 0.1, out_dim: int = 19, bottleneck_dim: int = 128,
                 use_mask_flag: bool = False):
        super().__init__()
        self.hidden = hidden
        self.use_mask_flag = use_mask_flag
        in_cat = (1 if self.use_mask_flag else 0) + hidden  # ← anchorを廃止
        #いつもと違ってin_catがある．これはmask_flag分の次元を加えたもの
        #mlpの第一引数にin_catを渡すと，linearレイヤーの入力次元がin_catになる
        self.in_proj = mlp([in_cat, bottleneck_dim, hidden], dropout=dropout)

        self.layers = nn.ModuleList([
            DownUpLayer(hidden, bottleneck_dim=bottleneck_dim, dropout=dropout)
            for _ in range(num_rounds)
        ])
        #mlpの第二引数に16を渡しているのは，中間層の次元数を16に設定しているため
        #つまり16次元になるのは
        self.out_proj = mlp([hidden, 16, out_dim], dropout=dropout)

    def forward(self,
                mask_flag: Optional[torch.Tensor],    # [N,1] or None
                node_context: torch.Tensor,           # [N, hidden]
                edge_index: torch.Tensor) -> torch.Tensor:
        if self.use_mask_flag:
            if mask_flag is None:
                mask_flag = node_context.new_zeros(node_context.size(0), 1)
            h_in = torch.cat([mask_flag, node_context], dim=1)
        else:
            h_in = node_context
        h = self.in_proj(h_in)
        # DownUpLayer を回す*2
        for layer in self.layers:
            h = layer(h, edge_index)
        
        # 出力投影[nondes_total,128]
        return self.out_proj(h)


# =========================================================
# 全体：MaskedTreeAutoencoder（mask_flag の有無を切替可能）
# =========================================================
class MaskedTreeAutoencoder(nn.Module):
    def __init__(self, in_dim=19, hidden=128, bottleneck_dim=128,
                 enc_rounds=5, dec_rounds=5, dropout=0.1,
                 mask_strategy: str = "none"):
        super().__init__()
        self.mask_strategy = mask_strategy
        self.use_mask_flag = (mask_strategy != "none")

        enc_in = in_dim + (1 if self.use_mask_flag else 0)
        self.encoder = TreeEncoder(enc_in, hidden, enc_rounds, dropout, bottleneck_dim)
        self.decoder = TreeDecoder(hidden=hidden, num_rounds=dec_rounds,
                                   dropout=dropout, out_dim=in_dim,
                                   bottleneck_dim=bottleneck_dim,
                                   use_mask_flag=self.use_mask_flag)
        self._forward_calls = 0

    def forward(self, data: Data, mask_idx: Optional[torch.Tensor] = None,
                recon_only_masked: bool = True) -> Tuple[torch.Tensor, dict]:
        x, ei = data.x, data.edge_index
        device = x.device
        self._forward_calls += 1
        # マスクノード選択
        if getattr(self, "_cfg", None):
            if self._cfg.verbose and self._cfg.log_interval:
                # training中かつ指定間隔でだけ出す
                if self.training and (self._forward_calls % self._cfg.log_interval == 0):
                    N = int(data.num_nodes); F = int(data.num_node_features); E = int(data.num_edges)
                    print(f"[MaskedTreeAE] forward: AE MODE | N={N}, F={F}, E={E}")

        # 入力構築
        if self.use_mask_flag:
            mask_flag = torch.zeros(x.size(0), 1, device=device)
            if mask_idx is not None and mask_idx.numel() > 0:
                x = x.clone()
                x[mask_idx] = 0.0
                mask_flag[mask_idx] = 1.0
                print(f"[MaskedTreeAE] masked nodes: {mask_idx.tolist()}")
            enc_in = torch.cat([x, mask_flag], dim=1)
        else:
            mask_flag = None
            mask_idx = None
            enc_in = x

        # Encoder
        h_enc = self.encoder(enc_in, ei)                # [N, hidden]

        # Decoder
        x_hat = self.decoder(mask_flag, h_enc, ei)
        # 軸成分(9,10,11と仮定)を正規化。イプシロンを入れてゼロ割を防ぐ
        #feature_names: ['deg', 'depth', 'mass', 'jtype_is_revolute', 'jtype_is_continuous', 'jtype_is_prismatic', 'jtype_is_fixed', 'axis_x', 'axis_y', 'axis_z', 'origin_x', 'origin_y', 'origin_z']
        #のためaxisは7,8,9
        axis_pred = x_hat[:, 7:10]
        axis_norm = axis_pred / (axis_pred.norm(dim=1, keepdim=True) + 1e-9)
        x_hat = torch.cat([x_hat[:, :9], axis_norm, x_hat[:, 12:]], dim=1)
        out = {"x_hat": x_hat, "mask_flag": mask_flag, "node_context": h_enc}
        if recon_only_masked and (mask_idx is not None) and (mask_idx.numel() > 0):
            out["recon_target_idx"] = mask_idx
        return x_hat, out


# =========================================================
# 学習用コンフィグ / 損失 / ループ
# =========================================================
@dataclass
# 既存の TrainCfg に2項目追加
@dataclass
class TrainCfg:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 1
    loss_weight: Optional[List[float]] = None  # 各特徴量ごとの損失重み付けリスト
    recon_only_masked: bool = True
    mask_strategy: str = "none"
    verbose: bool = True            # 詳細ログを出すか
    log_interval: int = 0           # バッチ内ログを出すステップ間隔(0で出さない)
    mask_k: int = 1                # 各グラフからマスクするノード数

def loss_reconstruction(x_hat: torch.Tensor, x_true: torch.Tensor, out: dict) -> torch.Tensor:
    """
    MAE（L1 Loss）。recon_target_idx がある場合はマスク行のみで計算。
    """
    if "recon_target_idx" in out:
        idx = out["recon_target_idx"]
        # TODO: マスク時もコサイン類似度損失を使うか確認
        return F.l1_loss(x_hat[idx], x_true[idx])  # 現在はマスク時L1のまま
    else:
        if False :
            return F.l1_loss(x_hat, x_true)
        else :
            # 安定化のための微小値
            eps = 1e-12 
            
            # --- スプリットを (axis_* が 7,8,9 列目になるよう) 修正 ---
            # 特徴量 [0:7] (deg, depth, mass, jtype*4)
            x_hat_1, x_true_1 = x_hat[:, :7], x_true[:, :7]
            # 特徴量 [7:10] (axis_x, axis_y, axis_z)
            x_hat_2, x_true_2 = x_hat[:, 7:10], x_true[:, 7:10]
            # 特徴量 [10:] (origin_x, origin_y, origin_z)
            x_hat_3, x_true_3 = x_hat[:, 10:], x_true[:, 10:]
            
            # L1 Loss (非ベクトル部分)
            loss_1 = F.l1_loss(x_hat_1, x_true_1)
            loss_3 = F.l1_loss(x_hat_3, x_true_3)
            
            # --- F.normalize に eps を指定してゼロ除算を回避 ---
            x_hat_2n = F.normalize(x_hat_2, p=2, dim=1, eps=eps)
            x_true_2n = F.normalize(x_true_2, p=2, dim=1, eps=eps)
            
            # コサイン類似度損失 (1 - cos_sim)
            dot_product = torch.sum(x_hat_2n * x_true_2n, dim=1).mean()
            loss_dir = 1 - dot_product
            
            return loss_1 + loss_3 + loss_dir


def train_one_epoch(model, loader, device, cfg, log_every_steps: int = 0):
    model.train()
    # lazy optimizer
    if not hasattr(model, "_opt") or model._opt is None:
        model._opt = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=getattr(cfg, "lr", 1e-3),
            weight_decay=getattr(cfg, "weight_decay", 1e-4),
        )

    # 損失重みの準備 (ループ外で1回だけ行う)
    w_tensor = None
    if cfg.loss_weight:
         # [1, D] の形状にしておけば、後で自動的にブロードキャストされる
         w_tensor = torch.tensor(cfg.loss_weight, device=device).unsqueeze(0)

    total_loss = 0.0
    for step, data in enumerate(loader, start=1):
        data = data.to(device)
        if model.training:
            data = apply_noise_augmentation(data)
        model._opt.zero_grad(set_to_none=True)

        # --- マスク index の作成 ---
        mask_idx = None
        if cfg.recon_only_masked and (cfg.mask_strategy in ("one", "k")):
            k = 1 if cfg.mask_strategy == "one" else max(1, int(cfg.mask_k))
            if hasattr(data, "ptr"):
                mask_idx = _pick_mask_indices_from_batch(data, k=k)
                mask_idx = mask_idx.to(device)

        # --- forward ---
        pred, out = model(data, mask_idx=mask_idx, recon_only_masked=cfg.recon_only_masked)
        
        # --- loss ---
        # ここで w_tensor を渡す
        #loss = loss_reconstruction(pred, data.x, out, weight=w_tensor)
        
        #aixisのlossではweightを使っていないのでへんこう
        loss = loss_reconstruction(pred, data.x, out)

        loss.backward()
        model._opt.step()

        total_loss += float(loss.item())

        if cfg.verbose and log_every_steps:
            if (step == 1) or (step == len(loader)) or (step % log_every_steps == 0):
                print(f"[loss] {float(loss.item()):.6f}  | graphs={data.num_graphs} nodes={data.num_nodes} edges={data.num_edges}")

    return total_loss / max(1, len(loader))


@torch.no_grad()
def eval_loss(model, loader, device, recon_only_masked: bool = True, mask_strategy: str = "none", log_every_steps: int = 0, verbose: bool = False):
    model.eval()
    total = 0.0
    # --- loss_weight の作成 ---
    cfg = getattr(model, "_cfg", None)
    w_tensor = None
    if cfg and cfg.loss_weight:
        w_tensor = torch.tensor(cfg.loss_weight, device=device).unsqueeze(0)

    for step, data in enumerate(loader, start=1):
        data = data.to(device)
        #out  = model(data)
        #pred = out[0] if isinstance(out, (tuple, list)) else out
        #loss = ((pred - data.x) ** 2).mean()
        # --- 評価時も同様にマスク index を作る（seed固定したいならここでnp RNG固定してもOK） ---
        mask_idx = None
        if recon_only_masked and (mask_strategy in ("one", "k")):
            k = 1 if mask_strategy == "one" else  max(1, int(getattr(getattr(model, "_cfg", None), "mask_k", 1)))
            if hasattr(data, "ptr"):
                mask_idx = _pick_mask_indices_from_batch(data, k=k).to(device)

        pred, out = model(data, mask_idx=mask_idx, recon_only_masked=recon_only_masked)
        #loss = loss_reconstruction(pred, data.x, out, weight=w_tensor)
        loss = loss_reconstruction(pred, data.x, out)
        total += float(loss.item())

        # 評価時は静かに。必要なら verbose で間引きログ
        if verbose and log_every_steps:
            if (step == 1) or (step == len(loader)) or (step % log_every_steps == 0):
                print(f"[val step] {step}/{len(loader)}  loss={float(loss.item()):.6f}")

    avg = total / max(1, len(loader))
    return avg

def _as_tensor(y, ref: torch.Tensor) -> torch.Tensor:
    """
    モデル出力が tuple/list/dict でも先頭 or よくあるキーでテンソル化する。
    """
    if isinstance(y, (tuple, list)):
        y = y[0]
    if isinstance(y, dict):
        for k in ("pred", "recon", "logits", "output"):
            if k in y:
                y = y[k]; break
        else:
            y = next(iter(y.values()))
    if not torch.is_tensor(y):
        y = torch.as_tensor(y, dtype=ref.dtype, device=ref.device)
    return y

def _pick_mask_indices_from_batch(batch, k, rng=None):
    """
    torch_geometricのBatchを想定。各グラフ境界 ptr から k 個ずつノードを無作為抽出。
    """
    if rng is None:
        rng = np.random.default_rng()
    ptr = batch.ptr.cpu().numpy()  # shape: [G+1]
    idxs = []
    for i in range(len(ptr) - 1):
        a, b = int(ptr[i]), int(ptr[i + 1])
        n = b - a
        kk = min(k, n)
        sel = rng.choice(n, size=kk, replace=False)
        idxs.append(torch.as_tensor(sel + a, dtype=torch.long))
    return torch.cat(idxs, dim=0)