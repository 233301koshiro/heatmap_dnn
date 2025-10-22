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
from urdf_to_graph_utilis import debug_edge_index, print_step_header, print_step_footer
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
        self.in_proj = mlp([in_cat, bottleneck_dim, hidden], dropout=dropout)

        self.layers = nn.ModuleList([
            DownUpLayer(hidden, bottleneck_dim=bottleneck_dim, dropout=dropout)
            for _ in range(num_rounds)
        ])
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
        for layer in self.layers:
            h = layer(h, edge_index)
        return self.out_proj(h)


# =========================================================
# 全体：MaskedTreeAutoencoder（mask_flag の有無を切替可能）
# =========================================================
class MaskedTreeAutoencoder(nn.Module):
    def __init__(self, in_dim=19, hidden=128, bottleneck_dim=128,
                 enc_rounds=2, dec_rounds=2, dropout=0.1,
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

    def forward(self, data: Data, mask_idx: Optional[torch.Tensor] = None,
                recon_only_masked: bool = True) -> Tuple[torch.Tensor, dict]:
        x, ei = data.x, data.edge_index
        device = x.device

        mode = "MASK MODE (mask_flag 使用)" if self.use_mask_flag else "AE MODE (mask_flag 無し)"
        print(f"[MaskedTreeAE] forward: {mode} | N={x.size(0)}, F={x.size(1)}, E={ei.size(1)}")

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

        # Decoder（anchorは完全廃止）
        x_hat = self.decoder(mask_flag, h_enc, ei)

        out = {"x_hat": x_hat, "mask_flag": mask_flag, "node_context": h_enc}
        if recon_only_masked and (mask_idx is not None) and (mask_idx.numel() > 0):
            out["recon_target_idx"] = mask_idx
        return x_hat, out


# =========================================================
# 学習用コンフィグ / 損失 / ループ
# =========================================================
@dataclass
class TrainCfg:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    recon_only_masked: bool = True
    mask_strategy: str = "none"  # 'one' or 'none'

def loss_reconstruction(x_hat: torch.Tensor, x_true: torch.Tensor, out: dict) -> torch.Tensor:
    """
    MAE（L1 Loss）。recon_target_idx がある場合はマスク行のみで計算。
    """
    if "recon_target_idx" in out:
        idx = out["recon_target_idx"]
        return F.l1_loss(x_hat[idx], x_true[idx])  # reduction='mean' がデフォ
    else:
        return F.l1_loss(x_hat, x_true)


def train_one_epoch(model: nn.Module, loader, device, cfg: TrainCfg,
                    verbose_mask: bool = False, show_edges: bool = False, tag: str = "train"):
    model.train()
    opt = getattr(model, "_opt", None)
    if opt is None:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        model._opt = opt

    total = 0.0
    for step, data in enumerate(loader, 1):
        print_step_header(step, tag)  # ★ step 開始

        data = data.to(device)

        # （任意）edge_index の先頭/末尾を確認
        if show_edges:
            debug_edge_index(data, k=20, title=f"{tag} step {step}", show_up_first=0)

        # マスク選択
        with torch.no_grad():
            mask_idx = choose_mask_idx(data, device, cfg.mask_strategy, verbose=verbose_mask)
            if mask_idx is None:
                print("[mask] none")
            else:
                print(f"[mask] idx={mask_idx.tolist()}")

        # forward + loss
        x_hat, out = model(data, mask_idx=mask_idx, recon_only_masked=cfg.recon_only_masked)
        loss = loss_reconstruction(x_hat, data.x, out)
        print(f"[loss] {float(loss.item()):.6f}  | graphs={data.num_graphs} nodes={data.num_nodes} edges={data.edge_index.size(1)}")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        total += float(loss.item()) * data.num_graphs

        print_step_footer(step, tag)  # ★ step 終了

    return total / max(1, len(loader.dataset))


@torch.no_grad()
@torch.no_grad()
def eval_loss(model: nn.Module, loader, device, **override):
    model.eval()
    total, n = 0.0, 0

    cfg = getattr(model, "_cfg", None)
    recon_only_masked = (cfg.recon_only_masked if cfg is not None else True)
    mask_strategy     = (getattr(cfg, "mask_strategy", "one") if cfg is not None else "one")

    if "recon_only_masked" in override:
        recon_only_masked = override["recon_only_masked"]
    if "mask_strategy" in override:
        mask_strategy = override["mask_strategy"]

    for step, data in enumerate(loader, 1):
        print_step_header(step, tag="val")

        data = data.to(device)
        mask_idx = choose_mask_idx(data, device, mask_strategy, verbose=False)

        x_hat, out = model(data, mask_idx=mask_idx, recon_only_masked=recon_only_masked)
        loss = loss_reconstruction(x_hat, data.x, out)
        print(f"[val loss] {float(loss.item()):.6f}")

        total += float(loss.item()) * data.num_graphs
        n += data.num_graphs

        print_step_footer(step, tag="val")

    return total / max(1, n)

