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
from urdf_print_debug_utils import debug_edge_index, print_step_header, print_step_footer

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
        # Down用とUp用のGINConvを定義
        self.down = GINConv(mlp([hidden, bottleneck_dim, hidden], dropout=dropout), train_eps=True)
        self.up   = GINConv(mlp([hidden, bottleneck_dim, hidden], dropout=dropout), train_eps=True)
        
        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden)
        
        # 方向埋め込み
        self.dir_emb = nn.Parameter(torch.randn(2, hidden) * 0.02)

        # ★追加: Downの結果とUpの結果を結合して元の次元に戻す層
        # input: hidden * 2 (downの結果 + upの結果), output: hidden
        self.combine = nn.Linear(hidden * 2, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # 1. 親→子 (Down Pass)
        # xを入力として計算
        h_down = self.ln1(F.relu(self.down(x, edge_index) + self.dir_emb[0]))

        # 2. 子→親 (Up Pass)
        # ★重要: ここでも 'x' (元の特徴量) を入力にするか、あるいはh_downを使うかですが、
        # 双方向の情報を独立して拾うため、元の 'x' を入力にする並列構造にします。
        rev = edge_index[[1, 0]]
        h_up = self.ln2(F.relu(self.up(x, rev) + self.dir_emb[1]))

        # 3. 結合 (Concatenate)
        # 親からの情報(h_down)と子からの情報(h_up)を横に繋げる
        h_cat = torch.cat([h_down, h_up], dim=-1) # shape: [N, hidden*2]
        
        # 4. 統合と出力
        out = self.combine(h_cat)
        return self.dropout(out)

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
    def __init__(self, in_dim: int, hidden: int = 128, num_rounds: int = 1, dropout: float = 0.1, bottleneck_dim: int = 128):
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
    def __init__(self, hidden: int = 128, num_rounds: int = 1,
                 dropout: float = 0.1, out_dim: int = 13, bottleneck_dim: int = 128,
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
        self.out_proj = mlp([hidden, bottleneck_dim, out_dim], dropout=dropout)

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

        #Decoderでgnnレイヤーを使わないとのご指摘(賛否両論あり)
        # DownUpLayer を回す
        for layer in self.layers:
            h = layer(h, edge_index)
        
        # 出力投影[nondes_total,128]
        return self.out_proj(h)


# =========================================================
# 全体：MaskedTreeAutoencoder（mask_flag の有無を切替可能）
# =========================================================
class MaskedTreeAutoencoder(nn.Module):
    def __init__(self, in_dim=19, hidden=128, bottleneck_dim=128,
                 enc_rounds=1, dec_rounds=1, dropout=0.1,
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
                    N = int(data.num_nodes); Fdim = int(data.num_node_features); E = int(data.num_edges)
                    print(f"[MaskedTreeAE] forward: AE MODE | N={N}, F={Fdim}, E={E}")

        # 入力構築
        if self.use_mask_flag:
            mask_flag = torch.zeros(x.size(0), 1, device=device)
            if mask_idx is not None and mask_idx.numel() > 0:
                x = x.clone()
                x[mask_idx] = 0.0
                mask_flag[mask_idx] = 1.0
                # ★修正: verbose制御下でのログ出力
                if getattr(self, "_cfg", None) and self._cfg.verbose:
                    print(f"[MaskedTreeAE] masked nodes: {mask_idx.tolist()}")
            enc_in = torch.cat([x, mask_flag], dim=1)
        else:
            mask_flag = None
            mask_idx = None
            enc_in = x

        # Encoder
        h_enc = self.encoder(enc_in, ei)                # [N, hidden]

        # Decoder
        x_raw = self.decoder(mask_flag, h_enc, ei)#生の予測値 [N, in_dim]

        """
        loss_recと合わせて二重に正規化しないように，ここでは正規化しない
        # 軸成分(9,10,11と仮定)を正規化。イプシロンを入れてゼロ割を防ぐ
        #feature_names: ['deg', 'depth', 'mass', 'jtype_is_revolute', 'jtype_is_continuous', 'jtype_is_prismatic', 'jtype_is_fixed', 'axis_x', 'axis_y', 'axis_z', 'origin_x', 'origin_y', 'origin_z']
        #のためaxisは7,8,9
        axis_pred = x_hat[:, 7:10]
        axis_norm = axis_pred / (axis_pred.norm(dim=1, keepdim=True) + 1e-9)
        # [0:7] -> deg から j_fix (7列)
        # [7:10] -> axis_norm (3列)
        # [10:13] -> origin_x から origin_z (3列)
        x_hat = torch.cat([
            x_hat[:, :7],   # 0から6 (deg, depth, mass, jtype...)
            axis_norm,      # 7から9 (正規化済み axis)
            x_hat[:, 10:]   # 10から12 (origin_x, origin_y, origin_z)
        ], dim=1)        
        out = {"x_hat": x_hat, "mask_flag": mask_flag, "node_context": h_enc}
        if recon_only_masked and (mask_idx is not None) and (mask_idx.numel() > 0):
            out["recon_target_idx"] = mask_idx
        return x_hat, 
        """

        # === unit axis をここで一元的に作る ===
        axis_norm = F.normalize(x_raw[:, 7:10], dim=-1, eps=1e-6)
        #quat_norm = F.normalize(x_raw[:, 13:17], dim=-1, eps=1e-6)
        rot6d_raw = x_raw[:, 13:19]  # 13番目から18番目までの6要素
        # === 出力テンソルを組み立て（評価/損失と整合）===
        # 0-2: deg, depth, mass は 0..1 に収めたいので sigmoid
        # 3-6: joint logits はそのまま（CE 用）
        # 7-9: axis は unit ベクトル
        # 10-12: origin は“今は”生のまま（min-maxと合わせたいなら後でここだけ sigmoid に切替可）
        x_hat = torch.cat([
            torch.sigmoid(x_raw[:, 0:3]),   # 0..2
            x_raw[:, 3:7],                  # 3..6
            axis_norm,                      # 7..9
            #torch.sigmoid(x_raw[:,10:13]),                # 10..12(origin)
            x_raw[:,10:13],
            #x_raw[:,10:13]#baxterなどの大きなロボットを扱うときはsigmoidは邪魔らしい(普段はいいけど)
            #quat_norm                       # 13..16
            rot6d_raw                       # 13..18 (rot6d) ★ここを変更
        ], dim=1)


        out = {"x_hat": x_hat, "mask_flag": mask_flag, "node_context": h_enc}
        if recon_only_masked and (mask_idx is not None) and (mask_idx.numel() > 0):
            out["recon_target_idx"] = mask_idx
        return x_hat, out


# =========================================================
# 学習用コンフィグ / 損失 / ループ
# =========================================================
@dataclass
class TrainCfg:
    lr: float
    weight_decay: float
    epochs: int
    loss_weight: Optional[List[float]] = None  # 各特徴量ごとの損失重み
    mask_strategy: str = "none"                # マスク戦略
    mask_k: int = 1                            # strategy='k' のときの k 値
    recon_only_masked: bool = True             # マスクノードのみ再構成損失を計算するか
    verbose: bool = True
    log_interval: int = 20                      # ログ出力間隔（ステップ数）

#def loss_reconstruction(x_hat: torch.Tensor, x_true: torch.Tensor, out: dict) -> torch.Tensor:#loss_weightなし
def loss_reconstruction(x_hat: torch.Tensor, x_true: torch.Tensor, out: dict,loss_weight: Optional[torch.Tensor] = None) -> torch.Tensor:#loss_weight有
    """
    再構成損失を計算する。
    各特徴量ごとに適切な損失関数を適用し、必要に応じてマスクを考慮する。
    Args:
        x_hat (torch.Tensor): モデルの再構成出力 [B, D]。
        x_true (torch.Tensor): 正解データ [B, D]。
        out (dict): モデルの追加出力情報（マスクインデックスなど）。
    Returns:
        torch.Tensor: 平均再構成損失（スカラー）。
    なお、各特徴量の損失関数は以下の通り：

    """
    B, D = x_hat.shape# バッチサイズと特徴量次元数を予測値の形状から取得
    # ★修正: マスクされたノードのみ損失を計算するよう、全ゼロで初期化してから選ばれたノードを1に設定
    target_mask = torch.zeros(B, D, device=x_hat.device)# 再構成損失を計算する際のマスクを初期化（全て0で初期化）
    if "recon_target_idx" in out:
        idx = out["recon_target_idx"]  # ノード（行）インデックス
        target_mask[idx, :] = 1.0    # マスクされたノード(idx)だけ損失を計算するように1を設定
    else:
        target_mask[:, :] = 1.0    # マスクがない場合は全ノード全特徴量で損失を計算

    # feature indices
    deg_index = 0
    depth_index = 1
    mass_idx = 2
    #joint_idx_start, joint_idx_end = 3, 8
    joint_idx_start, joint_idx_end = 3, 6#ほんとは6種類だけどdatasetに出てこないからrevolute,continuous,prismatic,fixedの4種類に限定
    rotate_joint_idx = 3#revoluteジョイントのindex
    axis_idx_start, axis_idx_end = 7, 9
    origin_idx_start, origin_idx_end = 10, 12
    #quat_idx_start, quat_idx_end = 13, 16
    rot6d_idx_start, rot6d_idx_end = 13, 18
    #moveable_idx = 15
    #rot_width_idx = 16
    #rot_lower_idx = 17
    #rot_upper_idx = 18

    # 各特徴ごとに同じshape (B, feature_dim) の損失を求める
    #dataloaderはバッチ処理の際に複数のグラフを縦に連結する
    #なのでBはバッチサイズではなく，バッチ内の全ノード数になる
    losses = torch.zeros_like(x_hat)

    #deg,depth,massはスカラーなのでL1loss(MAE:平均絶対誤差)
    losses[:, deg_index] = F.l1_loss(x_hat[:, deg_index], x_true[:, deg_index], reduction='none')    
    losses[:, depth_index] = F.l1_loss(x_hat[:, depth_index], x_true[:, depth_index], reduction='none')
    losses[:, mass_idx] = F.l1_loss(x_hat[:, mass_idx], x_true[:, mass_idx], reduction='none')

    #jointの種類はone-hotなのでcross-entropy loss(分類問題に使うloss.scoreを確率に正規化してから使うlossを計算する)
    losses[:, joint_idx_start:joint_idx_end+1] = F.cross_entropy(
        x_hat[:, joint_idx_start:joint_idx_end+1],
        x_true[:, joint_idx_start:joint_idx_end+1].argmax(dim=1),#one-hotの1がindxの何番目かを取得
        reduction='none'
    ).unsqueeze(1).expand(-1, joint_idx_end - joint_idx_start + 1)
    #cross-entropy-lossは各ノードごとに1つの値しか持たないが，それをjoint_idx_startからjoint_idx_endまでの各特徴量に同じ値を入れるためにunsqueezeとexpandを使っている

    # cosine sim.が1になるようにする．(√(x1^2 + x2^2 + x3^2) = 1)
    axis_vec_hat = F.normalize(x_hat[:, axis_idx_start:axis_idx_end+1], eps=1e-6)
    axis_vec_true = F.normalize(x_true[:, axis_idx_start:axis_idx_end+1], eps=1e-6)
    
    # cos類似度 (内積) を計算。形状は [B] となる
    # (内積=|a|・|b|・cosθ, normalize済みなので |a|=|b|=1 → 内積=cosθ)
    cosine_sim = (axis_vec_hat * axis_vec_true).sum(dim=1)

    # 1-cos類似度 を損失とする。形状は [B]
    # (誤差θ=0 のとき 1-cos(0) = 0 となり損失が最小になる)
    cosine_loss = 1 - cosine_sim

    # [B] の損失を [B, 1] に変形
    cosine_loss_unsqueezed = cosine_loss.unsqueeze(1)

    # [B, 1] を [B, 3] (axisの列数) に拡張して代入する
    num_axis_dims = axis_idx_end - axis_idx_start + 1
    losses[:, axis_idx_start:axis_idx_end+1] = cosine_loss_unsqueezed.expand(-1, num_axis_dims)
    
 
    # 回転軸を持たないジョイントの場合、軸の損失を0にする
    # (※注: rotate_joint_idx > 0 は「回転関節」です)
    # 変更後: revolute or continuous の“ときだけ”軸損失を残し、それ以外は0
    rev_idx  = 3
    cont_idx = 4
    pris_idx = 5
    fix_idx  = 6

    #revolute, continuous, prismatic の「いずれか」であるか
    is_moveable_with_axis = (x_true[:, rev_idx] > 0) | (x_true[:, cont_idx] > 0) | (x_true[:, pris_idx] > 0)
    #上記でないノード（= fixed）だけを True にする
    mask_axis = ~is_moveable_with_axis
    losses[:, axis_idx_start:axis_idx_end+1][mask_axis] = 0.

    '''
    # origin (L1損失)
    losses[:, origin_idx_start:origin_idx_end+1] = F.l1_loss(
        x_hat[:, origin_idx_start:origin_idx_end+1], x_true[:, origin_idx_start:origin_idx_end+1], reduction='none'
    )
    '''
    # origin (MSE損失に変更: 長さの誤差に厳しくする)
    losses[:, origin_idx_start:origin_idx_end+1] = F.mse_loss(
        x_hat[:, origin_idx_start:origin_idx_end+1], 
        x_true[:, origin_idx_start:origin_idx_end+1], 
        reduction='none'
    )

    # axisと同様に、内積を取って 1 - dot を損失とする
    # (x_hatはforwardで既にnormalize済み前提だが、念のためここでもしても良い)
    #q_hat = x_hat[:, quat_idx_start:quat_idx_end+1]
    #q_true = x_true[:, quat_idx_start:quat_idx_end+1]
    
    # クォータニオンは q と -q が同じ回転を表す「二重被覆」特性があるため、
    # 単純な内積ではなく、絶対値の内積、または min(|q - q_hat|, |q + q_hat|) を取る必要がある
    # しかし簡易的には 1 - |dot| で向きの反転を許容する
    #dot_prod = (q_hat * q_true).sum(dim=1).abs() # 絶対値をとることで q = -q 問題を回避
    #quat_loss = 1.0 - dot_prod
    
    # 次元拡張して格納
    #quat_loss_unsq = quat_loss.unsqueeze(1)
    #losses[:, quat_idx_start:quat_idx_end+1] = quat_loss_unsq.expand(-1, 4)

    #使ってない特徴量の損失計算はコメントアウト
    #losses[:, moveable_idx] = F.l1_loss(x_hat[:, moveable_idx], x_true[:, moveable_idx], reduction='none')
    #losses[:, rot_width_idx] = F.l1_loss(x_hat[:, rot_width_idx], x_true[:, rot_width_idx], reduction='none')
    #losses[:, rot_lower_idx] = F.l1_loss(x_hat[:, rot_lower_idx], x_true[:, rot_lower_idx], reduction='none')
    #losses[:, rot_upper_idx] = F.l1_loss(x_hat[:, rot_upper_idx], x_true[:, rot_upper_idx], reduction='none')

    # 6D回転表現の損失 (L1損失)
    losses[:, rot6d_idx_start:rot6d_idx_end+1] = F.l1_loss(
        x_hat[:, rot6d_idx_start:rot6d_idx_end+1],
        x_true[:, rot6d_idx_start:rot6d_idx_end+1],
        reduction='none'
    )

    masked_loss_tensor = losses * target_mask  # マスク適用
    # マスク適用 & 平均
    # 2. 特徴量ごとの重みを適用
    if loss_weight is not None:
        # loss_weight (shape [D]) が [B, D] にブロードキャスト（自動拡張）されて乗算される
        if loss_weight.shape[0] == D:
            masked_loss_tensor = masked_loss_tensor * loss_weight
        else:
            # 渡された重みの数と特徴量の数が合わない場合は警告
            print(f"[WARN] loss_weight size ({loss_weight.shape[0]}) != feature dimension ({D}). Weight ignored.")

    # 3. 最終的な平均
    return masked_loss_tensor.mean()


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
         w_tensor = torch.tensor(cfg.loss_weight, device=device)

    total_loss = 0.0
    for step, data in enumerate(loader, start=1):
        data = data.to(device)
        
        #まずはクリーンなデータで学習できるか,なのでノイズ付加はコメントアウト
        #if model.training:
        #    data = apply_noise_augmentation(data)
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
        loss = loss_reconstruction(pred, data.x, out, loss_weight=w_tensor)
        
        #aixisのlossではweightを使っていないのでへんこう
        #loss = loss_reconstruction(pred, data.x, out)

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
        w_tensor = torch.tensor(cfg.loss_weight, device=device)

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
        loss = loss_reconstruction(pred, data.x, out, loss_weight=w_tensor)#  weightを使う場合
        #loss = loss_reconstruction(pred, data.x, out)#weightを使っていない場合
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