# arm_likeness_gnn.py — edge_attrなし / ノードのみ特徴版
from dataclasses import dataclass
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINConv
from torch_geometric.nn.aggr import AttentionalAggregation

# ---- 小さめMLP（BatchNorm→LayerNormでバッチ=1も安全） ----
def mlp(sizes: List[int], dropout: float = 0.1) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if i < len(sizes) - 2:
            layers += [nn.ReLU(), nn.LayerNorm(sizes[i+1]), nn.Dropout(dropout)]
    return nn.Sequential(*layers)

# 追加: どの列が何かを固定（urdf_graph_utils の順序に合わせる）
# [deg, depth, mass] + [jtype_onehot(6)] + [axis(3)] + [origin(3)] + [movable, width, lower, upper]
IDX_DEG = 0
IDX_DEPTH = 1
IDX_MASS = 2
IDX_JTYPE_START, IDX_JTYPE_END = 3, 9      # 3..8
IDX_AXIS = slice(9, 12)                    # 9,10,11
IDX_ORG  = slice(12, 15)                   # 12,13,14
IDX_TAIL = slice(15, 19)                   # 4つ

class ArmLikenessGNN(nn.Module):
    def __init__(self, in_node: int, hidden: int = 128, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        # 小MLPでaxis/originをベクトルとして埋め込み（“タプルのまとまり”を学習）
        self.axis_mlp   = mlp([3, 16, 16], dropout=dropout)
        self.origin_mlp = mlp([3, 16, 16], dropout=dropout)

        # 残り特徴（axis/origin以外）を入れる前段
        # 入力次元 = (in_node - 3 - 3) + 16 + 16
        self.node_in = mlp([(in_node - 6) + 32, hidden], dropout=dropout)

        convs, norms = [], []
        for _ in range(n_layers):
            convs.append(GINConv(mlp([hidden, hidden, hidden], dropout=dropout), train_eps=True))
            norms.append(nn.LayerNorm(hidden))
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)

        self.pool = AttentionalAggregation(gate_nn=mlp([hidden, hidden // 2, 1], dropout=dropout))
        self.head = mlp([hidden, hidden // 2, 1], dropout=dropout)

    def forward(self, data: Data) -> torch.Tensor:
        x, ei = data.x, data.edge_index
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        axis   = x[:, IDX_AXIS]   # [N,3]
        origin = x[:, IDX_ORG]    # [N,3]
        others_left  = x[:, :IDX_AXIS.start]      # [N, 9] -> 3+6
        others_right = x[:, IDX_TAIL]             # [N, 4]
        others = torch.cat([others_left, others_right], dim=1)  # [N, 13]

        # ベクトルとしてのまとまりを保持したまま非線形埋め込み
        axis_z   = self.axis_mlp(axis)       # [N,16]
        origin_z = self.origin_mlp(origin)   # [N,16]

        x = torch.cat([others, axis_z, origin_z], dim=1)  # [N, 13+16+16 = 45]
        x = self.node_in(x)

        for conv, ln in zip(self.convs, self.norms):
            res = x
            x = conv(x, ei)
            x = ln(F.relu(x))
            x = x + res

        g = self.pool(x, batch)
        logit = self.head(g).squeeze(-1)
        return logit


# ---- 学習/評価ヘルパ（既存のまま） ----
@dataclass
class TrainCfg:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    pos_weight: float = 1.0

def train_one_epoch(model, loader, device, cfg: TrainCfg):
    model.train()
    opt = getattr(model, "_opt", None)
    if opt is None:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        model._opt = opt
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg.pos_weight], device=device))
    total = 0.0

    for data in loader:
        data = data.to(device)
        logit = model(data)  # [K]
        if logit.numel() == 0:
            continue
        present = torch.unique(data.batch)
        y = data.y.to(device).float()[present]  # [K]
        loss = crit(logit, y)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total += float(loss.item()) * data.num_graphs
    return total / max(1, len(loader.dataset))

@torch.no_grad()
def eval_loss(model, loader, device):
    model.eval()
    crit = nn.BCEWithLogitsLoss()
    total = 0.0
    n = 0
    for data in loader:
        data = data.to(device)
        logit = model(data)
        if logit.numel() == 0:
            continue
        present = torch.unique(data.batch)
        y = data.y.to(device).float()[present]
        loss = crit(logit, y)
        total += float(loss.item()) * data.num_graphs
        n += data.num_graphs
    return total / max(1, n)
