# arm_likeness_gnn.py
# pip install torch torch-geometric

from dataclasses import dataclass
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
# 変更前
# from torch_geometric.nn import GINEConv, GlobalAttention
# 変更後
from torch_geometric.nn import GINEConv
from torch_geometric.nn.aggr import AttentionalAggregation

from torch_geometric.loader import DataLoader

# ---- ユーティリティ（両方向エッジ化） ----
def make_bidirectional(edge_index: torch.Tensor, edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # edge_index: [2, M], edge_attr: [M, F]
    M, F = edge_attr.size()
    dir_fwd = torch.ones((M, 1), dtype=edge_attr.dtype, device=edge_attr.device)
    dir_rev = -torch.ones((M, 1), dtype=edge_attr.dtype, device=edge_attr.device)

    ei_rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
    e_fwd = torch.cat([edge_attr, dir_fwd], dim=1)
    e_rev = torch.cat([edge_attr, dir_rev], dim=1)

    edge_index_bi = torch.cat([edge_index, ei_rev], dim=1)
    edge_attr_bi  = torch.cat([e_fwd, e_rev], dim=0)
    return edge_index_bi, edge_attr_bi  # [2, 2M], [2M, F+1]

# ---- 小さめMLP ----
def mlp(sizes: List[int], dropout: float = 0.1) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if i < len(sizes) - 2:
            layers += [nn.ReLU(), nn.LayerNorm(sizes[i+1]), nn.Dropout(dropout)]
    return nn.Sequential(*layers)

# ---- モデル本体 ----
class ArmLikenessGNN(nn.Module):
    def __init__(self, in_node: int, in_edge: int, hidden: int = 128, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        # ★注意：ここで in_edge は「元の edge_attr 次元 + 1（方向フラグ）」にしておく
        self.edge_mlp = mlp([in_edge, hidden, hidden], dropout=dropout)
        self.node_in  = mlp([in_node, hidden], dropout=dropout)

        convs, norms = [], []
        for _ in range(n_layers):
            convs.append(GINEConv(nn=mlp([hidden, hidden, hidden], dropout=dropout), train_eps=True))
            norms.append(nn.LayerNorm(hidden))  # ← BatchNorm1d から LayerNorm へ
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)

        # 変更前: GlobalAttention(...)
        # 変更後: AttentionalAggregation を使う
        self.pool = AttentionalAggregation(gate_nn=mlp([hidden, hidden // 2, 1], dropout=dropout))

        self.head = mlp([hidden, hidden // 2, 1], dropout=dropout)

    def forward(self, data: Data) -> torch.Tensor:
        x, ei, ea, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # 両方向化（方向フラグ ±1 を付与）→ ea の次元が +1 される
        ei, ea = make_bidirectional(ei, ea)

        e = self.edge_mlp(ea)
        x = self.node_in(x)

        for conv, ln in zip(self.convs, self.norms):
            res = x
            x = conv(x, ei, e)
            x = F.relu(x)
            x = ln(x)      # ← LayerNorm
            x = x + res

        g = self.pool(x, batch)             # [K, hidden]  ※空グラフはそもそも g に現れない
        logit = self.head(g).squeeze(-1)    # [K]
        return logit


# ---- 参考：学習ループの最小骨格 ----
@dataclass
class TrainCfg:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    pos_weight: float = 1.0   # 正例が少ないなら>1に

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
        logit = model(data)                 # [K]  (K <= B になりうる)
        if logit.numel() == 0:
            # まれに全件が空グラフで K=0 のことがあるならスキップ
            continue

        # 実在したグラフIDだけ y を拾って K と整合させる
        present = torch.unique(data.batch)  # 例: tensor([0,1,3,5]) のように欠番あり
        y = data.y.to(device).float()[present]  # [K]

        loss = crit(logit, y)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total += float(loss.item()) * int(present.numel())
    return total / len(loader.dataset)


@torch.no_grad()
def eval_loss(model, loader, device):
    model.eval()
    crit = nn.BCEWithLogitsLoss()
    total = 0.0
    for data in loader:
        data = data.to(device)
        logit = model(data)
        y = data.y.view_as(logit).float()
        loss = crit(logit, y)
        total += float(loss.item()) * data.num_graphs
    return total / len(loader.dataset)
