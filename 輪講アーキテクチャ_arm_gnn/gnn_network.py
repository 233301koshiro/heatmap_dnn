# masked_tree_autoencoder.py
from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, AttentionalAggregation
from torch_scatter import scatter_mean

# ---------- small MLP（BNなし、LN+Dropoutで安定） ----------
def mlp(sizes: List[int], dropout: float = 0.1) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if i < len(sizes) - 2:
            layers += [nn.ReLU(), nn.LayerNorm(sizes[i+1]), nn.Dropout(dropout)]
    return nn.Sequential(*layers)

# ---------- 木向け 往復パス ----------
class DownUpLayer(nn.Module):
    #Tree_encoderとTree_decoderでmlpの後に使われるこれこそがGNNの本体
    #木構造に沿ったGINConvを使った往復伝播を行う
    def __init__(self, hidden: int, dropout: float = 0.1):
        super().__init__()
        #GINConvは「近傍の特徴を和で集めてMLPに渡す」だけのシンプルなGNNレイヤ
        #downとupの記述が同じだが、パラメータは別々
        self.down = GINConv(mlp([hidden, hidden, hidden], dropout=dropout), train_eps=True)
        self.up   = GINConv(mlp([hidden, hidden, hidden], dropout=dropout), train_eps=True)
        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.dir_emb = nn.Parameter(torch.randn(2, hidden) * 0.02)  # [down, up]

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        #こっちは木構造に沿った往復伝播
        #xはノード埋込み[N,H]
        #edge_indexに従ってmlpしたあとにGINConvで集約
        #入力をそのまま足す(Resnetみたい)
        #dir_embは伝播方向の情報を埋め込むためのパラメータ
        #downは木構造に沿った順向き伝播(edge_indexそのまま)
        #reluで活性化してlnで正規化
        x = self.ln1(F.relu(self.down(x, edge_index) + x + self.dir_emb[0]))
        rev = edge_index[[1, 0]]#エッジの向きを逆にする

        #こっちは木構造に沿った逆向き伝播(revはedge_indexの逆)
        x = self.ln2(F.relu(self.up(x, rev) + x + self.dir_emb[1]))
        return x

# ---------- rootプール（root_index優先、無ければin-degree==0平均、無ければAttention） ----------
#rootのインデックスが与えられたらそこを使う
def root_pool(x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, root_index: Optional[torch.Tensor]) -> torch.Tensor:
    #root_indexが与えられたらそれを使う
    if root_index is not None:
        return x[root_index]  # [B,H]
    
    #root_indexが無ければ、in-degree==0ノードの平均を使う
    N, E = x.size(0), edge_index.size(1)

    # 各ノードのin-degreeを計算
    indeg = torch.zeros(N, device=x.device, dtype=torch.long)
    indeg.scatter_add_(0, edge_index[1], torch.ones(E, device=x.device, dtype=torch.long))
    is_root = (indeg == 0).float()

    # バッチ中にin-degree==0ノードが無いグラフがあれば、Attentionで集約
    #ルートが存在しないグラフがあるかを判定
    if (is_root.sum() == 0) or (scatter_mean(is_root, batch, dim=0) == 0).any():
        attn = AttentionalAggregation(gate_nn=mlp([x.size(1), x.size(1)//2, 1]))
        return attn(x, batch)
    x_root = scatter_mean(x * is_root.unsqueeze(-1), batch, dim=0) / (scatter_mean(is_root, batch, dim=0).unsqueeze(-1) + 1e-6)
    return x_root

# ---------- エンコーダ ----------
#ノード特徴 + マスク情報を木構造に沿って集約し、各ノード埋め込み→グラフ潜在へつなぐ前段の表現づくり。
class TreeEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, num_rounds: int = 2, dropout: float = 0.1):
        super().__init__()
        #入力をhidden次元に変換
        #projはprojectionで射影・次元変換を意味する
        self.in_proj = mlp([in_dim, hidden], dropout=dropout)

        #木構造に沿った往復伝播を複数回行う
        self.layers = nn.ModuleList([DownUpLayer(hidden, dropout=dropout) for _ in range(num_rounds)])

    #順伝播
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        #最初のxは [N, in_dim](バッチ内全ノード, 1nodeの特徴数)maskありなら+1らしい
        h = self.in_proj(x)# [N,H]
        #木構造に沿って往復伝播
        for layer in self.layers:
            h = layer(h, edge_index)
        return h  # [N,H]

# ---------- デコーダ（zをbroadcastし、木MPで展開→元の次元に） ----------
#グラフ潜在を各ノードに展開し、元のノード特徴を再構成する。
class TreeDecoder(nn.Module):
    def __init__(self, anchor_dim: int, hidden: int = 128, num_rounds: int = 2, dropout: float = 0.1, out_dim: int = 19):
        super().__init__()
        # 入力は [anchor, mask_flag, z] を結合してhiddenへ
        
        self.in_proj = mlp([anchor_dim + 1 + hidden, hidden], dropout=dropout)
        self.layers = nn.ModuleList([DownUpLayer(hidden, dropout=dropout) for _ in range(num_rounds)])
        self.out_proj = mlp([hidden, hidden, out_dim], dropout=dropout)

    def forward(self, anchor: torch.Tensor, mask_flag: torch.Tensor, z: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # zを各ノードへbroadcast
        z_node = z[batch]                    # [N,H]
        h = torch.cat([anchor, mask_flag, z_node], dim=1)
        h = self.in_proj(h)
        for layer in self.layers:
            h = layer(h, edge_index)
        x_hat = self.out_proj(h)
        return x_hat  # [N, out_dim]

# ---------- 全体：Masked Graph Autoencoder ----------
#マスク生成→エンコード→プール→デコード→損失ターゲット選択までを一括で行う。
class MaskedTreeAutoencoder(nn.Module):
    def __init__(self, in_dim: int = 19, hidden: int = 128, enc_rounds: int = 2, dec_rounds: int = 2, dropout: float = 0.1, anchor_idx: Optional[slice] = None):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        #
        self.encoder = TreeEncoder(in_dim + 1, hidden, enc_rounds, dropout)   # +1 は mask_flag
        self.decoder = TreeDecoder(
            anchor_dim = (in_dim if anchor_idx is None else (anchor_idx.stop - anchor_idx.start)),
            hidden = hidden, num_rounds = dec_rounds, dropout = dropout, out_dim = in_dim
        )
        self.anchor_idx = anchor_idx  # 復元の“位置手掛かり”に使う列（例：deg/depth/jtype など）

    def forward(self, data: Data, mask_idx: Optional[torch.Tensor] = None, recon_only_masked: bool = True) -> Tuple[torch.Tensor, dict]:
        x, ei = data.x, data.edge_index
        device = x.device
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        root_index = getattr(data, "root_index", None)

        # ---- 入力を作る：mask_flag を特徴に追加。マスクは実値0にする（0パディング想定） ----
        mask_flag = torch.zeros(x.size(0), 1, device=device)
        if mask_idx is not None and mask_idx.numel() > 0:
            x = x.clone()
            x[mask_idx] = 0.0
            mask_flag[mask_idx] = 1.0

        # ---- Encoder：木MPでノード→rootへ集約し、zを得る ----
        #h_encはzを得るための中間表現
        h_enc = self.encoder(torch.cat([x, mask_flag], dim=1), ei)   # [N,H]
        #rootがあるかどうかでzの取り方を変える
        z = root_pool(h_enc, ei, batch, root_index=root_index)       # [B,H]

        # ---- Decoder：zを各ノードにbroadcastして木MPで展開、特徴を再構成 ----
        if self.anchor_idx is None:
            anchor = x
        else:
            anchor = x[:, self.anchor_idx]  # 位置/タイプの手掛かり用
        x_hat = self.decoder(anchor, mask_flag, z, ei, batch)         # [N,in_dim]

        # ---- 損失用の補助情報 ----
        out = {"x_hat": x_hat, "mask_flag": mask_flag, "z": z}
        if recon_only_masked and mask_idx is not None and mask_idx.numel() > 0:
            out["recon_target_idx"] = mask_idx
        return x_hat, out

# ---------- 学習ユーティリティ ----------
@dataclass
class TrainCfg:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    recon_only_masked: bool = True

# 損失計算：マスクノードのみ or 全ノード
#損失は推定値と真値のMSE
#マスクしたやつを再現した値を検査する
#そもそもマスクするかどうかはconfigで指定
def loss_reconstruction(x_hat: torch.Tensor, x_true: torch.Tensor, out: dict) -> torch.Tensor:
    if "recon_target_idx" in out:
        idx = out["recon_target_idx"]
        return F.mse_loss(x_hat[idx], x_true[idx])
    else:
        return F.mse_loss(x_hat, x_true)


# ---------- 学習ループなので一番大きな流れ ----------
def train_one_epoch(model: nn.Module, loader, device, cfg: TrainCfg):
    model.train()#学習モードに切り替え
    opt = getattr(model, "_opt", None)#モデルにオプティマイザがあれば使う
    if opt is None:
        #最適化手法であるAdamは勾配の移動平均を活用して、学習率を自動調整する
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        model._opt = opt
    total = 0.0
    # ミニバッチごとに学習
    for data in loader:
        data = data.to(device)#デバイスにデータを転送

        # 各グラフから1ノードをランダムマスク（0パディング）する
        with torch.no_grad():#勾配計算を無効化してメモリ節約
            # バッチ中の各グラフのノードインデックスを取得
            uniq = torch.unique(data.batch if hasattr(data, "batch") else torch.zeros(data.num_nodes, dtype=torch.long, device=device))
            
            chosen = []# マスクするノードインデックスのリスト
            for b in uniq.tolist():#maskするノードのリストを作成
                idx = (data.batch == b).nonzero(as_tuple=False).view(-1)#そのグラフのノードインデックス
                if idx.numel() > 0:#ノードが存在する場合
                    chosen.append(idx[torch.randint(0, idx.numel(), (1,))].item())#ランダムに1ノード選択してマスクリストに追加

            # マスクノードインデックスのテンソルを作成
            mask_idx = torch.tensor(chosen, device=device, dtype=torch.long) if len(chosen) > 0 else None

        # 順伝播と損失計算
        #TreeAutoencoderのforward関数はここで呼ばれる
        x_hat, out = model(data, mask_idx=mask_idx, recon_only_masked=cfg.recon_only_masked)
        #グラフが再現できているかの損失計算
        loss = loss_reconstruction(x_hat, data.x, out)

        opt.zero_grad()#勾配初期化
        loss.backward()#逆伝播で勾配計算
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)#勾配クリッピング
        opt.step()#パラメータ更新

        total += float(loss.item()) * data.num_graphs#バッチの損失を累積
        #item()はテンソルからPythonの数値を取得するメソッド.pytorchではキャストするときこれ使う

    return total / max(1, len(loader.dataset))#データセット全体の平均損失を返す

# ---------- 評価ループ ----------
@torch.no_grad()#@とはデコレータのこと。関数の前に置くことで、その関数内での勾配計算を無効化する
def eval_loss(model: nn.Module, loader, device, recon_only_masked: bool = True):
    model.eval()#評価モードに切り替え
    total, n = 0.0, 0#ノード数

    #ここの流れはtrain_one_epochと一緒
    for data in loader:
        data = data.to(device)

        #ここもtrain_one_epochと一緒
        with torch.no_grad():
            uniq = torch.unique(data.batch if hasattr(data, "batch") else torch.zeros(data.num_nodes, dtype=torch.long, device=device))
            chosen = []
            for b in uniq.tolist():
                idx = (data.batch == b).nonzero(as_tuple=False).view(-1)
                if idx.numel() > 0:
                    chosen.append(idx[torch.randint(0, idx.numel(), (1,))].item())
            mask_idx = torch.tensor(chosen, device=device, dtype=torch.long) if len(chosen) > 0 else None

        x_hat, out = model(data, mask_idx=mask_idx, recon_only_masked=recon_only_masked)

        
        #グラフが再現できているかの損失計算
        loss = loss_reconstruction(x_hat, data.x, out)
        total += float(loss.item()) * data.num_graphs
        n += data.num_graphs#最終的にnはデータセットのグラフ数になる

        #逆伝播とかはない
    return total / max(1, n)
