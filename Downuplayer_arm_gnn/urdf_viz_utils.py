import networkx as nx
import matplotlib.pyplot as plt
from urdf_core_utils import layered_layout

def draw_graph_png(S: nx.DiGraph, out_png: str, title: str, dpi: int = 100) -> None:
    """
    有向グラフを指定されたパスにPNG画像として保存します。
    レイアウトには階層構造を反映した `layered_layout` が使用されます。
    
    Args:
        S (nx.DiGraph): 描画対象の有向グラフ。
        out_png (str): 出力先PNGファイルのパス。
        title (str): グラフのタイトル。
        dpi (int): 保存時の解像度 (Dots Per Inch)。
    """
    plt.figure(figsize=(8, 6))
    pos = layered_layout(S)
    nx.draw_networkx(S, pos=pos, with_labels=True, arrows=True, node_size=600, node_color='lightblue')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    plt.close()