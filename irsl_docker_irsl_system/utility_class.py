#! /usr/bin/env python3
import time
import torch
import numpy as np
import cv2
import argparse
import networkx as nx
from scipy.spatial import distance
import matplotlib.pyplot as plt
import os
from scipy import stats

# 配列の表示を省略せずに全て表示する
np.set_printoptions(threshold=np.inf)

#自分で作った、画像に関する関数が入ったクラス
#似たような関数をクラス化してメソッドにしたものユーティリティクラスという
#ユーティリティクラスは__initやselfがそこまでいらない
class Mycv:
    @staticmethod
    def draw_local_max_image(heatmap_tf):#入力画像の上に推定グラフを描画
        # 画像の大きさを適切に設定
        height, width = 512, 512
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # オブジェクトポイントを青で描画
        for point in heatmap_tf:
            #print(f"Point: {point}, Shape: {np.shape(point)}")
            cv2.circle(image, tuple(point), 4, (255, 255, 255), -1)  # 白 (BGR)

        return image
    
    def draw_true_image(image,true_joint_points,true_end_points):#正解がある場合、正解の描画
        #trueのとき
        joint_points = true_joint_points[:, ::-1]
        end_points = true_end_points[:, ::-1]

        #推定ヒートマップからつくりたいとき
        #joint_points = list(zip(true_joint_points[0], true_joint_points[1]))
        #end_points = list(zip(true_end_points[0], true_end_points[1]))
        for point in joint_points:
           cv2.circle(image, tuple(point), 8, (255, 0, 0), -1)  # (BGR)

        # オブジェクトポイントを青で描画
        for point in end_points:
            cv2.circle(image, tuple(point), 4, (0, 0, 255), -1)  # (BGR)
        return image
    def make_color_heatmap(gray_img):
        jet_img = cv2.applyColorMap(gray_img, cv2.COLORMAP_JET)
        return jet_img
    
    def get_localmax(pred_sig, kernelsize, th):#局所最大値を求める関数。ヒートマップの中から特に明るい点をジョイントとするための関数
        padding_size = (kernelsize-1)//2
        pred_sig = pred_sig.unsqueeze(0).unsqueeze(0)  # (1, 1, 高さ, 幅)
        max_v = torch.nn.functional.max_pool2d(pred_sig, kernelsize, stride=1, padding=padding_size)
        pred_sig[pred_sig!=max_v] = 0
        pred_sig[pred_sig<th] = 0
        pred_sig[pred_sig!=0] = 1
        return pred_sig
    
    def binarize_to_fill_image(image,flag,input_flag):#塗りつぶし関数            
        # 画像をグレースケールに変換
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if(flag==0):#いつもの用
            # 輪郭を検出
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 同じサイズの空の画像を作成
            filled_image = np.zeros_like(image)
            
            # 輪郭内を塗りつぶす
            for contour in contours:
                # 輪郭の内部を塗りつぶす
                cv2.drawContours(filled_image, [contour], -1, (255,255,255), thickness=cv2.FILLED)
            return filled_image
        
        else:#手書き用

            #背景にどっちが多いかで決める
            # 白に近いピクセルと黒に近いピクセルをカウント
            white_pixels = np.sum(image == 255)  # 白のピクセル数（255）
            black_pixels = np.sum(image == 0)    # 黒のピクセル数（0）
            # 結果を表示
            print(f"白に近いピクセル数: {white_pixels}, 黒に近いピクセル数: {black_pixels}")

            # 反転条件を判定
            if black_pixels > white_pixels:
                print("黒の方が多いのでそのままにします")
                inverted=gray
            else:
                print("白の方が多いので反転します")
                inverted=cv2.bitwise_not(gray)

                # 4. モード値を計算
            mode_result = stats.mode(gray.flatten(), keepdims=False)  # モードを計算
            mode_value = float(mode_result[0])  # スカラー値として扱うために float に変換
            print(f"Background mode value: {mode_value}")
            # 二値化（閾値127を基準に黒か白にする）
            _, binary = cv2.threshold(inverted, mode_value, 255, cv2.THRESH_BINARY)
            # 輪郭を検出
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 輪郭内を塗りつぶす画像を作成（filled_image）
            filled_image = np.zeros_like(image)
            for contour in contours:
                cv2.drawContours(filled_image, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

            if(input_flag==0):#輪郭モード
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # 輪郭を描画するために全て黒の画像を作成
                input_image = np.zeros_like(binary)
                
                # 検出した輪郭を塗りつぶし
                cv2.drawContours(input_image, contours, -1, (255),2)
                return filled_image,input_image
            
            elif(input_flag==1):#元画像モード
                # 元画像のデザインを保持しつつ、背景を黒くする画像を作成（masked_image）
                mask = np.zeros_like(image)
                cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)  # 輪郭内を白にする
                masked_image = cv2.bitwise_and(image, mask)  # 元画像とマスクを論理積

                return filled_image, masked_image



            



    def ensure_same_dimensions(image1, image2):#画像サイズを揃えてくれる関数
        # Resize image2 to match the dimensions of image1 if necessary
        if image1.shape[:2] != image2.shape[:2]:
            image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
        return image2
    
    def draw_debug_image(G, object_points):#入力画像の上に推定グラフを描画
        # 画像の大きさを適切に設定
        height, width = 512, 512
        image = np.zeros((height, width, 3), dtype=np.uint8)

        for point in object_points:
            cv2.circle(image, tuple(point), 2, (255, 255, 255), -1)  # 白 (BGR)

        #ジョイントリンクを描画
        for start_node, end_node in G.edges:
            start_pos = G.nodes[start_node]['pos']  
            end_pos = G.nodes[end_node]['pos']      
            if start_pos is not None and end_pos is not None:
                start = tuple(start_pos)  
                end = tuple(end_pos)     
                cv2.line(image, start, end, (0, 255, 0), 3)  # 緑の線を描画

        #エンドエフェクタリンクを描画
        for start_node, end_node in G.edges:
            #forでとりだしたものはただの座標のタプル
            #グラフはノードの属性の値をそのままキーとしてアクセスできる
            #わけではなく今回はノードの名前が座標からつけられているから
            s_node=G.nodes[start_node]
            e_node=G.nodes[end_node]
            if(s_node['type']=="endeffector" or e_node['type']=="endeffector" ):
                start_pos = G.nodes[start_node]['pos']  # 始点ノードの座標
                end_pos = G.nodes[end_node]['pos']      # 終点ノードの座標
                if start_pos is not None and end_pos is not None:
                    start = tuple(start_pos)  # 始点
                    end = tuple(end_pos)      # 終点
                    cv2.line(image, start, end, (255, 0, 255), 1)  # 紫の線を描画
                    #cv2.line(image, start, end, (255, 255, 255), 3)  # 論文用

        # 各ノードを訪問し、'endeffector' なら赤、'joint' なら青で描画
        for node, attr in G.nodes(data=True):
            pos = attr.get('pos')
            node_type = attr.get('type')

            if pos is not None:
                color = (0, 0, 255) if node_type == 'endeffector' else (255, 0, 0)
                #color = (255, 255, 255) if node_type == 'endeffector' else (255, 0, 0)#論文用
                cv2.circle(image, pos, 6, color, -1)  # エンドエフェクタなら赤、ジョイントなら青
                # オブジェクトポイントを青で描画

        
        return image
    


#座標関係のクラス  
class Coordinate:
    @staticmethod

    
    def create_set(array,flag):#座標の一致を調べるために座標のsetをつくる
        # 1の座標をセットに格納
        object_set = set()
        length=len(array)
        for i in range(length):
            object_set.add(array[i][0]+512*array[i][1])
            if(flag=='joint'):
                    object_set.add( (array[i][0]+1)+ 512*(array[i][1]-1))
                    object_set.add( (array[i][0]+1)+ 512*(array[i][1]))
                    object_set.add( (array[i][0]+1)+ 512*(array[i][1]+1))
                    object_set.add( (array[i][0])+ 512*(array[i][1]+1))
                    object_set.add( (array[i][0])+ 512*(array[i][1]-1))
                    object_set.add( (array[i][0]-1)+ 512*(array[i][1]-1))
                    object_set.add( (array[i][0]-1)+ 512*(array[i][1]))
                    object_set.add( (array[i][0]-1)+ 512*(array[i][1]+1))


        return object_set
    
    
    def get_line_indices2(x1, y1, x2, y2):#リンクの候補座標を返す
        # ベクトル (x2 - x1, y2 - y1) を求める
        dx = x2 - x1
        dy = y2 - y1
        
        # ベクトルの長さを求める
        length = np.sqrt(dx**2 + dy**2)
        
        # ベクトルの正規化 (ベクトルの長さを1にする)
        dx_norm = dx / length
        dy_norm = dy / length
        
        # 始点から距離1進んだ新しい点を計算
        x_new = x1 + dx_norm
        y_new = y1 + dy_norm
        
        return (x_new, y_new)

    
    def generate_points(start,end):#2点を結んだ線上にある座標のリストを返す+各ジョイントの上下左右3マスも入れる
        x2, y2 = end
        points = [] 
         
        points.append(list(start))# 始点を最初の点として追加
        current_point = start
        
        # 終点まで進む処理
        while True:
            # 現在の点と終点の距離を計算
            dx = x2 - current_point[0]
            dy = y2 - current_point[1]
            distance_to_end = np.sqrt(dx**2 + dy**2)
            
            # 距離が1未満になったら終了
            if distance_to_end < 1:
                break
            
            # 次の点を取得
            next_point = Coordinate.get_line_indices2(current_point[0], current_point[1], x2, y2)
            points.append(list(next_point))
            
            # 次の点を現在の点とする
            current_point = next_point
        
        # 最後に終点を追加
        points.append(list(end))
        #座標は要素なので整数に
        for i in range(len(points)):
            points[i][0]=int(points[i][0])
            points[i][1]=int(points[i][1])

        return points
        
    
    def check_background_in_line(link_set,object_set,link_points_num,sikiiti):# 補助関数: 線分上に背景があるか確認
        count=len(link_set.intersection(object_set))#ロボット領域とエッジの重なった数
        #print(count/link_points_num)
        # object_points の座標が 80%以上存在すれば False を返す
        if count >= sikiiti * link_points_num:#リンクが長いほどこの値が大きくなるから釣り合いはとれてる
            return True,count/link_points_num

        
        return False,count/link_points_num
    
    
#グラフ関係のクラス
class Graph:
    @staticmethod
       
    def save_graph_image(G, output_path, pos_flag):  # グラフ保存関数
        # Create a dictionary of positions from node attributes
        pos = nx.get_node_attributes(G, 'pos')
        if 'Root' not in pos:  # 'Root'ノードが位置情報を持っていない場合
            pos['Root'] = (0, 0)  # 任意の位置
            pos['G_Root'] = (10, 10)  # 任意の位置
        
        plt.figure(figsize=(8, 8))
        
        # ノードの色を設定
        color_map = []
        for node in G.nodes():
            node_type = G.nodes[node].get('type', None)  # 'type' 属性を取得
            if node_type == 'joint':
                color_map.append('blue')  # 関節ノードは青
            elif node_type == 'endeffector':
                color_map.append('red')  # エンドエフェクタノードは赤
            else:
                color_map.append('gray')  # デフォルト色

        # グラフ描画
        nx.draw(G, pos, node_size=200, node_color=color_map, with_labels=False)

        # Add edge labels for the weights
        labels = nx.get_edge_attributes(G, 'weight')
        edge_labels = {k: f'{v:.2f}' for k, v in labels.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, label_pos=0.6)

        # ノードの座標をラベルとして表示（ラベル位置を調整）
        node_labels = {node: f'{pos[node]}' for node in G.nodes()}
        offset_pos = {node: (x, y + 0.1) for node, (x, y) in pos.items()}  # ラベルを少し上にずらす
        if pos_flag:
            nx.draw_networkx_labels(G, offset_pos, labels=node_labels, font_size=8, bbox=dict(facecolor='white', edgecolor='none', alpha=0.5))
        else:
            spring_pos = nx.spring_layout(G)
            nx.draw_networkx_labels(G, spring_pos, labels=node_labels, font_size=8, bbox=dict(facecolor='white', edgecolor='none', alpha=0.5))

        # Save the graph image
        plt.savefig(output_path, format='png')
        plt.close()


    
    def remove_longest_edge_in_cycle_undirected(G,tf,object_set,flag):#グラフ内のループを検出し、ループ内の最大の長さを削除
        #print(G.edges())

        try:
            cycle = nx.find_cycle(G)
        except nx.NetworkXNoCycle:
            return G,False
        # サイクルに関わるエッジを出力
        #print("\nサイクルに関わるエッジ:")
        for edge in cycle:
            #print(edge)
            pass
        
        # サイクルを成すノードを出力
        cycle_nodes = set()
        for u, v in cycle:
            cycle_nodes.add(u)
            cycle_nodes.add(v)

        # ノードの座標（pos）を取得
        cycle_node_positions = [G.nodes[node].get('pos', 'No pos') for node in cycle_nodes]
        #print(f"\nサイクルを成すノードの座標: {cycle_node_positions}")
        max_edge = None
        max_length = float('-inf')

        # サイクル内の各エッジの長さを確認
        for u, v in cycle:
            length = G[u][v].get('weight', 1)

            if length > max_length:
                max_length = length
                max_edge = (u, v)
        
        # 最も長いエッジを削除
        if max_edge:
                # ノードの pos を取得
            pos_u = G.nodes[max_edge[0]].get('pos', None)
            pos_v = G.nodes[max_edge[1]].get('pos', None)
            G.remove_edge(*max_edge)
            #print(f"削除されたエッジ: {max_edge}, 長さ: {max_length}")
        '''
        if(flag=='joint'):
            #普段は長いの消すけど
            pos_to_node = {tuple(G.nodes[node].get('pos')): node for node in cycle_nodes if G.nodes[node].get('pos')}
            min_parsent = float('inf')
            min_edge = None
            print(G.nodes)
            # サイクル内の各エッジについて調査
            # サイクルに関わるエッジのみを処理
            for u, v in cycle:
                # エッジの始点と終点の座標を取得
                u_pos = G.nodes[u].get('pos', 'No pos')
                v_pos = G.nodes[v].get('pos', 'No pos')
                
                # 座標が正しく取得できた場合に処理
                if u_pos != 'No pos' and v_pos != 'No pos':
                    # 始点と終点の座標からエッジを作成
                    link_points = Coordinate.generate_points(u_pos, v_pos)
                    link_set = Coordinate.create_set(link_points, flag='link')
                    
                    # オブジェクトの塗りつぶしとリンク候補地の塗りつぶしを比較
                    _, line_parsent = Coordinate.check_background_in_line(link_set, object_set, len(link_points), sikiiti=0.7)
                    
                    # 最も低い%のエッジを更新
                    if line_parsent < min_parsent:
                        min_parsent = line_parsent
                        min_edge = (u_pos, v_pos)

            # 最小の%を持つエッジを削除
            if min_edge:
                # `min_edge`の座標に対応するノード名を取得
                u_pos, v_pos = tuple(min_edge[0]), tuple(min_edge[1])
                u = pos_to_node.get(u_pos)
                v = pos_to_node.get(v_pos)
                print(u,v)
                if u is not None and v is not None and G.has_edge(u, v):  # エッジが存在するか確認
                    G.remove_edge(u, v)
                    print(f"削除されたエッジ: {min_edge} (ノード: {u}, {v}), %: {min_parsent}")
                else:
                    print(f"対応するノードが見つからない、またはエッジが存在しませんでした: {min_edge} (ノード: {u}, {v})")
        '''
        return G,True
   

    def docking_graph(JG, EG,object_set,joint_set):#グラフの結合
        #今のグラフの回数
        for i, node in enumerate(EG.nodes()):
            EG.nodes[node]["type"] = "endeffector"
            EG.nodes[node]["node"] = f"G{i}"

        # JGのノードに"type"属性を"joint"、"node"属性をf"L{i}"として追加
        for i, node in enumerate(JG.nodes()):
            JG.nodes[node]["type"] = "joint"
            JG.nodes[node]["node"] = f"L{i}"
        # ノード名をリラベル
        EG = nx.relabel_nodes(EG, lambda x: f"EG_{x}")
        # ノード名をリラベル
        JG = nx.relabel_nodes(JG, lambda x: f"JG_{x}")
        GG = nx.compose(EG, JG)#これ第一引数のノード名はそのままで、第２引数のグラフのノード名が適当になってる
        print("ただ結合しただけ",GG)
        # 結果の確認
        #print("ノード一覧:", GG.nodes(data=True))
        

        #どっちかのノードがいっこもなかったら何もしない
        if(EG.number_of_nodes()==0 or JG.number_of_nodes()==0):
            print("グラフ作れません")
            exit()

        #追加する工程--------------------------------------------------------------------------------------------------------
        #シン・連結部分 
        EG_pos = nx.get_node_attributes(EG, 'pos')
        JG_pos = nx.get_node_attributes(JG, 'pos')
        for eg_node, eg_pos in EG_pos.items():
            # 末端のジョイントノード（JG の中で次数が 1 のノード）を取得
            terminal_joints = [jg_node for jg_node in JG_pos if len(list(JG.neighbors(jg_node))) == 1]
            
            # 各末端ジョイントノードまでの距離を計算
            distances = {jg_node: np.linalg.norm(np.array(eg_pos) - np.array(JG_pos[jg_node])) for jg_node in terminal_joints}
            # 距離でソートして、最小値から順に処理
            sorted_distances = sorted(distances.items(), key=lambda item: item[1])  # 距離でソート
            print("今回探すエンドエフェクタ",eg_pos)

            # 各ジョイントをループして、0.6ルールをクリアするか試す
            for i in range(len(sorted_distances)):
                closest_jg_node = sorted_distances[i][0]  # 最も近いジョイント
                jg_pos = GG.nodes[closest_jg_node]['pos']
                
                # 近いジョイントとエンドエフェクタが0.6ルールクリアできなかったら次に近いジョイントを試す
                link_points = Coordinate.generate_points(eg_pos, jg_pos)
                link_set = Coordinate.create_set(link_points,'link')
                edge_tf, parsent = Coordinate.check_background_in_line(link_set, object_set, len(link_points), sikiiti=0.4)
                
                if edge_tf:
                    print(f"0.6ルールクリアでジョイントとエンドエフェクタを結んだよ", GG.nodes[closest_jg_node]['pos'], eg_pos, sorted_distances[i][1], parsent)
                    GG.add_edge(eg_node, closest_jg_node)
                    break  # 最初に条件を満たしたジョイントで結んだら終了
                else:
                    print(f"ジョイント {i+1} は0.4ルールをクリアできなかったので次のジョイントを試します", GG.nodes[closest_jg_node]['pos'], eg_pos, sorted_distances[i][1])
                    pass


        j_nodes_list = list(JG.nodes(data=True))
        e_nodes_list = list(EG.nodes(data=True))

        '''
        for e_node, e_attrs in e_nodes_list:
            for j_node, j_attrs in j_nodes_list:
                start = np.array(j_attrs['pos'])
                end = np.array(e_attrs['pos'])
                #始点と終点から距離1の座標を取り続けてsetにまでする
                link_points=Coordinate.generate_points(start,end)
                link_set=Coordinate.create_set(link_points)
                # リンクの候補地に他のジョイントがあるかどうかを確認
                if len(link_set.intersection(joint_set)) > 2:  # 開始・終了点を除いたジョイントが存在するかをチェック
                    print("エンドエフェクタとジョイントつなごうとしたけどブロッカーいるのでスキップ")
                    continue  # 他のジョイントがある場合はリンクを無視し、次のループに移行

                dist = distance.euclidean(start, end)                
                if dist<20:
                    print("ジョイントとエンドエフェクタが近すぎるのでスキップ",e_attrs['pos'],j_attrs['pos'],dist)
                    continue
                elif 15<=dist and dist<=75:
                    print("ジョイントとエンドエフェクタが程良く近いのでクリア",e_attrs['pos'],j_attrs['pos'],dist)
                    GG.add_edge(e_node, j_node, weight=dist)  
                    continue
                elif 90<dist:
                    print("遠すぎる",e_attrs['pos'],j_attrs['pos'],dist)
                    continue

                else:
                    print("ジョイントとエンドエフェクタが遠いので次の0.75ルール次第",e_attrs['pos'],j_attrs['pos'],dist)




                #線上に8割objectpointがあればTrue
                #オブジェクトの塗りつぶしとリンク候補地の塗りつぶしを論理積で比較、その結果が候補地の何割を満たすか
                edge_tf,parsent=Coordinate.check_background_in_line(link_set,object_set,len(link_points),sikiiti=0.75)
                if  edge_tf:
                    GG.add_edge(e_node, j_node, weight=dist)    
                    print("ジョイントとエンドエフェクタで0.75ルールクリア!",e_attrs['pos'],j_attrs['pos'],parsent)      
                else:
                    #print("ジョイントとエンドエフェクタで0.75ルール失敗!",e_attrs['pos'],j_attrs['pos'],parsent)
                    pass
        print("EGとJGを0.8ルールで結んだ後,\n",GG.edges)
        print("↑ループがあるぞ！")
        tf=True
        while(tf):
            GG,tf=Graph.remove_longest_edge_in_cycle_undirected(GG,tf)
        #print("b地点:",GG)
        #1つのジョイントから２つのエンドエフェクタが出ているときは
        # 各エンドエフェクタにそれぞれジョイントを生やす
        # エッジを削除して新しいノードを追加する処理
        print("粛清後,\n",GG.edges)
        
        '''
        #print("エンドエフェクタとジョイント結合後\n",GG.edges)
        multi_node=[]

        for node, attrs in list(GG.nodes(data=True)):
            if attrs.get("type") == "joint":
                connected_nodes = list(GG.neighbors(node))
                count = sum(1 for n in connected_nodes if GG.nodes[n].get("type") == "endeffector")
                
                if count < 2:
                    continue
                else:                    
                    print("このノードは複数のエンドエフェクタにつながってたから中間にノード増やすよ",node)
                    for neighbor in connected_nodes:
                        if(GG.nodes[neighbor]['type']=='joint'):
                            print("お前はjointだからスキップ",neighbor)
                            continue

                        multi_node.append(node)
                                    
                        GG.remove_edge(node, neighbor)

                        node_pos = attrs['pos']
                        neighbor_pos = GG.nodes[neighbor]['pos']
                        mid_pos1 = (int((node_pos[0] + neighbor_pos[0]) / 2), int((node_pos[1] + neighbor_pos[1]) / 2))
                        new_node_id1 = 'JG_'+str(mid_pos1)

                        mid_pos2 = (int((node_pos[0] + mid_pos1[0]) / 2), int((node_pos[1] + mid_pos1[1]) / 2))
                        new_node_id2 = 'JG_'+str(mid_pos2)
                        GG.add_node(new_node_id1, type="joint", pos=mid_pos1)
                        GG.add_node(new_node_id2, type="joint", pos=mid_pos2)
                        GG.add_edge(new_node_id1, neighbor)#neighbor newnode_id1 newnode_id2 node
                        GG.add_edge(new_node_id1, new_node_id2)
                        GG.add_edge(node, new_node_id2)

        #print("複数エンドエフェクタノード増やし後\n",GG.edges)
        #print("c地点:",GG)
        tf=True
        while(tf):
            GG,tf=Graph.remove_longest_edge_in_cycle_undirected(GG,tf,object_set,flag=None)
            
        #print("粛清後\n",GG.edges)
        #削除する工程---------------------------------------------------------------------------------------------------------------
        nodes_list = list(GG.nodes(data=True))
        
        #endeffectorが追加されてない末端のジョイントは削除 
        for node, attrs in nodes_list:
            if attrs['type']=='joint' and len(list(GG.neighbors(node)))==1:#ジョイントかつ末端
                #隣のノードの次数が3以上なら削除
                neighbor = list(GG.neighbors(node))[0]  # 隣接ノードを取得
                a = GG.degree(neighbor)  # ノードの次数を取得
                #print(a)
                if(a>=3):
                    GG.remove_node(node)  
                    print("エンドエフェクタがない末端のジョイントなので削除します",attrs['pos'])   

        print("エンドエフェクタがない末端のジョイント削除後\n",GG.edges)
        

        
        #endeffectorなのに2つ以上のジョイントとつながってるものは削除
        for node, attrs in nodes_list:
            if attrs['type']=='endeffector' and GG.degree(node)>=2:
                   print("endeffectorなのに2つ以上のジョイントとつながってるので削除",node,GG.degree(node))
                   GG.remove_node(node)      
        

        
        #nodes_to_remove = [node for node, degree in dict(GG.degree()).items() if degree == 0]
        #GG.remove_nodes_from(nodes_to_remove)
        #print("削除ノード\n",nodes_to_remove)
        
        tf=True
        while(tf):
            GG,tf=Graph.remove_longest_edge_in_cycle_undirected(GG,tf,object_set,flag=None)

        
        print("色々あったあと",GG)


        return GG,multi_node
    
    def teacher_graph(GG,multi_node):
        NG = nx.Graph()
            # jointとendeffectorのカウンタ
        joint_count = 0
        endeffector_count = 0
        # GGの各ノードを訪問

        for node, attrs in GG.nodes(data=True):
            # 新しいparams辞書を作成
            params = {
                'node': attrs.get('node', 'Unnamed'),  # ノード名、デフォルトは 'Unnamed'
                'type': attrs.get('type', 'unknown'),  # ノードの種類、デフォルトは 'unknown'
                'original_node': node,  # 元のノードIDを追加
                'translation':[0,0,0],
                'relative': True
                
            }
            # 'joint'や'geometry'情報を追加（typeに応じて設定）
            if attrs.get('type') == 'joint':
                connected_nodes = list(GG.neighbors(node))
                count = sum(1 for n in connected_nodes if GG.nodes[n].get("type") == "endeffector")
                if(count):#endeffectorにつながるjointだったら短いリンクにする
                    params['translation']=[0,0,0.5]
                else:
                    params['translation']=[0,0,1.0]
            
            params['node'] = f"L{joint_count}"
            params['type'] ='link'
            params['joint'] = {'type': 'ball'}
            shape = 'cylinder'
            joint_count=joint_count+1
                
            if attrs.get('type') == 'endeffector':
                params['node'] = f"G{endeffector_count}"
                params['type'] ='geom'
                params['geometry'] = {'primitive': 'box', 'args': {'x': 0.2,'y' : 0.5, 'z': 1.0,'color': [1, 0, 0]}}
                params['translation']=  [0,0,0.5]#z/2にする
                shape = 'box'
                endeffector_count=endeffector_count+1

            # NGにノードを追加
            NG.add_node(node, params=params, shape=shape)

            # GGのエッジを参照してNGにエッジを追加
            for edge in GG.edges(data=True):
                node1, node2, edge_attrs = edge
                if node1 in NG and node2 in NG:  # NGに両端のノードが存在する場合
                    NG.add_edge(node1, node2, **edge_attrs)  # エッジ属性も引き継ぐ

        NG.add_node("G_Root", shape="box",
            params={'node': "G_Root", 'type': 'geom', 
                    'geometry': {'primitive': 'box', 
                                    'args': {'x': 0.2,'y' : 0.2, 'z': 0.2,'color': [1, 1, 1]}},
                    'original_node': node
                    }
                    )
        NG.add_node("Root",shape="diamond",
            params={'node': "Root", 'type': 'link', 'joint': {'type': 'root'}})
        NG.add_edge("Root", "G_Root")
        #G_Rootと各エンドエフェクタから最も遠いノードを繋ぐ
        # エンドエフェクタノードを探す
        #G0に対してすべてのLの距離を計算してリストで返す[G0-L0の距離,G0-L1の距離、...]
        #G1に対してすべてのLの距離を計算してリストで返す[G1-L0の距離,G1-L1の距離、...]
        #各リストの同じ要素を足して最も大きい値を持つLとG_Rootをつなぐ
        # ノード名を変更するためのマッピングを作成
        
        for node, data in NG.nodes(data=True):
            if(multi_node==None):
                break
            if(node in multi_node):
                print("複数のエンドエフェクタの元ジョイントは",NG.nodes[node]['params']['node'])
            

       # Gノードを特定
        G_nodes = [n for n, d in NG.nodes(data=True) if d['params']['type'] == 'geom']

        # 全ノード数を取得して sum_distance を初期化
        all_nodes = list(NG.nodes())
        sum_distance = [0] * len(all_nodes)  # すべて 0 で初期化

        # 各Gノードについて最も遠いノードを見つける
        for G in G_nodes:
            # 全ノードへの距離を計算
            distances = nx.single_source_shortest_path_length(NG, G)

            # ノードIDをインデックスにマッピング
            distances_list = [distances.get(node, 0) for node in all_nodes]

            # 各要素を足す
            sum_distance = [x + y for x, y in zip(sum_distance, distances_list)]

        # 距離が最大のノードを特定
        farthest_index = sum_distance.index(max(sum_distance))
        farthest_node = all_nodes[farthest_index]

        # G_Rootと接続
        NG.add_edge("G_Root", farthest_node)
        print(f"各Gノードから最も遠いノード {farthest_node} を G_Root と接続しました。")
        mapping = {}
        for node, data in NG.nodes(data=True):
            # params['node'] が存在する場合に新しい名前を設定
            if 'params' in data and 'node' in data['params']:
                mapping[node] = data['params']['node']
        
        # ノード名をリネーム
        NG = nx.relabel_nodes(NG, mapping)
        

        
        return NG
    
    def draw_graph_with_hierarchy(graph,filepath):#先生のグラフを描画する
        # 描画のためのノードの位置を設定
        pos = pos = nx.spring_layout(graph, k=0.5)  # k を大きくするとノード間が広がる
        
        # ノードを描画
        for node, attrs in graph.nodes(data=True):
            params = attrs.get('params', {})
            shape = attrs.get('shape', 'circle')
            
            # 色と形状を params や type に応じて設定
            color = params.get('geometry', {}).get('args', {}).get('color', [0, 0, 0])  # デフォルトは黒
            color = tuple(color)  # matplotlib 用にタプルに変換
            
            # ノードの形を設定
            if shape == 'box':
                node_shape = 's'  # 四角
            elif shape == 'diamond':
                node_shape = 'D'  # ダイヤモンド
            elif shape == 'cylinder':
                node_shape = 'o'  # 円
            else:
                node_shape = 'o'  # デフォルト円
            
            # ノードの描画
            nx.draw_networkx_nodes(graph, pos,
                                nodelist=[node],
                                node_color=[color],
                                node_shape=node_shape,
                                node_size=500)
        
        # エッジを描画
        nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5)
        
        # ノードラベルを設定（params の 'node' 属性を表示）
        labels = {node: attrs['params'].get('node', node) for node, attrs in graph.nodes(data=True)}
        nx.draw_networkx_labels(graph, pos, labels, font_size=10,font_color='blue')
        
        # 保存または表示
        if filepath:
            plt.savefig(filepath)  # 画像を保存
            print(f"Graph saved to {filepath}")
        else:
            plt.show()  # 画面に表示

        # 描画をクリア（保存後に影響が出ないように）
        plt.clf()

    def create_rooted_directed_tree(G, root_param):#有向グラフ(木)にする
        """
        G: 無向グラフ (NetworkX グラフ)
        root_param: ルートノードのパラメータ（ルートとなるノード）
        """
        # 有向グラフを作成
        DG = nx.DiGraph()

        # ルートノードを追加
        if root_param in G:
            root_attrs = G.nodes[root_param]
            DG.add_node(root_param, **root_attrs)  # 属性を保持してノードを追加

            # ルートノードから他のノードに到達可能な有向エッジを追加
            # DFSまたはBFSでノードを巡回
            visited = set()  # 訪問済みノード
            def dfs(node):
                visited.add(node)
                for neighbor in G.neighbors(node):
                    if neighbor not in visited:
                        # 無向グラフから有向エッジを追加
                        DG.add_edge(node, neighbor)
                        # ノードの属性も保持
                        DG.add_node(neighbor, **G.nodes[neighbor])
                        dfs(neighbor)

            # ルートノードからDFSを開始
            dfs(root_param)
        
        else:
            raise ValueError(f"指定されたルートノード {root_param} はグラフに存在しません。")

        return DG


    def new_create_graph_from_points(heat_array, object_array,flag):#ヒートマップからグラフを作成する
        #ジョイントやオブジェクトのある座標のリストを作成
        #array=実際の値,points=その実際の値がある座標
        object_points = np.array(np.where(object_array != 0)).T[:, ::-1]  # Coordinates of non-zero pointsj
        heat_points = np.array(np.where(heat_array != 0)).T[:, ::-1]  # Coordinates of non-zero points
        object_set=Coordinate.create_set(object_points,flag='object')#objectがある座標をsetに
        
        G = nx.Graph()
        delete_indices = []
        if(flag=='endeffector'):
            print("endffector_points:",heat_points)
            #エンドエフェクタは輪郭のギリギリの場所にあるから、これだとわんちゃんうまくいかないことがありそう
            #エッジとかいらないけど座標とoutlineの中にあるかだけ見たほうがいい
            '''
            for i in range(len(heat_points)):
                #エンドエフェクタの座標セット
                endeffector_set= {heat_points[i][0] + heat_points[i][1] * 512}
                if endeffector_set.intersection(object_set):#エンドエフェクタがロボット領域にあるなら
                    pass
                else:
                    delete_indices.append(i)#消去するインデックスのリスト
                    print("一個消したよ")
            endeffector_points = np.delete(heat_points, delete_indices, axis=0)
            '''

            endeffector_points=heat_points
            for (x,y) in endeffector_points:
                G.add_node((x, y), pos=(x, y), type='endeffector')

            return G,endeffector_points.T


        elif(flag=='joint'):
            print("joint_points:",heat_points)
            #ロボット領域の外にあるジョイントを削除
            #ジョイント同士にすべて線を引いたとしたときの
            #その時の各リンクの各座標を覚える
            #リンク候補地にジョイントがあれば除く
            #リンクの座標が8割ロボット領域上にあればエッジを追加
            #孤立ノードの削除
            #グラフ内でループになっている箇所の削除
            #リンクをいくつか消しているので、all_link_pointの再計算
            #論文に乗せるヒートマップから局所最大値をとったもの

            joint_points=heat_points
            # 削除するインデックスを収集
            for i in range(len(joint_points)):
                a = {joint_points[i][0] + joint_points[i][1] * 512}
                if a.intersection(object_set):
                    #print("ok")
                    pass
                else:
                    delete_indices.append(i)
            # 削除対象の joint_points を取得
            deleted_joints = [joint_points[i] for i in delete_indices]

            print(f"ロボット領域にないjointを削除します: {deleted_joints}")

            #ロボット領域にないジョイントを一括削除
            joint_points = np.delete(joint_points, delete_indices, axis=0)        
            joint_set=Coordinate.create_set(joint_points,flag='joint')
            # enumerate()はインデックス番号と座標値を同時に取り出すもの
            for idx, coord in enumerate(joint_points):
                G.add_node(idx, pos=tuple(coord), type='joint')


            G_complete = nx.complete_graph(len(G.nodes))
            # 元のノード属性をコピー
            for node, attrs in G.nodes(data=True):  # ノードとその属性を取得
                G_complete.nodes[node].update(attrs)  # 属性を更新

            # G を更新
            G = G_complete
            # 各エッジに重みを追加
            for u, v in G.edges():
                    # u, v はノード番号なので、それに対応する座標を取得
                pos_u = G.nodes[u]['pos']  # u の座標
                pos_v = G.nodes[v]['pos']  # v の座標
                G[u][v]['weight'] = distance.euclidean(pos_u, pos_v)
            print("set,point",len(joint_set),len(joint_points))
            # 2重ループだけど探索範囲が三角
            for i in range(len(joint_points)):
                #near_flag=False

                for j in range(i + 1, len(joint_points)):
                    
                    start = np.array(joint_points[i])
                    end = np.array(joint_points[j])

                    dist = distance.euclidean(start, end)
                    if(dist<30):
                        print("joint同士が近すぎるから繋がねーわ",start,end,dist)
                        #near_flag=True#近すぎflagがオン
                        continue
                    else:
                        pass
                    #始点と終点から距離1の座標を取り続けてsetにまでする
                    link_points=Coordinate.generate_points(start,end)
                    link_set=Coordinate.create_set(link_points,flag='link')
                    
                    # リンクの候補地に他のジョイントがあるかどうかを確認
                    if len(link_set.intersection(joint_set)) > (2*9) :  # 開始・終了点を除いたジョイントが存在するかをチェック
                        print("2点を結んだら他のジョイント当たるわ",start,end,len(link_set.intersection(joint_set)))
                        #flag=True
                        G.remove_edge(i, j)  # エッジを削除
                        continue  # 他のジョイントがある場合はリンクを無視し、次のループに移行

                    #線上に8割objectpointがあればTrue
                    #オブジェクトの塗りつぶしとリンク候補地の塗りつぶしを論理積で比較、その結果が候補地の何割を満たすか
                    line_tf,parsent=Coordinate.check_background_in_line(link_set,object_set,len(link_points),sikiiti=0.7)
                    if  line_tf:
                        G.add_edge(i, j, weight=dist)                    
                        #print("ジョイント同士で0.8ルールクリア! %,dist：",start,end,parsent,int(dist))      
                    else:
                        #print("ジョイント同士で0.8ルール失敗!",start,end,parsent)   
                        G.remove_edge(i, j)  # エッジを削除 
                        pass              
                    

            '''
            nodes_to_remove = [node for node, degree in dict(G.degree()).items() if degree == 0]
            G.remove_nodes_from(nodes_to_remove)
            '''
  
            tf=True
            while(tf):
                G,tf=Graph.remove_longest_edge_in_cycle_undirected(G,tf,object_set,flag="joint")


            return G,object_points,joint_set,object_set,joint_points.T
    
    

        

        