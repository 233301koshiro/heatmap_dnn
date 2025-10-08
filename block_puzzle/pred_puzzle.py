#!
from multiprocessing import Pool
import os
import time
import numpy as np
import math
import yaml
import random
import argparse
import itertools
from scipy.spatial.transform import Rotation

import cv2

import torch

# 黒背景画像を作成（高さ480, 幅640, チャンネル3）
img_blank = np.zeros((512, 512, 3), dtype=np.uint8)

# コピーを作成して描画用に使う
img_draw = img_blank.copy()

#組み合わせのパターン数を計算する
def convination(n, r):
    # if (n, r) in memo_conv:
    #     return memo_conv[(n, r)]
    if r == 0 or n == r:
        return 1
    elif r == 1:
        return n
    elif r > n-r:
        return convination(n, n-r)
    else:
        ret = convination(n-1, r-1)+convination(n-1, r)
        # memo_conv[(n, r)] = ret
        return ret

# 順列のパターン数を計算する
def permutation(n, r):
    ret = 1
    for i in range(n-r+1, n+1):
        ret *= i
    return ret


# 局所最大値を取得する
def get_localmax(pred_sig, kernelsize, th=0.3):
    """Get localmax"""
    padding_size = (kernelsize - 1) // 2
    max_v = torch.nn.functional.max_pool2d(
        pred_sig, kernelsize, stride=1, padding=padding_size
    )
    ret = pred_sig.clone()
    ret[pred_sig != max_v] = 0
    ret[ret < th] = 0
    ret[ret != 0] = 1
    return ret


# 頂点を法線に対して右回りになるように並べ替える
def sort_vertex(model_vertex, plane_vector, plane_normal):
    """Get sorted vertex"""
    mean_vertex = np.mean(model_vertex, axis=0)
    tmp_vec = model_vertex - mean_vertex
    tmp_vec = tmp_vec / np.linalg.norm(tmp_vec, axis=1, keepdims=True)
    point_args = np.arctan2(
        np.dot(tmp_vec, plane_vector),
        np.dot(np.cross(tmp_vec, plane_vector), plane_normal),
    )
    sorted_idxs = np.argsort(point_args)
    model_vertex = model_vertex[sorted_idxs]
    return model_vertex


#ブロックのyamlファイルから読み込んだ凹凸頂点座標リスト(1*N)を(N*3)に変換する
#
def gen_fitting_model(blockdata, issort=True):
    """Generrate fitting model from config"""
    ret = {}
    eps = 1e-3

    for block in blockdata:
        convex_vertex_pos = np.array(
            block["convex_vertex_pos"]).reshape((-1, 3))
        concave_vertex_pos = np.array(
            block["concave_vertex_pos"]).reshape((-1, 3))

        model_vertexs = []
        z_max = np.max(convex_vertex_pos[:, 2])
        plane_normal = np.array([0, 0, 1])
        plane_vector = np.array([1, 0, 0])
        target_ids = np.where(convex_vertex_pos[:, 2] > (z_max - eps))[0]
        model_vertex = convex_vertex_pos[target_ids]
        if issort:
            model_vertexs.append(sort_vertex(
                model_vertex, plane_vector, plane_normal))
        else:
            model_vertexs.append(model_vertex)

        z_min = np.min(convex_vertex_pos[:, 2])
        plane_normal = np.array([0, 0, -1])
        plane_vector = np.array([1, 0, 0])
        target_ids = np.where(convex_vertex_pos[:, 2] < (z_min + eps))[0]
        model_vertex = convex_vertex_pos[target_ids]
        if issort:
            model_vertexs.append(sort_vertex(
                model_vertex, plane_vector, plane_normal))
        else:
            model_vertexs.append(model_vertex)

        x_max = np.max(convex_vertex_pos[:, 0])
        plane_normal = np.array([1, 0, 0])
        plane_vector = np.array([0, 1, 0])
        target_ids = np.where(convex_vertex_pos[:, 0] > (x_max - eps))[0]
        model_vertex = convex_vertex_pos[target_ids]
        if issort:
            model_vertexs.append(sort_vertex(
                model_vertex, plane_vector, plane_normal))
        else:
            model_vertexs.append(model_vertex)

        x_min = np.min(convex_vertex_pos[:, 0])
        plane_normal = np.array([-1, 0, 0])
        plane_vector = np.array([0, 1, 0])
        target_ids = np.where(convex_vertex_pos[:, 0] < (x_min + eps))[0]
        model_vertex = convex_vertex_pos[target_ids]
        if issort:
            model_vertexs.append(sort_vertex(
                model_vertex, plane_vector, plane_normal))
        else:
            model_vertexs.append(model_vertex)

        y_max = np.max(convex_vertex_pos[:, 1])
        plane_normal = np.array([0, 1, 0])
        plane_vector = np.array([0, 0, 1])
        target_ids = np.where(convex_vertex_pos[:, 1] > (y_max - eps))[0]
        model_vertex = convex_vertex_pos[target_ids]
        if issort:
            model_vertexs.append(sort_vertex(
                model_vertex, plane_vector, plane_normal))
        else:
            model_vertexs.append(model_vertex)

        y_min = np.min(convex_vertex_pos[:, 1])
        plane_normal = np.array([0, -1, 0])
        plane_vector = np.array([0, 0, 1])
        target_ids = np.where(convex_vertex_pos[:, 1] < (y_min + eps))[0]
        model_vertex = convex_vertex_pos[target_ids]
        if issort:
            model_vertexs.append(sort_vertex(
                model_vertex, plane_vector, plane_normal))
        else:
            model_vertexs.append(model_vertex)

        # print(model_vertexs)
        # bounding_point = [x_min, x_max, y_min, y_max, z_min, z_max]
        bounding_point = np.array(
            [
                [x_min, y_min, z_min],
                [x_max, y_min, z_min],
                [x_max, y_max, z_min],
                [x_min, y_max, z_min],
                [x_min, y_min, z_max],
                [x_max, y_min, z_max],
                [x_max, y_max, z_max],
                [x_min, y_max, z_max],
            ]
        )
        ret[block["name"]] = {
            "model_vertexs": model_vertexs,
            "convex_vertex_pos": convex_vertex_pos,
            "concave_vertex_pos": concave_vertex_pos,
            "bounding_point": bounding_point,
        }
    return ret
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def visualize_block_model(block_model, title="Block Model"):
    """
    block_model: gen_fitting_model が返す辞書のうち、特定のブロック名の値
      {
        "model_vertexs": [ 面0の頂点配列, 面1の頂点配列, … ],
        "convex_vertex_pos": (N×3) ndarray,
        "concave_vertex_pos": (M×3) ndarray,
        "bounding_point": (8×3) ndarray,
      }
    """
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    # 凸頂点 (青点)
    cv = block_model["convex_vertex_pos"]
    ax.scatter(cv[:,0], cv[:,1], cv[:,2], c='b', marker='o', label='convex')

    # 凹頂点 (赤点)
    hv = block_model["concave_vertex_pos"]
    if hv.size:
        ax.scatter(hv[:,0], hv[:,1], hv[:,2], c='r', marker='^', label='concave')

    # 各面のポリゴン
    for verts in block_model["model_vertexs"]:
        # verts は (k×3) array：面上の頂点を一筆書き順にソート済み
        poly = Poly3DCollection([verts], alpha=0.3, edgecolor='k')
        ax.add_collection3d(poly)

    # バウンディングボックスのエッジ
    bp = block_model["bounding_point"]
    # 8頂点の組み合わせで 12 本のエッジを描く
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]
    for i,j in edges:
        xs, ys, zs = zip(bp[i], bp[j])
        ax.plot(xs, ys, zs, c='gray', linewidth=1)

    ax.legend()
    # アスペクト比を等しくする
    max_range = (bp.max(axis=0) - bp.min(axis=0)).max() / 2
    mid = (bp.max(axis=0) + bp.min(axis=0)) / 2
    ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
    ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
    ax.set_zlim(mid[2]-max_range, mid[2]+max_range)
    # 描画をファイルに保存
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)  # メモリ解放

# モデルの頂点を法線に対して右回りになるように並べ替える
def gen_featrue_map(image, model, device, score_ths=[0.5, 0.3], min_convex=6):
    """Generate feature map"""
    height, width = image.shape[0:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    image_rgb = np.transpose(image_rgb, (2, 0, 1))[np.newaxis]
    image_tensor = torch.Tensor(image_rgb)
    pred = (model(image_tensor.to(device)).sigmoid()).detach()
    pred_ch = pred.shape[1]
    pred_convex = pred[:, 0:1].clone()
    pred_concave = pred[:, 1:2].clone()
    if pred_ch == 4:
        pred_hidden_convex = pred[:, 2:3].clone()
        pred_hidden_concave = pred[:, 3:4].clone()
    ret_dict = {}
    for score_th in score_ths:
        pred_convex_lm = get_localmax(pred_convex, 13, score_th)
        pred_concave_lm = get_localmax(pred_concave, 13, score_th)
        # pred_convex_np = (pred_convex.cpu()*255).numpy().astype(np.uint8)[0,0]
        # pred_concave_np = (pred_concave.cpu()*255).numpy().astype(np.uint8)[0,0]
        convex_pos = np.array(
            np.where(pred_convex_lm[0, 0].cpu().numpy() != 0)).T[:, ::-1]
        concave_pos = np.array(
            np.where(pred_concave_lm[0, 0].cpu().numpy() != 0)).T[:, ::-1]
        # TODO 画面周囲には特徴点がないと仮定するとマッチングがよくなるので実施しているがこの仮定良いのか？
        convex_pos = convex_pos[convex_pos[:, 0] != 0]
        convex_pos = convex_pos[convex_pos[:, 0] != width-1]
        convex_pos = convex_pos[convex_pos[:, 1] != 0]
        convex_pos = convex_pos[convex_pos[:, 1] != height-1]
        if len(convex_pos) > min_convex:
            print("a",score_th)
            break
    ret_dict = {
        "pred_convex": pred_convex,  # 凸頂点の予測ヒートマップ（tensor 形式）
        "pred_concave": pred_concave,  # 凹頂点の予測ヒートマップ（tensor 形式）
        "pred_convex_lm": pred_convex_lm,  # 凸頂点のランドマーク予測（局所化されたヒートマップ）
        "pred_concave_lm": pred_concave_lm,  # 凹頂点のランドマーク予測（局所化されたヒートマップ）
        "convex_pos": convex_pos,  # 凸頂点の2D座標のリスト（x, y）タプル形式
        "concave_pos": concave_pos,  # 凹頂点の2D座標のリスト（x, y）タプル形式
    }

    if pred_ch == 4:
        ret_dict["pred_hidden_convex"] = pred_hidden_convex
        ret_dict["pred_hidden_concave"] = pred_hidden_concave

    return ret_dict

# モデルの頂点を法線に対して右回りになるように並べ替える
def eval_score(points, image_width, image_height, score_map, hidden_score_map=None, min_score=1e-2):
    if True:
        min_logscore = math.log10(min_score)
        score_list = []
        for px, py in points:
            if px >= 0 and py >= 0 and px < image_width and py < image_height:
                if hidden_score_map is not None:
                    value = max(float(score_map[py, px]), float(
                        hidden_score_map[py, px]))
                else:
                    value = float(score_map[py, px])
                if value >= min_score:
                    score_list.append(np.log10(value))
                else:
                    score_list.append(min_logscore)
            else:
                score_list.append(min_logscore)
        return np.sum(score_list)
    else:
        score_list = []
        for px, py in points:
            if px >= 0 and py >= 0 and px < image_width and py < image_height:
                if hidden_score_map is not None:
                    value = max(float(score_map[py, px]), float(
                        hidden_score_map[py, px]))
                else:
                    value = float(score_map[py, px])
                if value >= min_score:
                    score_list.append(value)
                else:
                    score_list.append(min_score)
            else:
                score_list.append(min_score)
        return np.mean(score_list)


# モデルに対してfittingをおこなう
def fit2model(target_model, convex_pos, pred_convex, cam_mat, dist, image_width=512, image_height=512, itr_num=2000, issort=True, use_ransac=True, pred_hidden_convex=None):
    """Fitting Model"""
    model_vertexs = target_model["model_vertexs"]
    convex_vertex_pos = target_model["convex_vertex_pos"]
    # concave_vertex_pos = target_model['concave_vertex_pos']

    min_score = 1e-2
    min_logscore = np.log10(min_score)
    max_score = -1e100
    max_score_dat = {}
    cur_itr_num = 0

    score_map = pred_convex.cpu()[0, 0].numpy()
    if pred_hidden_convex is not None:
        hidden_score_map = pred_hidden_convex.cpu()[0, 0].numpy()
    else:
        hidden_score_map = None
    convex_pos_list = convex_pos.tolist()

    if use_ransac:
        if itr_num <= 0:
            p_img = min(0.8, len(convex_vertex_pos)/len(convex_pos_list))
            itr_num = math.ceil(
                math.log(1-0.995)/math.log(1-p_img**5/convination(len(convex_pos_list), 5)/6))
            # print(p_img, itr_num, len(convex_pos_list))
        for i in range(itr_num):
            # 当てはめる面のサンプル
            model_vertex = random.sample(model_vertexs, 1)[0]
            # 点のサンプル
            sample_dat = random.sample(convex_pos_list, len(model_vertex))
            if issort:
                # サンプリングした点を-zの単位ベクトルに対して右回りに並べなおす
                img_pos = np.array(sample_dat).astype(np.float64)
                rel_img_pos = img_pos - np.mean(img_pos, axis=0)
                rel_img_pos = rel_img_pos / \
                    np.linalg.norm(rel_img_pos, axis=1, keepdims=True)
                cost = np.dot(rel_img_pos, np.array([1, 0]))
                sint = np.dot(
                    np.cross(
                        np.hstack(
                            (rel_img_pos, np.zeros((len(model_vertex), 1)))),
                        np.array([1, 0, 0]),
                    ),
                    np.array([0, 0, -1]),
                )
                t = np.arctan2(cost, sint)
                sort_idx = np.argsort(t)
                img_pos = img_pos[sort_idx]
            else:
                img_pos = np.array(sample_dat).astype(np.float64)

            for j in range(len(img_pos)):
                # 順番を変えながらフィッティングを行う
                if j != 0:
                    img_posd = np.array(
                        img_pos.tolist()[j:] + img_pos.tolist()[:j])
                else:
                    img_posd = img_pos
                # PnPで並進，回転ベクトルを求める
                ret, rvec, tvec = cv2.solvePnP(
                    model_vertex, img_posd, cam_mat, dist)
                # モデルの頂点の投影位置を求める
                point, _ = cv2.projectPoints(
                    convex_vertex_pos, rvec, tvec, cam_mat, dist)

                # モデルの投影位置のヒートマップ値の積を求めたいが，
                # アンダーフローする可能性があるので，ヒートマップの積ではなくlogの和を取る
                point_int = point.astype(np.int64).reshape((-1, 2))
                score_list = []
                for px, py in point_int:
                    if 0 <= px < image_width and 0 <= py < image_height:
                        if hidden_score_map is not None:
                            value = max(float(score_map[py, px]), float(
                                hidden_score_map[py, px]))
                        else:
                            value = float(score_map[py, px])
                        if value >= min_score:
                            score_list.append(np.log10(value))
                        else:
                            score_list.append(min_logscore)
                    else:
                        score_list.append(min_logscore)
                # 出現した中で最大のものを保存する
                score_sum = np.sum(score_list)
                if score_sum > max_score:
                    max_score = score_sum
                    max_score_dat = {
                        "rvec": rvec,
                        "tvec": tvec,
                        "point": point.reshape((-1, 2)),
                    }
                cur_itr_num += 1
    else:
        for model_vertex in model_vertexs:
            for _, sample_dat in enumerate(list(itertools.permutations(convex_pos_list, len(model_vertex)))):
                img_posd = np.array(sample_dat).astype(np.float64)
                model_vertex_np = np.array(model_vertex).astype(np.float64)
                # PnPで並進，回転ベクトルを求める
                ret, rvec, tvec = cv2.solvePnP(
                    model_vertex_np, img_posd, cam_mat, dist)
                if not ret:
                    continue
                if np.linalg.norm(tvec) > 0.5:
                    continue
                # モデルの頂点の投影位置を求める
                point, _ = cv2.projectPoints(
                    convex_vertex_pos, rvec, tvec, cam_mat, dist)

                # モデルの投影位置のヒートマップ値の積を求めたいが，
                # アンダーフローする可能性があるので，ヒートマップの積ではなくlogの和を取る
                point_int = point.astype(np.int64).reshape((-1, 2))
                score_list = []
                for px, py in point_int:
                    if 0 <= px < image_width and 0 <= py < image_height:
                        if hidden_score_map is not None:
                            value = max(float(score_map[py, px]), float(
                                hidden_score_map[py, px]))
                        else:
                            value = float(score_map[py, px])
                        if value >= min_score:
                            score_list.append(np.log10(value))
                        else:
                            score_list.append(min_logscore)
                    else:
                        score_list.append(min_logscore)
                # 出現した中で最大のものを保存する
                score_sum = np.sum(score_list)
                if score_sum > max_score:
                    max_score = score_sum
                    max_score_dat = {
                        "rvec": rvec,
                        "tvec": tvec,
                        "point": point.reshape((-1, 2)),
                    }
                cur_itr_num += 1
    max_score_dat["itr_num"] = cur_itr_num
    return max_score, max_score_dat


def fit2model_p3p(target_model,
                  convex_pos,
                  pred_convex,
                  concave_pos,
                  pred_concave,
                  cam_mat,
                  dist,
                  image_width=512,
                  image_height=512,
                  itr_num=2000,
                  use_ransac=True,
                  bruteforce=False,
                  cut_score_th=-15,
                  pred_hidden_convex=None,
                  eval_type='convination'):
    """Fitting Model"""
    model_vertexs = target_model["model_vertexs"]
    convex_vertex_pos = target_model["convex_vertex_pos"]
    concave_vertex_pos = target_model['concave_vertex_pos']

    min_score = 1e-2
    min_logscore = np.log10(min_score)
    max_score = -1e100
    best_mean_dist = 0
    max_score_dat = {}
    cur_itr_num = 0

    score_map = pred_convex.cpu()[0, 0].numpy()
    if pred_hidden_convex is not None:
        hidden_score_map = pred_hidden_convex.cpu()[0, 0].numpy()
    else:
        hidden_score_map = None
    convex_pos_list = convex_pos.tolist()
    concave_pos_list = concave_pos.tolist()
    convex_vertex_pos_list = convex_vertex_pos.tolist()

    if use_ransac:
        if itr_num <= 0:
            p_img = min(0.8, len(convex_vertex_pos_list)/len(convex_pos_list))
            itr_num = math.ceil(
                math.log(1-0.995)/math.log(1-p_img**3/permutation(len(convex_vertex_pos_list), 3)))
        rvec_list = []
        tvec_list = []
        for i in range(itr_num):
            # 当てはめるモデル点のサンプル
            model_vertex = random.sample(convex_vertex_pos_list, 3)
            # 点のサンプル
            sample_dat = random.sample(convex_pos_list, 3)
            # PnPで並進，回転ベクトルを求める
            model_vertex_np = np.array(model_vertex).astype(np.float64)
            image_vertex_np = np.array(sample_dat).astype(np.float64)
            num, rvecs, tvecs = cv2.solveP3P(model_vertex_np, image_vertex_np,
                                             cam_mat, dist, flags=cv2.SOLVEPNP_P3P)
            rvec_list.extend(rvecs)
            tvec_list.extend(tvecs)

        # # モデルの頂点の投影位置を求める
        for rvec, tvec in zip(rvec_list, tvec_list):
            point, _ = cv2.projectPoints(
                convex_vertex_pos, rvec, tvec, cam_mat, dist)
            point_int = point.astype(np.int64).reshape((-1, 2))
            # score_list = []
            # for px, py in point_int:
            #     if 0 <= px < image_width and 0 <= py < image_height:
            #         if hidden_score_map is not None:
            #             value = max(float(score_map[py, px]), float(
            #                 hidden_score_map[py, px]))
            #         else:
            #             value = float(score_map[py, px])
            #         if value >= min_score:
            #             score_list.append(np.log10(value))
            #         else:
            #             score_list.append(min_logscore)
            #     else:
            #         score_list.append(min_logscore)
            # # 出現した中で最大のものを保存する
            # score_sum = np.sum(score_list)
            score = eval_score(point_int, image_width, image_height,
                               score_map, hidden_score_map, min_score)
            if score > max_score:
                max_score = score
                max_score_dat = {
                    "rvec": rvec,
                    "tvec": tvec,
                    "point": point.reshape((-1, 2)),
                }
            cur_itr_num += 1
    elif bruteforce:
        for _, model_vertex in enumerate(list(itertools.permutations(convex_vertex_pos_list, 3))):
            for _, sample_dat in enumerate(list(itertools.permutations(convex_pos_list, 3))):
                # PnPで並進，回転ベクトルを求める
                model_vertex_np = np.array(model_vertex).astype(np.float64)
                image_vertex_np = np.array(sample_dat).astype(np.float64)
                num, rvecs, tvecs = cv2.solveP3P(model_vertex_np, image_vertex_np,
                                                 cam_mat, dist, flags=cv2.SOLVEPNP_P3P)
                # # モデルの頂点の投影位置を求める
                for rvec, tvec in zip(rvecs, tvecs):
                    point, _ = cv2.projectPoints(
                        convex_vertex_pos, rvec, tvec, cam_mat, dist)
                    point_int = point.astype(np.int64).reshape((-1, 2))
                    score_list = []
                    for px, py in point_int:
                        if px >= 0 and py >= 0 and px < image_width and py < image_height:
                            if hidden_score_map is not None:
                                value = max(float(score_map[py, px]), float(
                                    hidden_score_map[py, px]))
                            else:
                                value = float(score_map[py, px])
                            if value >= min_score:
                                score_list.append(np.log10(value))
                            else:
                                score_list.append(min_logscore)
                        else:
                            score_list.append(min_logscore)
                    # 出現した中で最大のものを保存する
                    score_sum = np.sum(score_list)
                    if score_sum > max_score:
                        max_score = score_sum
                        max_score_dat = {
                            "rvec": rvec,
                            "tvec": tvec,
                            "point": point.reshape((-1, 2)),
                        }
    else:
        convex_pos_score = [float(score_map[p[1], p[0]])
                            for p in convex_pos_list]
        idxs = np.argsort(convex_pos_score)[::-1]

        point_img = np.ones((image_height, image_width), dtype=np.uint8)*255
        for p in convex_pos_list:
            point_img[int(p[1]), int(p[0])] = 0
        dist_img = cv2.distanceTransform(point_img,
                                         distanceType=cv2.DIST_L2,
                                         maskSize=5
                                         )

        
        if eval_type == 'area_size':
            def eval_func(c_p): return abs(np.cross(
                np.array(c_p[1])-np.array(c_p[0]), np.array(c_p[2])-np.array(c_p[0]))[2])/2
        elif eval_type == 'prob':
            def eval_func(c_p): return np.prod(
                [float(score_map[int(p[1]), int(p[0])]) for p in c_p])
        elif eval_type == 'convination':
            def eval_func(c_p): return abs(np.cross(np.array(c_p[1])-np.array(c_p[0]), np.array(
                c_p[2])-np.array(c_p[0]))[2])/2 * np.prod([float(score_map[int(p[1]), int(p[0])]) for p in c_p])

        # convination_points = list(itertools.permutations(np.hstack((np.array(convex_pos_list), np.zeros((len(convex_pos_list),1)))).tolist(), 3))
        # sorted_convination_points = sorted(convination_points, key=eval_func, reverse=True)
        # sorted_convination_points_idx = sorted(list(range(len(convination_points))), key=[eval_func(p) for p in convination_points].__getitem__, reverse=True)

        convination_idxs_list = list(itertools.permutations(
            list(range(len(convex_pos_list))), 3))
        sorted_convination_idxs_list_idx = sorted(list(range(len(convination_idxs_list))),
                                                  key=[eval_func(np.array(
                                                      [convex_pos_list[idx]+[0] for idx in idxs])) for idxs in convination_idxs_list].__getitem__,
                                                  reverse=True)

        for cp_idx in sorted_convination_idxs_list_idx:
            # image_vertex_np = np.array(c_p)[:,:2].astype(np.float64)
            image_vertex_np = np.array(
                [convex_pos_list[p_idx] for p_idx in convination_idxs_list[cp_idx]]).astype(np.float64)
            other_convex_pos_list = [convex_pos_list[i] for i in range(
                len(convex_pos_list)) if i not in convination_idxs_list[cp_idx]]
            for _, model_vertex in enumerate(list(itertools.permutations(convex_vertex_pos_list, 3))):
                model_vertex_np = np.array(model_vertex).astype(np.float64)
                num, rvecs, tvecs = cv2.solveP3P(model_vertex_np, image_vertex_np,
                                                 cam_mat, dist, flags=cv2.SOLVEPNP_P3P)
                # # モデルの頂点の投影位置を求める
                for rvec, tvec in zip(rvecs, tvecs):
                    conv_point, _ = cv2.projectPoints(
                        convex_vertex_pos, rvec, tvec, cam_mat, dist)
                    # conv_point_int = conv_point.astype(np.int64).reshape((-1, 2))
                    if concave_vertex_pos.shape[0] != 0:
                        conc_point, _ = cv2.projectPoints(
                            concave_vertex_pos, rvec, tvec, cam_mat, dist)
                    else :
                        conc_point = np.zeros((0,3))
                    # socore map base evaluation
                    # score = eval_score(
                    #     point_int, image_width, image_height, score_map, hidden_score_map, min_score)

                    # distance base evaluation
                    # score = 0
                    # max_dist = 32
                    # for p in point_int:
                    #     if 0<=p[0]<image_width and 0<=p[1]<image_height:
                    #         score -= min(max_dist, dist_img[p[1], p[0]])
                    #     else:
                    #         score -= max_dist



                    match_cnt = 0
                    dist_list = []

                    # for p in point.reshape((-1, 2)):
                    #     tmp_val = np.sort(np.linalg.norm(np.array(other_convex_pos_list)-p,axis=1))
                    #     if tmp_val[0] < 5 and tmp_val[1]/tmp_val[0] > 20:
                    #         match_cnt +=1
                    #         dist_list.appned(tmp_val[0])

                    for p in other_convex_pos_list:
                        tmp_val = np.sort(np.linalg.norm(
                            conv_point.reshape((-1, 2))-p, axis=1))
                        if tmp_val[0] < 5 and tmp_val[1] > 20:
                            match_cnt += 1
                            dist_list.append(tmp_val[0])

                    # for p in concave_pos_list:
                    #     tmp_val = np.sort(np.linalg.norm(
                    #         conc_point.reshape((-1, 2))-p, axis=1))
                    #     if tmp_val[0] < 5 and tmp_val[1] > 20:
                    #         match_cnt += 1
                    #         dist_list.append(tmp_val[0])
                            
                    score = match_cnt
                    mean_dist = np.mean(dist_list)
                    # print(match_cnt, score, cut_score_th)
                    if score > max_score or (score == max_score and best_mean_dist > mean_dist):
                        max_score = score
                        best_mean_dist = mean_dist
                        max_score_dat = {
                            "rvec": rvec,
                            "tvec": tvec,
                            "point": conv_point.reshape((-1, 2)),
                        }
                    cur_itr_num += 1
                    if max_score > cut_score_th:
                        break
                if max_score > cut_score_th:
                    break
            # print(cur_itr_num)
            if max_score > cut_score_th:
                break

    max_score_dat["itr_num"] = cur_itr_num
    return max_score, max_score_dat

    """
        for _, sample_dat_idx in enumerate(list(itertools.permutations(idxs, 3))):
            for _, model_vertex in enumerate(list(itertools.permutations(convex_vertex_pos_list, 3))):
                # PnPで並進，回転ベクトルを求める
                model_vertex_np = np.array(model_vertex).astype(np.float64)
                image_vertex_np = np.array(
                    [convex_pos_list[idx] for idx in sample_dat_idx]).astype(np.float64)
                num, rvecs, tvecs = cv2.solveP3P(model_vertex_np, image_vertex_np,
                                                 cam_mat, dist, flags=cv2.SOLVEPNP_P3P)
                # # モデルの頂点の投影位置を求める
                for rvec, tvec in zip(rvecs, tvecs):
                    point, _ = cv2.projectPoints(
                        convex_vertex_pos, rvec, tvec, cam_mat, dist)
                    point_int = point.astype(np.int64).reshape((-1, 2))
                    # score_list = []
                    # for px, py in point_int:
                    #     if px >= 0 and py >= 0 and px < image_width and py < image_height:
                    #         if hidden_score_map is not None:
                    #             value = max(float(score_map[py, px]), float(
                    #                 hidden_score_map[py, px]))
                    #         else:
                    #             value = float(score_map[py, px])
                    #         if value >= min_score:
                    #             score_list.append(np.log10(value))
                    #         else:
                    #             score_list.append(min_logscore)
                    #     else:
                    #         score_list.append(min_logscore)
                    # # 出現した中で最大のものを保存する
                    # score_sum = np.sum(score_list)
                    score = eval_score(
                        point_int, image_width, image_height, score_map, hidden_score_map, min_score)
                    if score > max_score:
                        max_score = score
                        max_score_dat = {
                            "rvec": rvec,
                            "tvec": tvec,
                            "point": point.reshape((-1, 2)),
                        }
                cur_itr_num += 1
                if max_score > cut_score_th:
                    break
            if max_score > cut_score_th:
                break
    max_score_dat["itr_num"] = cur_itr_num
    return max_score, max_score_dat
    """


def quat2rvec(quat):
    """クオータニオンをrotation vectorに変換する"""
    theta = 2 * np.arccos(quat[3])
    vec = np.array(quat[:3])
    vec = vec / np.linalg.norm(vec)
    return theta * vec


def get_Pscore(Pmat):
    diff_trans = np.linalg.norm(Pmat[0:3, 3])
    diff_rot = np.linalg.norm(Rotation.from_matrix(Pmat[0:3, 0:3]).as_rotvec())
    return diff_trans, diff_rot


bbox_lines = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)
def make_result_image(image, target_model, max_score_dat, pred_convex, pred_hidden_convex, pred_convex_lm, cam_mat, dist):
    result = image.copy()
    bounding_point = target_model["bounding_point"]
    bounding_img_point, _ = cv2.projectPoints(
        bounding_point,
        max_score_dat["rvec"],
        max_score_dat["tvec"],
        cam_mat,
        dist,
    )
    bounding_img_point = bounding_img_point.reshape(
        (-1, 2)).astype(np.int64)
    for line in bbox_lines:
        result = cv2.line(
            result,
            tuple(bounding_img_point[line[0]]),
            tuple(bounding_img_point[line[1]]),
            (255, 255, 255),
            2,
        )
    if pred_hidden_convex is not None:
        image_list = [
            image,
            cv2.cvtColor(
                (pred_convex.cpu() * 255).numpy().astype(np.uint8)[0, 0], cv2.COLOR_GRAY2BGR,),
            #  cv2.cvtColor(
            #      (pred_concave.cpu() * 255).numpy().astype(np.uint8)[0, 0], cv2.COLOR_GRAY2BGR,),
            cv2.cvtColor(
                (pred_hidden_convex.cpu() * 255).numpy().astype(np.uint8)[0, 0], cv2.COLOR_GRAY2BGR,),
            cv2.cvtColor(
                (pred_convex_lm.cpu() * 255).numpy().astype(np.uint8)[0, 0], cv2.COLOR_GRAY2BGR),
            result,
        ]
    else:
        image_list = [
            image,
            cv2.cvtColor(
                (pred_convex.cpu() * 255).numpy().astype(np.uint8)[0, 0], cv2.COLOR_GRAY2BGR,),
            #  cv2.cvtColor(
            #      (pred_concave.cpu() * 255).numpy().astype(np.uint8)[0, 0], cv2.COLOR_GRAY2BGR,),
            cv2.cvtColor(
                (pred_convex_lm.cpu() * 255).numpy().astype(np.uint8)[0, 0], cv2.COLOR_GRAY2BGR),
            result,
        ]
    if pred_hidden_convex is not None:
        image_list = [
            image,
            cv2.cvtColor(
                (pred_convex.cpu() * 255).numpy().astype(np.uint8)[0, 0], cv2.COLOR_GRAY2BGR,),
            #  cv2.cvtColor(
            #      (pred_concave.cpu() * 255).numpy().astype(np.uint8)[0, 0], cv2.COLOR_GRAY2BGR,),
            cv2.cvtColor(
                (pred_hidden_convex.cpu() * 255).numpy().astype(np.uint8)[0, 0], cv2.COLOR_GRAY2BGR,),
            cv2.cvtColor(
                (pred_convex_lm.cpu() * 255).numpy().astype(np.uint8)[0, 0], cv2.COLOR_GRAY2BGR),
            result,
        ]
    else:
        image_list = [
            image,
            cv2.cvtColor(
                (pred_convex.cpu() * 255).numpy().astype(np.uint8)[0, 0], cv2.COLOR_GRAY2BGR,),
            #  cv2.cvtColor(
            #      (pred_concave.cpu() * 255).numpy().astype(np.uint8)[0, 0], cv2.COLOR_GRAY2BGR,),
            cv2.cvtColor(
                (pred_convex_lm.cpu() * 255).numpy().astype(np.uint8)[0, 0], cv2.COLOR_GRAY2BGR),
            result,
        ]
    output_image = cv2.hconcat(image_list)
    return output_image

def main():
    from setuptools._distutils.util import strtobool
    parser = argparse.ArgumentParser(
        prog="",  # プログラム名
        usage="",  # プログラムの利用方法
        add_help=True,  # -h/–help オプションの追加
    )

    parser.add_argument("--dnnmodel", type=str, default="./model_010.pth")#学習済みモデル
    parser.add_argument("--input", type=str, default="./test/image_000000.png")#入力画像
    parser.add_argument("--blockmodel", type=str, default="yellow_block")#blockmodelの名前
    parser.add_argument(
        "--annotatefile", type=str, default="./test/annotation.yaml"
    )#バッチ評価モード用のアノテーション YAML
    parser.add_argument("--output", type=str, default="result.png")#出力画像
    parser.add_argument("--outputdir", type=str, default="./output")#出力ディレクトリ
    parser.add_argument("--device", type=str,
                        default="cuda", choices=["cpu", "cuda"])#デバイス
    parser.add_argument('--score_th', type=float,
                        # nargs='*', default=[0.5, 0.3, 0.25, 0.2, 0.15, 0.1])
                        nargs='*', default=[0.4, 0.3, 0.25, 0.2, 0.15, 0.1])#スコアのしきい値
    parser.add_argument('--itr_num', type=int, default=0)#イテレーション数
    parser.add_argument('--issort', type=strtobool, default=1)#ソートするかどうか

    parser.add_argument('--use_p3p', type=strtobool, default=1)#P3Pを使うかどうか
    parser.add_argument('--use_ransac', type=strtobool, default=1)#RANSACを使うかどうか
    parser.add_argument('--cut_score_th', type=float, default=-7)#特徴マップ生成時のスコア閾値リスト
    parser.add_argument('--eval_type', type=str,
                        default='convination', choices=['area_size', 'prob', 'convination'])#評価の種類
    
    
    args = parser.parse_args()

    device = args.device
    issort = args.issort == 1
    use_p3p = args.use_p3p == 1
    use_ransac = args.use_ransac == 1

    #yamlを読み込んでカメラパラメータやブロックごとの#D情報を取得
    with open(args.annotatefile, "r", encoding="utf-8") as f:
        anno_dat = yaml.safe_load(f)


    model = torch.load(args.dnnmodel)
    model.to(device)
    model.eval()

    #各ブロック種別の頂点位置や結合情報から
    #各ブロックをP3P/RANSACに適用しやすい形に変形
    fitting_model = gen_fitting_model(anno_dat["blocks"], issort=issort)
    #print(fitting_model)
    visualize_block_model(fitting_model["yellow_block"], title="Yellow Block")
    #input(単一画像であれば)これ
    #空であればyaml内の全アノテーション複数画像をループするバッチを評価するモード
    if args.input != "":
        print("aaaaaaaaaaaaaaaaaaaa")
        cam_mat = np.array(anno_dat["annotations"]
                           [0]["camera_matrix"]).reshape(3, 3)  # TODO
        dist = np.zeros((5))
        start_time = time.time()
        image = cv2.imread(args.input)
        # (
        #     pred_convex,
        #     pred_concave,
        #     pred_convex_lm,
        #     pred_concave_lm,
        #     convex_pos,
        #     concave_pos,
        # )
        #print(image.shape)
        #image = cv2.resize(image, (512, 512))
        #print(image.shape)

        #ヒートマップ変数候補2
        #これもなんかちがうわ色んな情報が入ってる
        ret_dict = gen_featrue_map(
            image, model, device, score_ths=args.score_th)
        print(gen_featrue_map(
            image, model, device, score_ths=args.score_th))
        print(ret_dict)#何故か空

        pred_convex = ret_dict["pred_convex"]#凸頂点ヒートマップ
        pred_concave = ret_dict["pred_concave"]#凹頂点ヒートマップ
        pred_convex_lm = ret_dict["pred_convex_lm"]#凸頂点ランドマーク回帰マップ
        pred_concave_lm = ret_dict["pred_concave_lm"]#凹頂点ランドマーク回帰マップ
        convex_pos = ret_dict["convex_pos"]#ヒートマップから抽出した凸頂点の2D座標
        concave_pos = ret_dict["concave_pos"]#ヒートマップから抽出した凹頂点の2D座標

        if "pred_hidden_convex" in ret_dict:
            pred_hidden_convex = ret_dict["pred_hidden_convex"]
        else:
            pred_hidden_convex = None
        if "pred_hidden_concave" in ret_dict:
            pred_hidden_concave = ret_dict["pred_hidden_concave"]
        else:
            pred_hidden_concave = None

        if use_p3p:
            max_score, max_score_dat = fit2model_p3p(
                fitting_model[args.blockmodel],
                convex_pos, pred_convex,
                concave_pos,  pred_concave,
                cam_mat, dist, itr_num=args.itr_num, use_ransac=use_ransac, cut_score_th=args.cut_score_th, pred_hidden_convex=pred_hidden_convex,
                eval_type=args.eval_type
            )
        else:
            max_score, max_score_dat = fit2model(
                fitting_model[args.blockmodel], convex_pos, pred_convex, cam_mat, dist, itr_num=args.itr_num, issort=issort, use_ransac=use_ransac
            )
        end_time = time.time()
        print(end_time-start_time)

        result = image.copy()
        bounding_point = fitting_model[args.blockmodel]["bounding_point"]
        bounding_img_point, _ = cv2.projectPoints(
            bounding_point, max_score_dat["rvec"], max_score_dat["tvec"], cam_mat, dist
        )
        bounding_img_point = bounding_img_point.reshape(
            (-1, 2)).astype(np.int64)
        for line in bbox_lines:
            result = cv2.line(
                result,
                tuple(bounding_img_point[line[0]]),
                tuple(bounding_img_point[line[1]]),
                (255, 255, 255),
                2,
            )

        output_image = cv2.hconcat(
            [image,
             cv2.cvtColor(
                 (pred_convex.cpu() * 255).numpy().astype(np.uint8)[0, 0], cv2.COLOR_GRAY2BGR),
             cv2.cvtColor(
                 (pred_concave.cpu() * 255).numpy().astype(np.uint8)[0, 0], cv2.COLOR_GRAY2BGR),
             cv2.cvtColor(
                 (pred_convex_lm.cpu() * 255).numpy().astype(np.uint8)[0, 0], cv2.COLOR_GRAY2BGR),
             result,
             ]
        )
        # 凸頂点を青で描画
        for i, (x, y) in enumerate(convex_pos):
            cv2.circle(img_draw, (int(x), int(y)), radius=5, color=(255, 0, 0), thickness=-1)  # 青 (BGR)

        # 凹頂点を赤で描画
        for i, (x, y) in enumerate(concave_pos):
            cv2.circle(img_draw, (int(x), int(y)), radius=5, color=(0, 0, 255), thickness=-1)  # 赤 (BGR)

        cv2.imwrite("./output/output_with_colored_points.png", img_draw)
        output_image = cv2.hconcat(
            [image,
             cv2.cvtColor(
                 (pred_convex.cpu() * 255).numpy().astype(np.uint8)[0, 0], cv2.COLOR_GRAY2BGR),
             cv2.cvtColor(
                 (pred_concave.cpu() * 255).numpy().astype(np.uint8)[0, 0], cv2.COLOR_GRAY2BGR),
            img_draw,
             result,
             ]
        )
        if args.output != '':
            cv2.imwrite(args.output, output_image)
        print(args.input, max_score)#-4.35はマックススコアだった
        
        #計算に使ったもの
        # 内部パラメータ K の出力
        print("\n=====内部パラメータ K:=====")
        print(cam_mat)

        # ブロックの3Dデータの出力
        print("\n=====ブロックの3Dデータ:=====")
        convex_vertices = fitting_model[args.blockmodel]["convex_vertex_pos"]
        concave_vertices = fitting_model[args.blockmodel]["concave_vertex_pos"]

        print("凸頂点:")
        for i, vertex in enumerate(convex_vertices.reshape(-1, 3)):
            print(f"p{i+1}: {tuple(vertex)}")

        print("\n凹頂点:")
        for i, vertex in enumerate(concave_vertices.reshape(-1, 3)):
            print(f"p{i+1}: {tuple(vertex)}")
    

        # 凸頂点を青で描画
        for i, (x, y) in enumerate(convex_pos):
            print(f"凸 p{i+1}: ({x}, {y})")

        # 凹頂点を赤で描画
        for i, (x, y) in enumerate(concave_pos):
            print(f"凹 p{i+1}: ({x}, {y})")

        # P3P で求めた R と T の出力
        print("\n=====P3P で求めた R:=====")
        print(max_score_dat["rvec"])
        print("\n=====P3P で求めた T:=====")
        print(max_score_dat["tvec"])

        # Rodrigues 変換で求めた R を行列に変換
        R, _ = cv2.Rodrigues(max_score_dat["rvec"])
        print("\n=====Rodrigues 変換後の R:=====")
        print(R)

        # 世界座標 P_w を求める
        P_local = np.array(convex_vertices).reshape(-1, 3)  # モデル座標系の点群
        P_world = (R @ P_local.T).T + max_score_dat["tvec"].T  # ワールド座標に変換

        # 最終的な world 座標の出力
        print("\n=====最終的に求めた world 座標 P_w:=====")
        print("凸頂点:")
        for i, (x, y, z) in enumerate(P_world):
            print(f"p{i+1}: ({x}, {y}, {z})")

        P_local = np.array(concave_vertices).reshape(-1, 3)
        P_world = (R @ P_local.T).T + max_score_dat["tvec"].T

        print("凹頂点:")
        for i, (x, y, z) in enumerate(P_world):
            print(f"p{i+1}: ({x}, {y}, {z})")

    else:
        print("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
        # 評価用コード
        proc_time = []
        for anno_idx, anno in enumerate(anno_dat["annotations"]):
            # if anno_idx < 100:
            #     continue
            # if anno["imagefile"] in ['image_000007.png', 'image_000022.png', 'image_000047.png', 'image_000064.png', 'image_000072.png', 'image_000144.png']:
            #     continue

            dist = np.zeros((5))
            image_file = os.path.join(os.path.dirname(
                args.annotatefile), anno["imagefile"])
            cam_mat = np.array(anno["camera_matrix"]).reshape(3, 3)
            image = cv2.imread(image_file)
            blockmodel = anno["block_name"]

            ret_dict = gen_featrue_map(
                image, model, device, score_ths=args.score_th)
            pred_convex = ret_dict["pred_convex"]
            pred_concave = ret_dict["pred_concave"]
            pred_convex_lm = ret_dict["pred_convex_lm"]
            pred_concave_lm = ret_dict["pred_concave_lm"]
            convex_pos = ret_dict["convex_pos"]
            concave_pos = ret_dict["concave_pos"]
            if "pred_hidden_convex" in ret_dict:
                pred_hidden_convex = ret_dict["pred_hidden_convex"]
            else:
                pred_hidden_convex = None
            if "pred_hidden_concave" in ret_dict:
                pred_hidden_concave = ret_dict["pred_hidden_concave"]
            else:
                pred_hidden_concave = None
            start_time = time.time()
            if use_p3p:
                max_score, max_score_dat = fit2model_p3p(
                    fitting_model[blockmodel],
                    convex_pos, pred_convex,
                    concave_pos,  pred_concave,
                    cam_mat, dist, itr_num=args.itr_num, use_ransac=use_ransac, cut_score_th=args.cut_score_th, pred_hidden_convex=pred_hidden_convex,
                    eval_type=args.eval_type)
            else:
                max_score, max_score_dat = fit2model(
                    fitting_model[blockmodel], convex_pos, pred_convex, cam_mat, dist, itr_num=args.itr_num, issort=issort, use_ransac=use_ransac)
            end_time = time.time()
            proc_time.append(end_time-start_time)

            output_image = make_result_image(image, fitting_model[blockmodel], max_score_dat, pred_convex, pred_hidden_convex, pred_convex_lm, cam_mat, dist)

            if args.output != '':
                cv2.imwrite(args.output, output_image)
            else:
                os.makedirs(args.outputdir, exist_ok=True)
                output_file = os.path.basename(
                    image_file).replace('image', 'result')
                cv2.imwrite(os.path.join(args.outputdir, output_file), output_image)

            camera_rvec = quat2rvec(anno["camera_orientation"])
            R1, _ = cv2.Rodrigues(camera_rvec)
            Proj1 = np.zeros((4, 4))
            Proj1[0:3, 0:3] = R1
            Proj1[0:3, 3] = anno["camera_position"]
            Proj1[3, 3] = 1.0

            estimate_rvec = max_score_dat["rvec"].reshape(-1)
            R2, _ = cv2.Rodrigues(estimate_rvec)
            Proj2 = np.zeros((4, 4))
            Proj2[0:3, 0:3] = R2
            Proj2[0:3, 3] = max_score_dat["tvec"].reshape(-1)
            Proj2[3, 3] = 1.0
            Proj = Proj2 @ Proj1
            diff_trans, diff_rot = get_Pscore(Proj)
            if blockmodel == "cyan_block":
                alt_proj = Proj2 @ np.array([[-1, 0, 0, 0], [0, -1, 0, 0],
                                            [0, 0, 1, 0], [0, 0, 0, 1]]) @ Proj1
                alt_diff_trans, alt_diff_rot = get_Pscore(alt_proj)
                if alt_diff_rot < diff_rot and alt_diff_trans < diff_trans:
                    diff_rot = alt_diff_rot
                    diff_trans = alt_diff_trans
            elif blockmodel == "red_block":
                alt_proj = Proj2 @ np.array([[1, 0, 0, 0], [0, -1, 0, 0],
                                            [0, 0, -1, 0.03], [0, 0, 0, 1]]) @ Proj1
                alt_diff_trans, alt_diff_rot = get_Pscore(alt_proj)
                if alt_diff_rot < diff_rot and alt_diff_trans < diff_trans:
                    diff_rot = alt_diff_rot
                    diff_trans = alt_diff_trans
            elif blockmodel == "green_block":
                alt_proj1 = Proj2 @ np.array([[0, 0, -1, 0.03], [1, 0, 0, 0], [
                                             0, -1, 0, 0.03], [0, 0, 0, 1]]) @ Proj1
                alt_proj2 = Proj2 @ np.array(
                    [[0, 1, 0, 0], [0, 0, -1, 0.03], [-1, 0, 0, 0.03], [0, 0, 0, 1]]) @ Proj1
                for alt_proj in [alt_proj1, alt_proj2]:
                    alt_diff_trans, alt_diff_rot = get_Pscore(alt_proj)
                    if alt_diff_rot < diff_rot and alt_diff_trans < diff_trans:
                        diff_rot = alt_diff_rot
                        diff_trans = alt_diff_trans
            elif blockmodel == "brown_block":
                alt_proj1 = Proj2 @ np.array(
                    [[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0.03], [0, 0, 0, 1]]) @ Proj1
                alt_proj2 = Proj2 @ np.array([[-1, 0, 0, 0.03], [0, 1, 0, 0], [
                                             0, 0, -1, 0.03], [0, 0, 0, 1]]) @ Proj1
                alt_proj3 = Proj2 @ np.array([[0, 0, -1, 0.03],
                                             [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]) @ Proj1
                alt_proj4 = Proj2 @ np.array(
                    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0.03], [0, 0, 0, 1]]) @ Proj1
                alt_proj5 = Proj2 @ np.array(
                    [[0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]) @ Proj1
                alt_proj6 = Proj2 @ np.array([[-1, 0, 0, 0.03], [0, -1, 0, 0], [
                                             0, 0, 1, 0], [0, 0, 0, 1]]) @ Proj1
                alt_proj7 = Proj2 @ np.array(
                    [[0, 0, -1, 0.03], [0, -1, 0, 0], [-1, 0, 0, 0.03], [0, 0, 0, 1]]) @ Proj1
                for alt_proj in [alt_proj1, alt_proj2, alt_proj3, alt_proj4, alt_proj5, alt_proj6, alt_proj7]:
                    alt_diff_trans, alt_diff_rot = get_Pscore(alt_proj)
                    if alt_diff_rot < diff_rot and alt_diff_trans < diff_trans:
                        diff_rot = alt_diff_rot
                        diff_trans = alt_diff_trans
            elif blockmodel == "purple_block":
                alt_proj = Proj2 @ np.array([[-1, 0, 0, 0.03], [0, 0, 1, -0.03], [
                                            0, 1, 0, 0.03], [0, 0, 0, 1]]) @ Proj1
                alt_diff_trans, alt_diff_rot = get_Pscore(alt_proj)
                if alt_diff_rot < diff_rot and alt_diff_trans < diff_trans:
                    diff_rot = alt_diff_rot
                    diff_trans = alt_diff_trans
            elif blockmodel == "lightgreen_block":
                alt_proj = Proj2 @ np.array([[-1, 0, 0, 0], [0, 0, 1, -0.03], [
                                            0, 1, 0, 0.03], [0, 0, 0, 1]]) @ Proj1
                alt_diff_trans, alt_diff_rot = get_Pscore(alt_proj)
                if alt_diff_rot < diff_rot and alt_diff_trans < diff_trans:
                    diff_rot = alt_diff_rot
                    diff_trans = alt_diff_trans

            print("image_file",image_file)
            print("max_score",max_score)
            print("diff_trains",diff_trans)
            print("diff_rot",diff_rot)
            print("proc_time[-1]",proc_time[-1])
            print("max_score_dat[itr_num]",max_score_dat["itr_num"])
                  
            




            # _ = input()
        print(np.mean(proc_time))


if __name__ == "__main__":
    main()
