import os
import glob
import random
import numpy as np
import xml.etree.ElementTree as ET

def parse_xyz(xyz_str):
    """ 'x y z' 文字列を numpy 配列に変換 """
    if not xyz_str:
        return np.array([0.0, 0.0, 0.0])
    return np.array([float(v) for v in xyz_str.split()])

def to_xyz_str(xyz_arr):
    """ numpy 配列を 'x y z' 文字列に変換 """
    return f"{xyz_arr[0]:.6f} {xyz_arr[1]:.6f} {xyz_arr[2]:.6f}"

# rpy_noise_range は微小ノイズ用として残しておきます
def augment_urdf(input_path, output_path, robot_name, scale_geom_range=(0.8, 1.2), scale_mass_range=(0.8, 1.2), rpy_noise_range=0.1):
    """
    URDFを読み込み、パラメータを「行儀よく」ランダムに拡張して保存する
    """
    try:
        ET.register_namespace('xacro', "http://www.ros.org/wiki/xacro")
        tree = ET.parse(input_path)
        root = tree.getroot()
    except Exception as e:
        print(f"[Error] Failed to parse {input_path}: {e}")
        return False

    # ロボット名をファイル名に合わせて変更
    new_robot_name_attr = os.path.splitext(os.path.basename(output_path))[0]
    root.set('name', new_robot_name_attr)

    # 0. メッシュパスの修正
    for mesh in root.findall('.//mesh'):
        filename = mesh.get('filename')
        if filename and filename.startswith('../meshes'):
            new_filename = filename.replace('../meshes', 'meshes')
            mesh.set('filename', new_filename)

    # 1. Jointの拡張 (ここが今回のメイン修正)
    for joint in root.findall('joint'):
        origin = joint.find('origin')
        if origin is not None:
            # === XYZ (位置): 主軸に沿って伸ばす ===
            xyz_str = origin.get('xyz')
            if xyz_str:
                base_xyz = parse_xyz(xyz_str)
            else:
                base_xyz = np.array([0.0, 0.0, 0.0])

            # 確率80%で「行儀の良い（1軸だけ伸びた）」データを作る
            if random.random() < 0.8:
                # 元のデータで最も長い軸を「主軸」とする (全部0ならランダム)
                if np.linalg.norm(base_xyz) > 1e-6:
                    main_axis_idx = np.argmax(np.abs(base_xyz))
                else:
                    main_axis_idx = random.randint(0, 2)
                
                new_xyz = base_xyz.copy()
                
                # 主軸: しっかり伸縮させる (0.5倍 ~ 1.5倍)
                # 元が0だった場合に備えて、最低でも少しは伸ばす処理を入れる
                scale = random.uniform(0.5, 3.0)
                if abs(new_xyz[main_axis_idx]) < 0.01:
                    # 元がほぼ0なら、強制的に 5cm~20cm くらい伸ばしてみる（符号はランダム）
                    new_xyz[main_axis_idx] = random.uniform(0.05, 0.2) * random.choice([1, -1])
                else:
                    new_xyz[main_axis_idx] *= scale
                
                # 副軸: ほぼゼロにする（微小ノイズのみ）
                # これにより「ジグザグ」を抑制し、「まっすぐ」にする
                for i in range(3):
                    if i != main_axis_idx:
                        new_xyz[i] = random.uniform(-0.005, 0.005) # ±5mm以内の誤差
            else:
                # 残り20%は従来の全体スケーリング（L字パーツなどの多様性維持）
                scale = random.uniform(*scale_geom_range)
                new_xyz = base_xyz * scale

            origin.set('xyz', to_xyz_str(new_xyz))
            
            # === RPY (回転): 90度刻みで直交性を維持 ===
            rpy_str = origin.get('rpy')
            if rpy_str:
                rpy = parse_xyz(rpy_str)
            else:
                rpy = np.array([0.0, 0.0, 0.0])
            
            # 確率50%で「構造的な回転（90度単位）」を変更
            if random.random() < 0.5:
                # X, Y, Z のいずれかの軸周りに 0, 90, -90, 180 度回転
                axis_idx = random.randint(0, 2)
                angle = random.choice([0, 1.5708, -1.5708, 3.1416]) # 90度刻み
                
                base_rotation = np.array([0.0, 0.0, 0.0])
                base_rotation[axis_idx] = angle
                new_rpy_base = base_rotation
            else:
                # 回転を変えない
                new_rpy_base = rpy

            # 最後に微小ノイズ (±5度程度) を乗せる
            noise = np.random.uniform(-0.08, 0.08, 3)
            final_rpy = new_rpy_base + noise
            
            origin.set('rpy', to_xyz_str(final_rpy))

    # 2. Linkの拡張 (Mass等は変更なし)
    for link in root.findall('link'):
        inertial = link.find('inertial')
        if inertial is not None:
            mass_elem = inertial.find('mass')
            if mass_elem is not None:
                try:
                    original_mass = float(mass_elem.get('value'))
                    mass_scale = random.uniform(*scale_mass_range)
                    new_mass = original_mass * mass_scale
                    mass_elem.set('value', str(new_mass))
                except: pass

                inertia_elem = inertial.find('inertia')
                if inertia_elem is not None:
                    for attr in ['ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz']:
                        val = inertia_elem.get(attr)
                        if val:
                            try:
                                new_val = float(val) * mass_scale
                                inertia_elem.set(attr, str(new_val))
                            except: pass

    try:
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        return True
    except Exception as e:
        print(f"[Error] Failed to write {output_path}: {e}")
        return False
def main():
    # --- 設定 ---
    input_dir = "./merge_joint_robots"      
    output_root_dir = "./augmented_dataset" 
    num_augmentations = 200                
    
    geom_range = (0.8, 1.2) 
    mass_range = (0.5, 1.5)
    
    rpy_noise_rad = 0.1 

    exclude_keywords = ['merge_kinova', 'merge_a1','merge_nao']
    
    # --- 実行 ---
    input_files = glob.glob(os.path.join(input_dir, "*.urdf"))
    print(f"Found {len(input_files)} original URDF files.")

    total_generated = 0
    
    for f in input_files:
        filename = os.path.basename(f)
        robot_name, ext = os.path.splitext(filename)
        
        if any(keyword in robot_name for keyword in exclude_keywords):
            print(f"[Skip] Skipping excluded robot: {robot_name}")
            continue

        print(f"Processing: {robot_name} ...")
        
        save_dir = os.path.join(output_root_dir, robot_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for i in range(num_augmentations):
            new_filename = f"{robot_name}_{i:03d}.urdf"
            output_path = os.path.join(save_dir, new_filename)
            
            if augment_urdf(f, output_path, robot_name, geom_range, mass_range, rpy_noise_range=rpy_noise_rad):
                total_generated += 1
            
    print(f"\n[Done] Generated {total_generated} augmented robots in '{output_root_dir}'.")

if __name__ == '__main__':
    main()