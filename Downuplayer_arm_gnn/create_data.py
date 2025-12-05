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

def augment_urdf(input_path, output_path, robot_name, scale_geom_range=(0.8, 1.2), scale_mass_range=(0.8, 1.2)):
    """
    URDFを読み込み、パラメータをランダムに拡張して保存する
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

    # 1. Jointの拡張
    for joint in root.findall('joint'):
        origin = joint.find('origin')
        if origin is not None:
            xyz_str = origin.get('xyz')
            if xyz_str:
                xyz = parse_xyz(xyz_str)
                scale = random.uniform(*scale_geom_range)
                new_xyz = xyz * scale
                origin.set('xyz', to_xyz_str(new_xyz))

    # 2. Linkの拡張
    for link in root.findall('link'):
        inertial = link.find('inertial')
        if inertial is not None:
            mass_elem = inertial.find('mass')
            if mass_elem is not None:
                original_mass = float(mass_elem.get('value'))
                mass_scale = random.uniform(*scale_mass_range)
                new_mass = original_mass * mass_scale
                mass_elem.set('value', str(new_mass))

                inertia_elem = inertial.find('inertia')
                if inertia_elem is not None:
                    for attr in ['ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz']:
                        val = inertia_elem.get(attr)
                        if val:
                            new_val = float(val) * mass_scale
                            inertia_elem.set(attr, str(new_val))

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
    num_augmentations = 100                
    
    geom_range = (0.8, 1.2) 
    mass_range = (0.5, 1.5) 

    # ★ 除外するロボット名のリスト（部分一致で判定します）
    exclude_keywords = ['merge_kinova', 'merge_a1','merge_nao']
    
    # --- 実行 ---
    input_files = glob.glob(os.path.join(input_dir, "*.urdf"))
    print(f"Found {len(input_files)} original URDF files.")

    total_generated = 0
    
    for f in input_files:
        filename = os.path.basename(f)
        robot_name, ext = os.path.splitext(filename)
        
        # ★★★ 例外処理：指定したキーワードが含まれていたらスキップ ★★★
        # any() を使ってリスト内のどれか一つでも含まれていればTrueになります
        if any(keyword in robot_name for keyword in exclude_keywords):
            print(f"[Skip] Skipping excluded robot: {robot_name}")
            continue

        print(f"Processing: {robot_name} ...")
        
        # ロボットごとの保存フォルダを作成
        save_dir = os.path.join(output_root_dir, robot_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 指定数だけ生成
        for i in range(num_augmentations):
            new_filename = f"{robot_name}_{i:03d}.urdf"
            output_path = os.path.join(save_dir, new_filename)
            
            if augment_urdf(f, output_path, robot_name, geom_range, mass_range):
                total_generated += 1
            
    print(f"\n[Done] Generated {total_generated} augmented robots in '{output_root_dir}'.")

if __name__ == '__main__':
    main()