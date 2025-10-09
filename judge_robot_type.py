# Update the script to accept a regex pattern to select target URDF files.
# - Adds `--pattern` (regex, default: r'.*\.urdf$') and `--no-recursive` flags.
# - If a file is passed, it's included regardless; directories are scanned and filtered by the regex.
from pathlib import Path
from textwrap import dedent


#!/usr/bin/env python3

import sys
import re
import csv
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Set

def _try_import_urdfpy():
    try:
        import urdfpy  # type: ignore
        return urdfpy
    except Exception:
        return None

URDFPY = _try_import_urdfpy()

@dataclass
class URDFModel:
    links: List[str]
    joints: List[Tuple[str, str, str]]  # (name, parent_link, child_link)
    joint_types: Dict[str, str]         # joint_name -> type
    base_link: Optional[str]

def load_urdf(path: Path) -> URDFModel:
    if URDFPY is not None:
        from urdfpy import URDF  # type: ignore
        m = URDF.load(str(path))
        links = [ln.name for ln in m.links]
        joints = [(jn.name, jn.parent, jn.child) for jn in m.joints]
        joint_types = {jn.name: jn.joint_type for jn in m.joints}
        child_links = {jn.child for jn in m.joints}
        base_candidates = [ln for ln in links if ln not in child_links]
        base = base_candidates[0] if base_candidates else None
        return URDFModel(links, joints, joint_types, base)
    else:
        import xml.etree.ElementTree as ET
        tree = ET.parse(str(path))
        root = tree.getroot()
        ns = ''
        if root.tag.startswith('{'):
            ns = root.tag.split('}')[0] + '}'
        links, joints, joint_types = [], [], {}
        for ln in root.findall(f'{ns}link'):
            name = ln.attrib.get('name', '')
            if name:
                links.append(name)
        for j in root.findall(f'{ns}joint'):
            jname = j.attrib.get('name', '')
            jtype = j.attrib.get('type', '')
            parent = j.find(f'{ns}parent')
            child = j.find(f'{ns}child')
            p = parent.attrib.get('link','') if parent is not None else ''
            c = child.attrib.get('link','') if child is not None else ''
            if jname and p and c:
                joints.append((jname, p, c))
            if jname:
                joint_types[jname] = jtype
        child_links = {c for _,_,c in joints}
        base_candidates = [ln for ln in links if ln not in child_links]
        base = base_candidates[0] if base_candidates else None
        return URDFModel(links, joints, joint_types, base)

def build_graph(joints: List[Tuple[str,str,str]]) -> Dict[str, Set[str]]:
    g: Dict[str, Set[str]] = {}
    for _, p, c in joints:
        g.setdefault(p, set()).add(c)
        g.setdefault(c, set())
    return g

def get_end_links(graph: Dict[str, Set[str]]) -> List[str]:
    return [n for n, outs in graph.items() if len(outs) == 0]

def find_chains(graph: Dict[str, Set[str]], base: Optional[str]) -> List[List[str]]:
    if base is None or base not in graph:
        roots = [n for n, outs in graph.items() if len(outs) > 0]
        if not roots:
            return []
        base = roots[0]
    chains: List[List[str]] = []
    stack: List[Tuple[str, List[str]]] = [(base, [base])]
    while stack:
        node, path = stack.pop()
        outs = list(graph.get(node, []))
        if not outs:
            chains.append(path)
        else:
            for nxt in outs:
                stack.append((nxt, path+[nxt]))
    return chains

TOKENS_LEG = r'(leg|foot|paw|toe|ankle|hip|thigh|calf|knee)'
TOKENS_ARM = r'(arm|gripper|hand|wrist|elbow|shoulder|tool)'
TOKENS_TORSO = r'(torso|chest|trunk|pelvis|waist|base|body)'
TOKENS_HEAD = r'(head|neck)'
RE_LEG = re.compile(TOKENS_LEG, re.I)
RE_ARM = re.compile(TOKENS_ARM, re.I)
RE_TORSO = re.compile(TOKENS_TORSO, re.I)
RE_HEAD = re.compile(TOKENS_HEAD, re.I)

def limb_hints(name: str):
    s = name.lower()
    return bool(RE_LEG.search(s)), bool(RE_ARM.search(s)), bool(RE_TORSO.search(s)), bool(RE_HEAD.search(s))

def classify_model(m: URDFModel) -> Dict[str, object]:
    graph = build_graph(m.joints)
    ends = get_end_links(graph)
    chains = find_chains(graph, m.base_link)

    leg_ends = []
    arm_ends = []
    other_ends = []
    for e in ends:
        is_leg, is_arm, _, _ = limb_hints(e)
        chain_len = next((len(p) for p in chains if p and p[-1] == e), 1)
        if chain_len < 3:
            other_ends.append(e)
            continue
        if is_leg:
            leg_ends.append(e)
        elif is_arm:
            arm_ends.append(e)
        else:
            other_ends.append(e)

    legs_hint = sum(limb_hints(x)[0] for x in m.links + [j[0] for j in m.joints])
    arms_hint = sum(limb_hints(x)[1] for x in m.links + [j[0] for j in m.joints])
    head_hint = any(limb_hints(x)[3] for x in m.links)

    leg_count = max(len(leg_ends), min(legs_hint // 3, 6))
    arm_count = max(len(arm_ends), min(arms_hint // 3, 6))

    label = 'その他'
    if leg_count >= 2 and arm_count >= 2 and (any(RE_TORSO.search(x.lower()) for x in m.links) or head_hint):
        label = 'ヒューマノイド'
    elif leg_count >= 4 and arm_count >= 1:
        label = '四脚*アーム'
    elif leg_count >= 4 and arm_count == 0:
        label = '四脚'
    elif leg_count == 0 and arm_count >= 1:
        label = 'アームロボット'

    n_links = len(m.links)
    n_joints = len(m.joints)
    dof_guess = sum(1 for t in m.joint_types.values() if t in ('revolute','prismatic','continuous','planar'))
    return {
        'label': label,
        'links': n_links,
        'joints': n_joints,
        'dof_guess': dof_guess,
        'leg_end_count': len(leg_ends),
        'arm_end_count': len(arm_ends),
        'other_end_count': len(other_ends),
        'base_link': m.base_link or '',
        'ends_preview': ';'.join(ends[:10])
    }

def collect_urdf_files(paths: List[Path], regex: re.Pattern, recursive: bool=True) -> List[Path]:
    files: List[Path] = []
    for p in paths:
        if p.is_file():
            if regex.search(p.as_posix()):
                files.append(p)
        elif p.is_dir():
            it = p.rglob('*') if recursive else p.glob('*')
            for x in it:
                if x.is_file() and regex.search(x.as_posix()):
                    files.append(x)
    return sorted(set(files))

def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+', help='URDFファイル or ディレクトリ（複数可）')
    ap.add_argument('--pattern', default=r'.*\.urdf$', help='対象ファイルの正規表現（デフォルト: .*\\.urdf$）')
    ap.add_argument('--no-recursive', action='store_true', help='ディレクトリ直下のみを探索（デフォルトは再帰）')
    args = ap.parse_args(argv[1:])

    try:
        rx = re.compile(args.pattern, re.I)
    except re.error as e:
        print(f'正規表現エラー: {e}')
        return 3

    inputs = [Path(a).expanduser().resolve() for a in args.paths]
    urdfs = collect_urdf_files(inputs, rx, recursive=(not args.no_recursive))
    if not urdfs:
        print("一致するファイルが見つかりませんでした。--pattern を見直してください。")
        return 2

    rows = []
    for up in urdfs:
        try:
            model = load_urdf(up)
            info = classify_model(model)
            row = {'urdf_path': str(up), **info}
        except Exception as e:
            row = {'urdf_path': str(up), 'label': f'解析失敗: {e}', 'links': '', 'joints': '', 'dof_guess':'',
                   'leg_end_count':'', 'arm_end_count':'', 'other_end_count':'', 'base_link':'', 'ends_preview':''}
        rows.append(row)

    out_csv = Path('urdf_robot_classification.csv')
    with out_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== 分類結果 ===")
    for r in rows:
        print(f"[{r['label']}] {r['urdf_path']}  (links={r['links']}, joints={r['joints']}, legs={r.get('leg_end_count')}, arms={r.get('arm_end_count')})")
    print(f"\nCSV 出力: {out_csv.resolve()}")
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))

out = Path('/mnt/data/urdf_robot_classifier.py')
out.write_text(script, encoding='utf-8')
print("Updated script saved to:", out)
print("Run examples:")
print("  python urdf_robot_classifier.py ./robots")
print("  python urdf_robot_classifier.py --pattern '.*a1.*\\.urdf$' ./robots")
print("  python urdf_robot_classifier.py --pattern '.*\\.(urdf|xml)$' ./robots --no-recursive")
