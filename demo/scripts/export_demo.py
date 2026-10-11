"""Export a curated gallery from public recordings without changing research data."""
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'demo/public'
STAGING = ROOT / 'demo/.export'
TOOLS = {
    'claw_hammer': ('Claw hammer', 'hammer/claw_hammer/claw_hammer.urdf'),
    'mallet_hammer': ('Mallet', 'hammer/mallet_hammer/mallet_hammer.urdf'),
    'cuboid_hammer_v014': ('Cuboid hammer', 'training/hammer/cuboid_hammer_v014/cuboid_hammer_v014.urdf'),
    'long_screwdriver': ('Long screwdriver', 'screwdriver/long_screwdriver/long_screwdriver.urdf'),
    'short_screwdriver': ('Short screwdriver', 'screwdriver/short_screwdriver/short_screwdriver.urdf'),
    'cylinder_screwdriver_v3009': ('Cylinder screwdriver', 'training/screwdriver/cylinder_screwdriver_v3009/cylinder_screwdriver_v3009.urdf'),
}

# Read a saved artifact with no imports from the research runtime.
def read(path):
    return json.loads(path.read_text())

# Write deterministic compact JSON into the demo directory only.
def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(',', ':'), allow_nan=False) + '\n')

# Find strong successful samples, using tracking error after goal completion.
def ranked(run, tool, mode=None):
    root = ROOT / 'experiments/results' / run
    entries = read(root / 'execution/replay/manifest.json')['entries']
    candidates = []
    for entry in entries:
        if entry['object_name'] != tool or (mode and entry['mode'] != mode):
            continue
        raw = read(root / 'execution/raw' / (entry['trial_id'] + '.json'))
        score = (not entry['execution_success'], -entry['goal_completion_pct'],
                 raw.get('translation_rmse_m', math.inf), raw.get('rotation_rmse_deg', math.inf), entry['trial_id'])
        candidates.append((score, root, entry, raw))
    return sorted(candidates, key=lambda row: row[0])

# Parse URDF vectors with the standard origin defaults.
def vector(element, attr, default):
    return [float(x) for x in element.get(attr, default).split()] if element is not None else [float(x) for x in default.split()]

# Build a rendering-only URDF graph with source mesh hashes and named joints.
def urdf_graph(path, meshes):
    robot = ET.parse(path).getroot()
    materials = {m.get('name'): vector(m.find('color'), 'rgba', '.55 .58 .6 1') for m in robot.findall('material')}
    links = []
    for link in robot.findall('link'):
        visuals = []
        for visual in link.findall('visual'):
            origin = visual.find('origin'); material = visual.find('material'); geom = visual.find('geometry')
            color = materials.get(material.get('name'), [.65, .68, .7, 1]) if material is not None else [.65, .68, .7, 1]
            if material is not None and material.find('color') is not None:
                color = vector(material.find('color'), 'rgba', '.65 .68 .7 1')
            item = {'xyz': vector(origin, 'xyz', '0 0 0'), 'rpy': vector(origin, 'rpy', '0 0 0'), 'color': color}
            mesh = geom.find('mesh')
            if mesh is not None:
                source = (path.parent / mesh.get('filename')).resolve()
                if not source.is_file():
                    raise ValueError(f'Missing mesh: {source.relative_to(ROOT)}')
                key = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
                meshes[key] = {'source': str(source.relative_to(ROOT)), 'format': source.suffix.lower(), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
                item.update(mesh=key, scale=vector(mesh, 'scale', '1 1 1'))
            elif geom.find('box') is not None:
                item['box'] = vector(geom.find('box'), 'size', '1 1 1')
            elif geom.find('cylinder') is not None:
                c = geom.find('cylinder');item['cylinder'] = [float(c.get('radius')), float(c.get('length'))]
            elif geom.find('sphere') is not None:
                item['sphere'] = float(geom.find('sphere').get('radius'))
            else:
                raise ValueError('Unsupported visual geometry')
            visuals.append(item)
        links.append({'name': link.get('name'), 'visuals': visuals})
    joints = []
    for j in robot.findall('joint'):
        o = j.find('origin');m = j.find('mimic')
        joints.append({'name': j.get('name'), 'type': j.get('type'), 'parent': j.find('parent').get('link'), 'child': j.find('child').get('link'), 'xyz': vector(o, 'xyz', '0 0 0'), 'rpy': vector(o, 'rpy', '0 0 0'), 'axis': vector(j.find('axis'), 'xyz', '1 0 0'), 'mimic': ({'joint': m.get('joint'), 'multiplier': float(m.get('multiplier', '1')), 'offset': float(m.get('offset', '0'))} if m is not None else None)})
    return {'links': links, 'joints': joints, 'source': str(path.relative_to(ROOT))}

# Retain complete recorded sampling with six decimal places (sub-micrometre positions).
def rounded(values):
    result = [round(float(v), 6) for v in values]
    if not all(math.isfinite(v) for v in result):
        raise ValueError('Non-finite recording')
    return result

# Export selected recordings and static scene metadata.
def main():
    meshes = {}; examples = []; tool_manifest = []
    for tool, (name, rel) in TOOLS.items():
        if rel.startswith('training/'):
            path = ROOT / 'assets/urdf/dextoolbench_training' / rel.removeprefix('training/')
        else:
            path = ROOT / 'assets/urdf/dextoolbench' / rel
        graph = urdf_graph(path, meshes)
        write(OUT / 'assets' / (tool + '.json'), graph)
        tool_manifest.append({'id': tool, 'name': name, 'family': 'hammer' if 'hammer' in tool else 'screwdriver', 'asset': f'assets/{tool}.json'})
        run = 'old/thesis_main_v3' if tool == 'cylinder_screwdriver_v3009' else 'thesis_main_v3_v4'
        selections = [ranked(run, tool)[0]]
        if tool == 'claw_hammer':
            selections += [ranked(run, tool, mode)[0] for mode in ('llm_lie', 'predefined')]
        for _, root, entry, raw in selections:
            path = root / entry['replay_artifact_path'];payload = read(path);source = next(iter(payload['sources'].values()))
            frames = source['policy_track']['frames'];names = source['metadata'].get('robot_joint_names') or payload['runtime_snapshot']['policy_interface']['policy_joint_names']
            if len(names) != 29 or not all(len(f.get('robot_joint_positions') or []) == 29 for f in frames):
                raise ValueError('Invalid joint mapping')
            geo_id = entry['geometry_trial_id']; geo = read(root / 'geometry/raw' / (geo_id + '.json'))
            for f in frames:
                for key in ('tool_pose', 'goal_pose'):
                    pose = f[key];norm = sum(v*v for v in pose[3:]) ** .5
                    if len(pose) != 7 or not .99 < norm < 1.01:
                        raise ValueError('Invalid pose quaternion')
            # Exclude a trailing post-completion reset, retaining all motion through success.
            trimmed = 0
            while entry['execution_success'] and len(frames) > 1 and frames[-1]['goal_index'] == 0 and frames[-2]['goal_index'] > 0:
                frames = frames[:-1];trimmed += 1
            start = frames[0]['sim_time_sec'];times = [round(f['sim_time_sec'] - start, 6) for f in frames]
            if any(b <= a for a, b in zip(times, times[1:])):
                raise ValueError('Nonmonotonic timestamps')
            record = {'schema': 1, 'jointNames': names, 'times': times, 'tool': [rounded(f['tool_pose']) for f in frames], 'goal': [rounded(f['goal_pose']) for f in frames], 'joints': [rounded(f['robot_joint_positions']) for f in frames], 'goalIndices': [f['goal_index'] for f in frames], 'events': [f.get('event') for f in frames], 'reference': [rounded(p) for p in source['reference_track']['tool_poses']], 'referenceInterval': source['reference_track']['sample_interval_sec'], 'referenceDuration': source['reference_track']['duration_sec']}
            ident = tool + '-' + entry['mode'];write(OUT / 'data' / (ident + '.json'), record)
            examples.append({'id': ident, 'tool': tool, 'method': entry['mode'], 'task': entry['task_name'], 'target': entry.get('target_xy'), 'completion': entry['goal_completion_pct'], 'success': entry['execution_success'], 'translationRmse': raw.get('translation_rmse_m'), 'rotationRmse': raw.get('rotation_rmse_deg'), 'duration': times[-1], 'frames': len(frames), 'data': f'data/{ident}.json', 'provenance': {'run': str(root.relative_to(ROOT / 'experiments/results')), 'geometryTrial': geo_id, 'executionTrial': entry['trial_id'], 'source': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'generationContext': geo.get('generation_context', {}), 'goalTransform': source['metadata'].get('execution_goal_transform', 'identity'), 'trimmedTerminalResetFrames': trimmed, 'descriptionIsAuthored': True}})
    robot = urdf_graph(ROOT / 'assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf', meshes)
    write(OUT / 'assets/robot.json', robot)
    write(STAGING / 'meshes.json', meshes)
    write(OUT / 'data/manifest.json', {'schema': 1, 'units': {'position': 'metres', 'joints': 'radians', 'time': 'seconds'}, 'quaternion': 'xyzw', 'up': 'z', 'robotPosition': [0, .8, 0], 'tableTop': .53, 'tools': tool_manifest, 'examples': examples})
    print(f'Exported {len(examples)} examples, {len(meshes)} unique source meshes')
    for e in examples:
        print(e['id'], e['provenance']['run'], f"{e['duration']:.1f}s", f"{e['completion']:.0f}% goals", e['frames'], 'frames')

if __name__ == '__main__':
    main()
