"""Rich backgrounds for the landscape clips, built with the realistic held-out scene kit (lib.py / env.py).

    python design/website_scenes.py --tasks <tasks.jsonl with spec/render_spec> --kit <scenekit dir> --out <decor dir>

Each task gets one of five settings chosen from its skeleton's environment atom (street, park, winter lot,
parking lot, indoor hall), varied by a per-task seed. Props are static, non-colliding geoms placed outside a
clear corridor around the robot's workspace (|y| < 4 m along the +x travel direction), mostly on the +y / -x
side the default camera (azimuth 135 deg) looks towards. The render copy of each spec gets
`render_scene.website_decor` (scene XML + ground material) for the patched renderer to attach, plus a matching
sky and fog. Training specs are never modified.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

SETTING = {  # environment atom (substring) -> setting
    'friction': 'winter', 'unstable_surface': 'park', 'rough_terrain': 'park', 'dynamic_stability': 'park',
    'slalom': 'parking', 'obstacle': 'parking', 'velocity_tracking': 'street', 'bipedal_velocity': 'street',
    'agile': 'street', 'coordinated': 'street', 'precision_docking': 'hall', 'actuator': 'hall',
    'object_pushing': 'hall', 'arm_reach': 'hall', 'dynamic_balance': 'hall', 'velocity': 'street',
}
SKY = {
    'outdoor': {'sky_zenith_rgb': [0.42, 0.62, 0.86], 'sky_horizon_rgb': [0.90, 0.93, 0.96], 'fog_rgba': [0.86, 0.90, 0.95, 0.12]},
    'indoor': {'sky_zenith_rgb': [0.92, 0.92, 0.91], 'sky_horizon_rgb': [0.92, 0.92, 0.91], 'fog_rgba': [0.92, 0.92, 0.91, 0.08]},
}


def setting_for(bindings):
    for role in ('environment', 'goal', 'constraint'):
        for binding in bindings:
            if binding['role'] == role:
                for key, setting in SETTING.items():
                    if key in binding['atom']:
                        return setting
    return 'street'


def skeletons(histories, robot):
    """FOSA's distinct programs in history order, keyed like design/landscape_tasks.jsonl."""
    programs = {}
    for line in (histories / robot / 'fosa_learn_history.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row.get('status') == 'trained':
            bindings = row['atom_program']['bindings']
            programs.setdefault(json.dumps(bindings, sort_keys=True), bindings)
    return {f'{robot}-{i:03d}': b for i, b in enumerate(programs.values())}


def street(env, dec, rng):
    env.road(dec, (-30, 7.2), (45, 7.2), width=3.4, curbs=True)
    env.patch(dec, (7.5, 10.6), (37.5, 1.5), 'm_concrete_light', z=0.004)
    x = -26.0
    while x < 44:
        w = rng.uniform(9, 14)
        env.building(dec, (x + w / 2, 16.5), (w - 0.4, 5.0, rng.uniform(5.5, 9.0)),
                     mat=rng.choice(['m_brick', 'm_brick', 'm_wall_paint']), win_rows=2, win_cols=max(2, int(w / 2.5)))
        x += w
    for x in np.arange(-24, 44, 7.0):
        env.tree(dec, (x + rng.uniform(-1, 1), 12.2), h=rng.uniform(4.2, 5.4), r=rng.uniform(1.1, 1.5))
    for x in np.arange(-20, 44, 9.0):
        env.lamp_post(dec, (x, 5.3), h=3.6)
        env.bench(dec, (x + 3.0, 5.0), yaw=np.pi)
    for x in np.arange(-18, 44, 11.0):
        env.bush(dec, (x + rng.uniform(-1, 1), 4.6), r=rng.uniform(0.4, 0.6))
    env.building(dec, (-17.0, 2.0), (10.0, 6.0, 7.0), yaw=np.pi / 2, mat='m_brick', win_rows=2, win_cols=3)
    return 'm_concrete', 'outdoor'


def park(env, dec, rng):
    env.patch(dec, (7.5, 0.0), (40.0, 1.3), 'm_dirt', z=0.003)
    for _ in range(26):
        x, y = rng.uniform(-24, 44), rng.uniform(6.5, 22)
        env.tree(dec, (x, y), h=rng.uniform(3.8, 6.0), r=rng.uniform(1.0, 1.7))
    for _ in range(14):
        env.bush(dec, (rng.uniform(-20, 44), rng.uniform(4.4, 6.5)), r=rng.uniform(0.35, 0.65))
    for x in np.arange(-16, 44, 12.0):
        env.bench(dec, (x, 4.6), yaw=np.pi)
        env.lamp_post(dec, (x + 4, 4.8), h=3.4)
    env.fence(dec, (-24, 24), (46, 24), h=1.1)
    for _ in range(5):
        env.rocks(dec, (rng.uniform(-14, 40), rng.uniform(4.5, 8)), radius=0.7, n=8, seed=int(rng.integers(1e6)))
    for y in (-6.0, 0.0, 6.0):
        env.tree(dec, (-12.0 + rng.uniform(-1, 1), y + rng.uniform(-1, 1)), h=5.0, r=1.4)
    return 'm_grass', 'outdoor'


def winter(env, dec, rng):
    env.patch(dec, (8.0, 0.0), (16.0, 3.0), 'm_ice', z=0.003)
    env.fence(dec, (-24, 8.5), (46, 8.5), h=1.0, mat='m_wood_dark')
    x = -26.0
    while x < 44:
        w = rng.uniform(10, 15)
        env.building(dec, (x + w / 2, 15.5), (w - 0.4, 5.0, rng.uniform(6, 9)), mat='m_brick', win_rows=2,
                     win_cols=max(2, int(w / 2.5)))
        x += w
    for x in np.arange(-20, 44, 8.0):
        env.lamp_post(dec, (x, 6.4), h=3.6)
    for x in np.arange(-18, 44, 6.0):
        env.tree(dec, (x + rng.uniform(-1, 1), 11.5), h=rng.uniform(4.0, 5.2), r=rng.uniform(0.9, 1.3))
    for x in np.arange(-6, 30, 6.0):
        env.traffic_cone(dec, (x, 4.3))
    env.bench(dec, (2.0, 6.0), yaw=np.pi)
    return 'm_snow', 'outdoor'


def parking(env, dec, rng):
    for x in np.arange(-22, 44, 2.7):  # stall lines on the backdrop side
        env.patch(dec, (x, 7.0), (0.05, 2.4), 'm_white_paint', z=0.004)
    env.patch(dec, (10, 4.55), (34, 0.06), 'm_yellow_paint', z=0.004)
    env.patch(dec, (10, 11.6), (36, 1.2), 'm_concrete_light', z=0.004)
    env.building(dec, (8.0, 17.5), (52.0, 6.0, 7.5), mat=rng.choice(['m_brick', 'm_wall_paint_gray']), win_rows=2, win_cols=16)
    for x in np.arange(-20, 44, 10.0):
        env.lamp_post(dec, (x, 11.0), h=4.0)
        env.tree(dec, (x + 5, 12.4), h=rng.uniform(4.0, 5.0), r=1.2)
    for x in np.arange(-6, 32, 3.0):
        env.traffic_cone(dec, (x + rng.uniform(-0.3, 0.3), 4.2))
    env.building(dec, (-17.0, 2.0), (10.0, 6.0, 6.5), yaw=np.pi / 2, mat='m_brick', win_rows=2, win_cols=3)
    return 'm_asphalt', 'outdoor'


def hall(env, dec, rng):
    env.room(dec, -14, 46, -14, 12, h=6.0, walls=('ymax', 'xmin'), wall_mat='m_wall_paint_gray', floor_box=False)
    for x in np.arange(-10, 44, 6.5):
        env.window(dec, (x, 11.95), (0, -1), width=2.4, height=1.8, sill_z=2.4, curtains=False)
    for x in np.arange(-11, 44, 4.5):
        env.shelf(dec, (x, 11.6), yaw=np.pi, size=(1.8, 0.5, 2.2), seed=int(rng.integers(1e6)))
    for x in np.arange(-8, 44, 9.0):
        env.table(dec, (x, 8.2), size=(1.6, 0.8, 0.76))
        env.chair(dec, (x - 0.5, 7.4), yaw=np.pi / 2)
        env.cabinet(dec, (x + 4.0, 10.9), yaw=np.pi)
        env.plant(dec, (x + 2.5, 9.5), h=1.1)
    for _ in range(10):
        c = (rng.uniform(-10, 42), rng.uniform(5, 9))
        env.box_package(dec, (c[0], c[1], 0.25), (0.3, 0.3, 0.25), yaw=rng.uniform(0, np.pi))
    return 'm_wood_floor', 'indoor'


LIGHTS = {  # softer than the studio house style, so textured grounds do not blow out
    'outdoor': {'ambient_rgb': [0.17, 0.17, 0.18], 'headlight_diffuse_rgb': [0.18, 0.18, 0.18], 'lights': [
        {'name': 'key', 'position_m': [4.0, -4.0, 9.0], 'direction': [-0.35, 0.45, -1.0], 'color': [1.0, 0.95, 0.86, 1.0],
         'intensity': 0.72, 'castshadow': True},
        {'name': 'fill', 'position_m': [-4.0, 4.0, 6.0], 'direction': [0.4, -0.5, -1.0], 'color': [0.75, 0.82, 1.0, 1.0],
         'intensity': 0.22, 'castshadow': False}]},
    'indoor': {'ambient_rgb': [0.18, 0.18, 0.18], 'headlight_diffuse_rgb': [0.20, 0.20, 0.20], 'lights': [
        {'name': 'key', 'position_m': [1.5, -2.0, 5.5], 'direction': [-0.2, 0.3, -1.0], 'color': [1.0, 0.94, 0.84, 1.0],
         'intensity': 0.68, 'castshadow': True},
        {'name': 'fill', 'position_m': [-3.0, 3.0, 5.0], 'direction': [0.3, -0.4, -1.0], 'color': [0.85, 0.88, 0.95, 1.0],
         'intensity': 0.25, 'castshadow': False}]},
}

BUILDERS = {'street': street, 'park': park, 'winter': winter, 'parking': parking, 'hall': hall}


def scene_xml(lib, env, setting, seed):
    dec = lib.Decor()
    ground, sky = BUILDERS[setting](env, dec, np.random.default_rng(seed))
    world = '\n'.join(dec.world)
    used = set(re.findall(r'material="([^"]+)"', world)) | {ground}
    keep = []
    lines = [line.strip() for line in lib.real_assets().splitlines() if line.strip()]
    for line in lines:
        name = re.search(r'name="([^"]+)"', line)
        if line.startswith('<material') and name and name.group(1) in used:
            keep.append(line)
            tex = re.search(r'texture="([^"]+)"', line)
            if tex:
                keep.append(next(l for l in lines if l.startswith('<texture') and f'name="{tex.group(1)}"' in l))
    assets = '\n'.join(dict.fromkeys(keep + dec.assets))
    return f'<mujoco>\n<asset>\n{assets}\n</asset>\n<worldbody>\n{world}\n</worldbody>\n</mujoco>', ground, sky


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--kit', type=Path, required=True, help='directory with lib.py, env.py and tex/')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--histories', type=Path, required=True, help='source-rendered/vlm discovery histories')
    parser.add_argument('--skip-robots', default='inspire_hand', help='robots that keep the studio look')
    args = parser.parse_args()
    sys.path.insert(0, str(args.kit))
    import lib  # noqa: E402  (scene kit)
    import env  # noqa: E402
    args.out.mkdir(parents=True, exist_ok=True)
    counts, programs = {}, {}
    for line in args.tasks.read_text().splitlines():
        row = json.loads(line)
        if row['robot'] in args.skip_robots.split(','):
            continue
        render_path = Path(row['render_spec'])
        document = json.loads(render_path.read_text())
        if row['robot'] not in programs:
            programs[row['robot']] = skeletons(args.histories, row['robot'])
        setting = setting_for(programs[row['robot']][row['key']])
        seed = int(re.sub(r'\D', '', row['key']) or 0)
        xml, ground, sky = scene_xml(lib, env, setting, seed)
        decor = args.out / f"{row['key']}.json"
        decor.write_text(json.dumps({'xml': xml, 'ground': ground, 'setting': setting}))
        scene = document['render_scene']
        scene['website_decor'] = str(decor.resolve())
        scene['atmosphere'] = SKY[sky]
        scene['lighting'] = LIGHTS[sky]
        scene['camera'] = {'preset': 'robot_default', 'distance_scale': 1.0, 'elevation_deg': -12.0}
        render_path.write_text(json.dumps(document, indent=2))
        counts[setting] = counts.get(setting, 0) + 1
    print(counts)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
