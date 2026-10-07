"""Write one consistent render scene into copies of the task specs (no LLM, no decorative props).

    python design/house_style.py --tasks tasks_local.jsonl

For every row, the training spec (`spec`) is copied to `render_spec` with its render scene replaced
by a curated style: a warm sky, soft key/fill lights with shadows, a matte checker ground that
shows motion, and a closer camera. The task's own terrain and objects come from the training scene.
"""
import argparse
import copy
import json
from pathlib import Path

GROUND = {  # matte, mid-tone grounds; the checker accent reads motion without glare
    'jackal': ([0.56, 0.50, 0.43, 1.0], [0.47, 0.42, 0.36, 1.0]),
    'go2': ([0.50, 0.53, 0.47, 1.0], [0.42, 0.45, 0.40, 1.0]),
    'booster': ([0.55, 0.49, 0.44, 1.0], [0.46, 0.41, 0.37, 1.0]),
    'inspire_hand': ([0.60, 0.55, 0.50, 1.0], [0.52, 0.47, 0.43, 1.0]),
}
CAMERA = {
    'jackal': {'preset': 'robot_default', 'distance_scale': 0.9},
    'go2': {'preset': 'robot_default', 'distance_scale': 0.8},
    'booster': {'preset': 'robot_default', 'distance_scale': 0.85},
    # The hand floats freely; the local renderer follows its root body (see README).
    'inspire_hand': {'preset': 'robot_default', 'distance_scale': 1.5},
}


def house_scene(template: dict, robot: str) -> dict:
    scene = copy.deepcopy(template)
    ground, accent = GROUND[robot]
    scene.update({
        'source': 'fosa_web_house_style',
        'scene_style': 'fosa_web_house_style',
        'atmosphere': {'sky_zenith_rgb': [0.40, 0.55, 0.70], 'sky_horizon_rgb': [0.88, 0.82, 0.74],
                       'fog_rgba': [0.86, 0.80, 0.72, 0.10]},
        'lighting': {
            'ambient_rgb': [0.26, 0.25, 0.24],
            'headlight_diffuse_rgb': [0.30, 0.30, 0.30],
            'lights': [
                {'name': 'key', 'position_m': [4.0, -4.0, 7.0], 'direction': [-0.45, 0.45, -1.0],
                 'color': [1.0, 0.93, 0.84, 1.0], 'intensity': 1.0, 'castshadow': True},
                {'name': 'fill', 'position_m': [-4.0, 4.0, 5.0], 'direction': [0.5, -0.5, -1.0],
                 'color': [0.80, 0.86, 1.0, 1.0], 'intensity': 0.35, 'castshadow': False},
            ],
        },
        'materials': {'ground_rgba': ground, 'ground_accent_rgba': accent, 'ground_texture': 'checker',
                      'ground_reflectance': 0.02, 'accent_palette': [[0.75, 0.34, 0.0, 1.0], [0.25, 0.50, 0.55, 1.0]]},
        'camera': CAMERA[robot],
        'scene_layout': {'style': 'fosa_web_house_style', 'show_path_markers': True, 'assemblies': []},
        'objects': [],
    })
    return scene


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', type=Path, required=True)
    args = parser.parse_args()
    for line in args.tasks.read_text().splitlines():
        row = json.loads(line)
        document = json.loads(Path(row['spec']).read_text())
        document['render_scene'] = house_scene(document.get('render_scene') or {}, row['robot'])
        Path(row['render_spec']).write_text(json.dumps(document, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
