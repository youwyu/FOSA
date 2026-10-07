"""Build the frontier landscape shown on the website, one file per robot.

    PYTHONPATH=<numpy+scikit-learn> python design/build_landscape.py \
        --histories <logs/figures/gemma/source-rendered/vlm> --videos static/videos/landscape

Every skeleton evaluated by a grammar-based discovery run is embedded in 2D by the weighted
Jaccard distance between its bindings (t-SNE). The surface is the kernel-smoothed frontier
score F over that plane, estimated from all of them; the dots are FOSA's discovered tasks
that have a rendered clip in --videos.
"""
import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np
from sklearn.manifold import TSNE

ROBOTS = ['jackal', 'inspire_hand', 'go2', 'booster']
GRAMMAR_METHODS = ['mcts', 'llm_mcts', 'muzero', 'fosa_grammar', 'fosa_llm', 'fosa_learn']
GRID = 72
OUT = Path(__file__).resolve().parents[1] / 'static' / 'data'


def frontier(j0, j):
    return 0.0 if j <= 0 else 4 * j * (1 - j) * max(0.0, j - j0) / j


def tokens(bindings):
    weights = {}
    for binding in bindings:
        atom = f"{binding['role']}:{binding['atom']}"
        weights[atom] = 1.0
        for name, value in (binding.get('arguments') or {}).items():
            if isinstance(value, float):
                value = f'{value:.2g}'
            weights[f'{atom}:{name}={value}'] = 0.35
    return weights


def jaccard_distances(items):
    vocab = {t: i for i, t in enumerate(sorted({t for w in items for t in w}))}
    matrix = np.zeros((len(items), len(vocab)))
    for row, weights in enumerate(items):
        for token, weight in weights.items():
            matrix[row, vocab[token]] = weight
    distances = np.empty((len(items), len(items)))
    for row in range(len(items)):
        mins = np.minimum(matrix[row], matrix).sum(1)
        maxs = np.maximum(matrix[row], matrix).sum(1)
        distances[row] = 1.0 - mins / np.maximum(maxs, 1e-9)
    return distances


def load_programs(histories, robot):
    programs = collections.OrderedDict()
    for method in GRAMMAR_METHODS:
        seen = collections.OrderedDict()
        for line in (histories / robot / f'{method}_history.jsonl').open():
            row = json.loads(line)
            if row.get('status') != 'trained' or not row.get('atom_program'):
                continue
            key = json.dumps(row['atom_program']['bindings'], sort_keys=True)
            seen.setdefault(key, []).append(row)
        for index, (key, rows) in enumerate(seen.items()):
            entry = programs.setdefault(key, {'bindings': rows[0]['atom_program']['bindings'], 'scores': [], 'fosa': None})
            entry['scores'].extend(frontier(r['success_curve'][0], r['success_rate']) for r in rows)
            if method == 'fosa_learn':
                entry['fosa'] = {
                    'key': f'{robot}-{index:03d}',
                    'j0': sum(r['success_curve'][0] for r in rows) / len(rows),
                    'j': sum(r['success_rate'] for r in rows) / len(rows),
                    'f': sum(frontier(r['success_curve'][0], r['success_rate']) for r in rows) / len(rows),
                    'description': rows[0]['description'].strip(),
                }
    return list(programs.values())


def build(histories, videos, robot, all_dots=False):
    programs = load_programs(histories, robot)
    distances = jaccard_distances([tokens(p['bindings']) for p in programs])
    xy = TSNE(n_components=2, metric='precomputed', init='random', perplexity=40,
              random_state=0).fit_transform(distances)
    xy -= xy.min(0)
    xy /= xy.max(0)
    xy = xy * 0.84 + 0.08  # keep a margin of flat ground around the landscape
    score = np.array([np.mean(p['scores']) for p in programs])

    # Kernel-smoothed F, pulled to zero where few skeletons were evaluated.
    axis = np.linspace(0, 1, GRID)
    gx, gy = np.meshgrid(axis, axis)
    grid = np.stack([gx.ravel(), gy.ravel()], 1)
    bandwidth = 0.045
    kernel = np.exp(-((grid[:, None, :] - xy[None, :, :]) ** 2).sum(-1) / (2 * bandwidth ** 2))
    density = kernel.sum(1)
    height = (kernel @ score) / (density + 1.5)
    height /= height.max()

    def surface_at(point):
        k = np.exp(-((point[None, :] - xy) ** 2).sum(-1) / (2 * bandwidth ** 2))
        return float((k @ score) / (k.sum() + 1.5)) / float(((kernel @ score) / (density + 1.5)).max())

    dots = []
    for point, program in zip(xy, programs):
        fosa = program['fosa']
        if fosa is None or not (all_dots or (videos / f"{fosa['key']}.mp4").exists()):
            continue
        dots.append([round(float(point[0]), 4), round(float(point[1]), 4), round(surface_at(point), 4),
                     round(fosa['f'], 3), round(fosa['j0'], 3), round(fosa['j'], 3), fosa['key'], fosa['description']])
    payload = {'grid': GRID, 'height': [round(float(h), 4) for h in height], 'dots': dots,
               'evaluated': len(programs)}
    text = json.dumps(payload, separators=(',', ':'), ensure_ascii=False)
    (OUT / f'landscape-{robot}.js').write_text(
        f'(window.FOSA_LANDSCAPE=window.FOSA_LANDSCAPE||{{}})[{json.dumps(robot)}]={text};\n')
    return len(programs), len(dots)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--histories', type=Path, required=True)
    parser.add_argument('--videos', type=Path, required=True)
    parser.add_argument('--all-dots', action='store_true', help='Preview: keep FOSA dots without clips.')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for robot in ROBOTS:
        evaluated, dots = build(args.histories, args.videos, robot, args.all_dots)
        print(f'{robot}: {evaluated} evaluated skeletons, {dots} FOSA dots with clips')


if __name__ == '__main__':
    main()
