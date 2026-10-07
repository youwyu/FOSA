"""Build the website's result data from the paper's frozen discovery logs.

Usage:
    python design/build_data.py \
        --histories <logs/figures/gemma/source-rendered/vlm> \
        --paper-final <logs/figures/paper-final>

Writes static/data/frontier-<robot>.js (one lazily loaded file per robot) and
static/data/summary.js. Frontier-Prox is recomputed per distinct program and
checked against SUMMARY.md, so the page cannot drift from the paper's numbers.
"""
import argparse
import collections
import json
import re
from pathlib import Path

ROBOTS = {
    'jackal': 'Vehicle',
    'inspire_hand': 'Hand',
    'go2': 'Quadruped',
    'booster': 'Humanoid',
}
# History file stem -> (paper name, SUMMARY.md row name, table row name).
METHODS = {
    'curriculum': ('LLM-CL', 'LLM-CL', 'LLM-CL'),
    'mcts': ('MCTS', 'MCTS', 'MCTS'),
    'llm_mcts': ('LLM-MCTS', 'LLM-MCTS', 'LLM-MCTS'),
    'muzero': ('MCTS-LEARN', 'MCTS-LEARN', 'MCTS-LEARN'),
    'fosa_grammar': ('FOSA-RND', 'FOSA-RND', 'FOSA-RND'),
    'fosa_llm': ('FOSA-LLM', 'FOSA-LLM', 'FOSA-LLM'),
    'fosa_learn': ('FOSA', 'FOSA', 'FOSA (ours)'),
}
AXES = ['Frontier-Prox', 'Expl-Eff', 'Dist-Embed1', 'Dist-Embed2', 'Div-SB4', 'Div-SB5']
OUT = Path(__file__).resolve().parents[1] / 'static' / 'data'


def frontier(j0, j):
    """Eq. (frontier-score): positive exactly when 0 <= J0 < J < 1."""
    if j <= 0:
        return 0.0
    return 4 * j * (1 - j) * max(0.0, j - j0) / j


def short_atom(atom, role):
    name = atom[: -len(role) - 1] if role and atom.endswith('_' + role) else atom
    return name.replace('_', ' ')


def format_args(arguments):
    parts = []
    for key, value in arguments.items():
        if isinstance(value, float):
            value = f'{value:g}'
        parts.append(f"{key.replace('_', ' ')} {value}")
    return ', '.join(parts)


def build_robot(histories, robot):
    atoms, atom_index = [], {}

    def atom_id(role, atom):
        key = (role, atom)
        if key not in atom_index:
            atom_index[key] = len(atoms)
            atoms.append([role, short_atom(atom, role)])
        return atom_index[key]

    methods = {}
    for stem in METHODS:
        programs = collections.OrderedDict()
        for line in (histories / robot / f'{stem}_history.jsonl').open():
            row = json.loads(line)
            if row.get('status') != 'trained' or not row.get('success_curve'):
                continue
            program = row.get('atom_program')
            bindings = program['bindings'] if program else None
            key = json.dumps(bindings, sort_keys=True) if bindings else row['spec_hash']
            programs.setdefault(key, {'rows': [], 'bindings': bindings})['rows'].append(row)
        tasks = []
        for item in programs.values():
            rows = item['rows']
            j0 = sum(r['success_curve'][0] for r in rows) / len(rows)
            j = sum(r['success_rate'] for r in rows) / len(rows)
            f = sum(frontier(r['success_curve'][0], r['success_rate']) for r in rows) / len(rows)
            skeleton = [[atom_id(b['role'], b['atom']), format_args(b.get('arguments') or {})]
                        for b in (item['bindings'] or [])]
            tasks.append([round(j0, 3), round(j, 3), round(f, 3), len(rows),
                          rows[0]['description'].strip(), skeleton])
        methods[stem] = tasks
    return {'atoms': atoms, 'methods': methods}


def parse_summary(path):
    """Radar tables in SUMMARY.md: {robot: {method: [[value, best], ...]}}."""
    radar, robot = {}, None
    names = {v: k for k, v in ROBOTS.items()}
    for line in path.read_text().splitlines():
        heading = re.fullmatch(r'\*\*(\w+)\*\*', line.strip())
        if heading and heading.group(1) in names:
            robot = names[heading.group(1)]
            if robot in radar:  # the held-out tables reuse the headings
                robot = None
            else:
                radar[robot] = {}
            continue
        if line.startswith('## Held-out'):
            break
        if robot and line.startswith('| ') and not line.startswith('| method'):
            cells = [c.strip() for c in line.strip('|').split('|')]
            values = [[float(c.strip('*')), c.startswith('**')] for c in cells[1:]]
            radar[robot][cells[0]] = values
    return radar


def parse_tex_rows(path):
    rows = {}
    for line in path.read_text().splitlines():
        if '&' not in line or line.lstrip().startswith(('Method', '\\')):
            continue
        cells = [c.strip() for c in line.rstrip('\\ ').split('&')]
        rows[cells[0]] = cells[1:]
    return rows


def parse_heldout(path):
    rows = {}
    for name, cells in parse_tex_rows(path).items():
        robots = []
        for cell in cells[:4]:
            match = re.search(r'([\d.]+) \\tiny\{\[([\d.]+), ([\d.]+)\]\}', cell)
            robots.append([float(match[1]), float(match[2]), float(match[3]), cell.startswith('\\textbf')])
        rows[name] = {'robots': robots, 'mean': float(cells[4]), 'above_half': cells[5]}
    return rows


def parse_alloc(path):
    return {name: [[float(c.replace('\\textbf{', '').rstrip('}')), c.startswith('\\textbf')] for c in cells]
            for name, cells in parse_tex_rows(path).items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--histories', type=Path, required=True)
    parser.add_argument('--paper-final', type=Path, required=True)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    radar = parse_summary(args.paper_final / 'SUMMARY.md')
    for robot in ROBOTS:
        data = build_robot(args.histories, robot)
        for stem, (_, summary_name, _) in METHODS.items():
            tasks = data['methods'][stem]
            prox = sum(t[2] for t in tasks) / len(tasks)
            expected = radar[robot][summary_name][0][0]
            assert abs(prox - expected) < 2e-3, (robot, stem, prox, expected)
        payload = json.dumps(data, separators=(',', ':'), ensure_ascii=False)
        (OUT / f'frontier-{robot}.js').write_text(
            f'(window.FOSA_FRONTIER=window.FOSA_FRONTIER||{{}})[{json.dumps(robot)}]={payload};\n')

    heldout = parse_heldout(args.paper_final / 'heldout6' / 'tables' / 'sim-policy.tex')
    alloc = parse_alloc(args.paper_final / 'tables' / 'alloc.tex')
    summary = {
        'robots': ROBOTS,
        'axes': AXES,
        'methods': {stem: names[0] for stem, names in METHODS.items()},
        'radar': {robot: {stem: radar[robot][names[1]] for stem, names in METHODS.items()} for robot in ROBOTS},
        'alloc': {stem: alloc[names[2]] for stem, names in METHODS.items() if names[2] in alloc},
        'heldout': {stem: heldout[names[2]] for stem, names in METHODS.items()},
    }
    payload = json.dumps(summary, separators=(',', ':'))
    (OUT / 'summary.js').write_text(f'window.FOSA_SUMMARY={payload};\n')
    for path in sorted(OUT.iterdir()):
        print(f'{path.name}: {path.stat().st_size / 1024:.0f} KiB')


if __name__ == '__main__':
    main()
