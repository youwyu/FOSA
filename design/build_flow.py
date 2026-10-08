"""Build the construction figure from the tasks FOSA actually sampled on the vehicle.

    python design/build_flow.py --history <logs/figures/gemma/source-rendered/vlm/jackal/fosa_learn_history.jsonl>

Every one of FOSA's 256 sampled vehicle tasks flows through the semantics its skeleton binds
(capability, goal, environment, constraint) into the outcome of PPO training: band width is the
number of tasks. The counts are saved to design/flow-jackal.json, and the figure is written
between the <!-- flow:start --> and <!-- flow:end --> markers in index.html.
"""
import argparse
import collections
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROLES = ['capability', 'goal', 'environment', 'constraint']
OUTCOMES = {'frontier': 'on the frontier', 'mastered': 'mastered', 'unlearned': 'not learned'}
COLUMN_X = [30, 238, 458, 678, 900]
TOP, HEIGHT, GAP, BAR = 44, 300, 7, 14


def frontier(j0, j):
    return 0.0 if j <= 0 else 4 * j * (1 - j) * max(0.0, j - j0) / j


def short(atom, role):
    return (atom[: -len(role) - 1] if atom.endswith('_' + role) else atom).replace('_', ' ')


def outcome(row):
    j0, j = row['success_curve'][0], row['success_rate']
    return 'mastered' if j >= 1 else 'frontier' if j0 < j else 'unlearned'


def load(history):
    tasks = []
    for line in history.read_text().splitlines():
        row = json.loads(line)
        if row.get('status') != 'trained':
            continue
        roles = {b['role']: short(b['atom'], b['role']) for b in row['atom_program']['bindings']}
        tasks.append({
            'path': [roles[r] for r in ROLES] + [outcome(row)],
            'f': frontier(row['success_curve'][0], row['success_rate']),
            'j0': row['success_curve'][0], 'j': row['success_rate'],
            'description': row['description'].strip(),
            'actions': row['actions'],
            'logprobs': row.get('sampled_action_logprobs') or [],
        })
    return tasks


def edit_list(example):
    items = ['<li class="edit-start"><span class="empty-node" aria-hidden="true">∅</span>empty skeleton</li>']
    for i, action in enumerate(example['actions']):
        if action['action_type'] == 'stop':
            body = '<code class="edit-op">stop</code>'
        elif action['action_type'] == 'refine_arg':
            arg = action['arguments']
            change = f"→ {arg['value']}" if 'value' in arg else ('↑' if arg.get('direction') == 'high' else '↓')
            body = (f'<code class="edit-op">refine_arg</code><span class="edit-role">{action["role"]}</span>'
                    f'{html.escape(arg["argument"].replace("_", " "))} {html.escape(str(change))}')
        else:
            body = (f'<code class="edit-op">{action["action_type"]}</code><span class="edit-role">{action["role"]}</span>'
                    f'{html.escape(short(action["atom"], action["role"]))}')
        items.append(f'<li style="--p:.35; --i:{i + 1}"><span class="edit-body">{body}</span></li>')
    return '<ol class="edits flow-edits" aria-label="Edits of the highlighted task">\n' + '\n'.join(items) + '\n</ol>'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', type=Path, required=True)
    args = parser.parse_args()
    tasks = load(args.history)
    total = len(tasks)

    # The example: from the most common frontier composition, the clearest trajectory (one refinement,
    # so every edit type appears once), then the highest frontier score, then the fewest edits.
    frontier_paths = collections.Counter(tuple(t['path'][:4]) for t in tasks if t['path'][4] == 'frontier')
    best_path = frontier_paths.most_common(1)[0][0]
    refinements = lambda t: sum(a['action_type'] == 'refine_arg' for a in t['actions'])
    example = max((t for t in tasks if tuple(t['path'][:4]) == best_path and t['path'][4] == 'frontier'),
                  key=lambda t: (refinements(t) == 1, round(t['f'], 2), -len(t['actions'])))

    columns = []
    for c in range(5):
        counts = collections.Counter(t['path'][c] for t in tasks)
        order = [k for k, _ in counts.most_common()] if c < 4 else [k for k in OUTCOMES if k in counts]
        columns.append((order, counts))
    scale = (HEIGHT - GAP * (max(len(o) for o, _ in columns) - 1)) / total
    nodes = {}
    for c, (order, counts) in enumerate(columns):
        span = sum(counts[k] for k in order) * scale + GAP * (len(order) - 1)
        y = TOP + (HEIGHT - span) / 2
        for key in order:
            h = counts[key] * scale
            nodes[(c, key)] = {'y': y, 'h': h, 'n': counts[key], 'out': y, 'in': y}
            y += h + GAP

    links = collections.Counter((c, t['path'][c], t['path'][c + 1]) for t in tasks for c in range(4))
    rank = {(c, k): i for c, (order, _) in enumerate(columns) for i, k in enumerate(order)}
    parts = []
    for (c, a, b), n in sorted(links.items(), key=lambda kv: (kv[0][0], rank[(kv[0][0], kv[0][1])], rank[(kv[0][0] + 1, kv[0][2])])):
        src, dst, h = nodes[(c, a)], nodes[(c + 1, b)], n * scale
        x1, x2 = COLUMN_X[c] + BAR, COLUMN_X[c + 1]
        y1, y2 = src['out'], dst['in']
        src['out'] += h
        dst['in'] += h
        mid = (x1 + x2) / 2
        cls = 'f-link is-main' if (a, b) == (example['path'][c], example['path'][c + 1]) else 'f-link'
        parts.append(f'<path class="{cls}" style="--d:{0.18 * c:.2f}s" d="M{x1},{y1:.1f} C{mid},{y1:.1f} {mid},{y2:.1f} {x2},{y2:.1f} '
                     f'L{x2},{y2 + h:.1f} C{mid},{y2 + h:.1f} {mid},{y1 + h:.1f} {x1},{y1 + h:.1f} Z">'
                     f'<title>{html.escape(a)} → {html.escape(OUTCOMES.get(b, b))}: {n} tasks</title></path>')
    for c, (order, counts) in enumerate(columns):
        role = ROLES[c] if c < 4 else 'outcome'
        anchor, lx = 'start', COLUMN_X[c] + BAR + 8
        for key in order:
            node = nodes[(c, key)]
            on_path = key == example['path'][c]
            name = OUTCOMES.get(key, key) if c == 4 else key
            fill = f'o-{key}' if c == 4 else f'r-{role}'
            parts.append(f'<g class="f-node{" is-main" if on_path else ""}" style="--d:{0.18 * c:.2f}s">'
                         f'<rect class="{fill}" x="{COLUMN_X[c]}" y="{node["y"]:.1f}" width="{BAR}" height="{max(node["h"], 1.5):.1f}" rx="2"/>')
            if node['h'] >= 9 or on_path:
                parts.append(f'<text class="f-name" x="{lx}" y="{node["y"] + node["h"] / 2 + 4.5:.1f}" text-anchor="{anchor}">'
                             f'{html.escape(name)} <tspan class="f-count">{node["n"]}</tspan></text>')
            parts.append('</g>')
    headers = ''.join(f'<text class="f-head" x="{x}" y="22">{r}</text>' for r, x in zip(ROLES + ['after PPO'], COLUMN_X))
    svg = ('<svg class="flow-dag" viewBox="0 0 1080 360" role="img" aria-label="'
           f'Semantics of the {total} vehicle tasks FOSA sampled: the velocity capability, {len(columns[1][0])} goal families, '
           f'{len(columns[2][0])} environments and {len(columns[3][0])} constraints, then the outcome of PPO training.">\n'
           + headers + '\n' + '\n'.join(parts) + '\n</svg>')

    example_html = (f'<div class="flow-example"><p class="flow-example-label">Example task</p>{edit_list(example)}'
                    f'<q>{html.escape(example["description"])}</q></div>')
    figure = '<figure class="flow">\n<div class="flow-grid">\n' + svg + '\n' + example_html + '\n</div>\n</figure>'
    page = (ROOT / 'index.html').read_text()
    start, end = page.index('<!-- flow:start -->'), page.index('<!-- flow:end -->')
    (ROOT / 'index.html').write_text(page[:start] + '<!-- flow:start -->\n' + figure + '\n        ' + page[end:])
    summary = {'total': total, 'columns': [{k: columns[c][1][k] for k in columns[c][0]} for c in range(5)],
               'links': {f'{c}|{a}|{b}': n for (c, a, b), n in links.items()},
               'example': {k: example[k] for k in ('path', 'f', 'j0', 'j', 'description', 'actions', 'logprobs')}}
    Path(__file__).with_name('flow-jackal.json').write_text(json.dumps(summary, indent=1))
    print(f'{total} tasks; highlighted {example["path"]} F={example["f"]:.2f}')


if __name__ == '__main__':
    main()
