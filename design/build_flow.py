"""Build the construction figure from the trained GFlowNet's P_F (design/flow-jackal.json).

flow-jackal.json is the output of design/policy_flow.py (a read-only beam search over the trained
forward policy of FOSA's final vehicle run). This script writes the figure between the
<!-- flow:start --> and <!-- flow:end --> markers in index.html.
"""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).with_name('flow-jackal.json')


def name(option):
    atom = option.get('atom') or ''
    role = option.get('role') or ''
    return atom[: -len(role) - 1].replace('_', ' ') if atom.endswith('_' + role) else atom.replace('_', ' ')


def main():
    leaves = json.loads(SOURCE.read_text())['leaves']
    main_path, alt_path = leaves[1]['path'], leaves[2]['path']  # same skeleton, two edit orders
    same = lambda leaf: sorted(json.dumps(b, sort_keys=True) for b in leaf['bindings'])
    assert same(leaves[1]) == same(leaves[2]), 'the two edit orders must reach one skeleton'
    first = main_path[1]['options']                      # edits from {velocity}
    second = main_path[2]['options']                     # edits from {velocity, velocity tracking}
    p = {
        'velocity': main_path[0]['chosen']['p'],
        'A': first[0], 'B': first[1], 'C': main_path[1]['chosen'],
        'E': main_path[2]['chosen'], 'G': alt_path[2]['chosen'],
        'E_card': main_path[3]['chosen'], 'G_card': alt_path[3]['chosen'],
        'stop': main_path[4]['chosen'], 'refine': main_path[4]['options'][1],
        'more1': main_path[1]['valid'] - 3, 'more2': main_path[2]['valid'] - 2,
    }
    assert p['C'] in first and p['E'] in second and p['G'] in second

    def width(prob):
        return f'{1.6 + 10 * prob:.1f}'

    def prob(value):
        return f'{value:.2f}'.lstrip('0') if value < 1 else '1.00'

    def node(x, y, w, role, label, cls='', delay=0.0, h=48):
        return (f'<g class="f-node {cls}" style="--d:{delay}s"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10"/>'
                f'<rect class="f-role r-{role}" x="{x + 12}" y="{y + 8}" width="{len(role) * 6.4 + 12:.0f}" height="16" rx="4"/>'
                f'<text class="f-role-text" x="{x + 18}" y="{y + 20}">{role}</text>'
                f'<text class="f-label" x="{x + 12}" y="{y + 40}">{html.escape(label)}</text></g>')

    def edge(d, prob_value, cls='', delay=0.0, label=None, lx=0, ly=0, anchor='middle'):
        out = f'<path class="f-edge {cls}" style="--d:{delay}s; stroke-width:{width(prob_value)}" pathLength="1" d="{d}"/>'
        if label is not None:
            out += f'<text class="f-prob {cls}" style="--d:{delay}s" x="{lx}" y="{ly}" text-anchor="{anchor}">{label}</text>'
        return out

    def curve(x1, y1, x2, y2):
        return f'M{x1},{y1} C{x1 + 46},{y1} {x2 - 46},{y2} {x2},{y2}'

    parts = [
        # Faint alternatives first, so the most probable path draws on top.
        edge(curve(184, 190, 256, 64), p['A']['p'], 'is-alt', .9, prob(p['A']['p']), 248, 50, 'end'),
        edge(curve(184, 190, 256, 136), p['B']['p'], 'is-alt', .9, prob(p['B']['p']), 248, 122, 'end'),
        edge(curve(184, 190, 256, 279), .02, 'is-stub', .9),
        node(256, 40, 184, p['A']['role'], name(p['A']), 'is-alt', .9),
        node(256, 112, 184, p['B']['role'], name(p['B']), 'is-alt', .9),
        f'<text class="f-more is-alt" style="--d:.9s" x="262" y="284">+{p["more1"]} more edits</text>',
        edge(curve(440, 208, 500, 260), p['G']['p'], 'is-alt', 1.4, prob(p['G']['p']), 494, 250, 'end'),
        edge(curve(440, 208, 500, 313), .02, 'is-stub', 1.4),
        node(500, 236, 184, p['G']['role'], name(p['G']), 'is-alt', 1.4),
        f'<text class="f-more is-alt" style="--d:1.4s" x="506" y="318">+{p["more2"]} more edits</text>',
        edge(curve(684, 260, 790, 236), p['G_card']['p'], 'is-alt', 1.9,
             f'+ {name(p["G_card"])} {prob(p["G_card"]["p"])}', 784, 300, 'end'),
        edge('M980,136 L980,88', p['refine']['p'], 'is-alt', 2.4),
        (f'<g class="f-node is-alt" style="--d:2.4s"><rect x="890" y="44" width="180" height="44" rx="10"/>'
         f'<text class="f-small" x="902" y="62"><tspan class="f-op">refine_arg</tspan> {prob(p["refine"]["p"])}</text>'
         f'<text class="f-small" x="902" y="79">completion time → {p["refine"]["result"]:g} s</text></g>'),
        # The most probable path.
        '<g class="f-node is-main f-empty" style="--d:0s"><circle cx="30" cy="190" r="20"/><text x="30" y="196" text-anchor="middle">∅</text></g>',
        edge('M50,190 L72,190', p['velocity'], 'is-main', .2),
        node(72, 166, 112, 'capability', 'velocity', 'is-main', .4),
        edge(curve(184, 190, 256, 208), p['C']['p'], 'is-main', .7, prob(p['C']['p']), 248, 236, 'end'),
        node(256, 184, 184, p['C']['role'], name(p['C']), 'is-main', 1.0),
        edge(curve(440, 208, 500, 174), p['E']['p'], 'is-main', 1.3, prob(p['E']['p']), 494, 162, 'end'),
        node(500, 150, 184, p['E']['role'], name(p['E']), 'is-main', 1.6),
        edge(curve(684, 174, 790, 196), p['E_card']['p'], 'is-main', 1.9,
             f'+ {name(p["E_card"])} {prob(p["E_card"]["p"])}', 784, 142, 'end'),
    ]
    rows = [('capability', 'velocity'),
            ('constraint', 'velocity tracking flat'),
            ('environment', 'friction adaptation'),
            ('goal', 'slalom navigation')]
    card = ['<g class="f-node f-card is-main" style="--d:2.1s"><rect x="790" y="136" width="280" height="150" rx="12"/>',
            '<text class="f-card-title" x="804" y="158">SKELETON <tspan class="f-math">x</tspan></text>']
    for i, (role, text) in enumerate(rows):
        y = 170 + i * 27
        card.append(f'<rect class="f-role r-{role}" x="804" y="{y}" width="{len(role) * 6.4 + 12:.0f}" height="16" rx="4"/>'
                    f'<text class="f-role-text" x="810" y="{y + 12}">{role}</text>'
                    f'<text class="f-row" x="{804 + len(role) * 6.4 + 20:.0f}" y="{y + 12.5}">{html.escape(text)}</text>')
    card.append('</g>')
    parts += card
    parts += [
        edge('M930,286 L930,330', p['stop']['p'], 'is-main', 2.5, f'stop {prob(p["stop"]["p"])}', 944, 314, 'start'),
        ('<g class="f-node f-reward is-main" style="--d:2.8s"><rect x="830" y="330" width="200" height="44" rx="10"/>'
         '<text x="930" y="357" text-anchor="middle">compile · PPO · <tspan font-style="italic">R</tspan>(<tspan font-weight="700">x</tspan>)</text></g>'),
        edge('M830,352 L56,352 Q30,352 30,326 L30,210', .25, 'is-loop', 3.1, 'trajectory balance updates the GFlowNet', 430, 342, 'middle'),
    ]
    svg = ('<svg class="flow-dag" viewBox="0 0 1080 384" role="img" aria-label="The trained GFlowNet constructing a vehicle task: '
           'from the empty skeleton it adds velocity, then velocity tracking, then friction adaptation or slalom navigation in either order, '
           'reaching the same skeleton, and stops with probability 0.60.">\n' + '\n'.join(parts) + '\n</svg>')

    list_items = [
        '<li class="edit-start"><span class="empty-node" aria-hidden="true">∅</span>empty skeleton</li>',
    ]
    for i, step in enumerate(main_path):
        c = step['chosen']
        if c['op'] == 'stop':
            body = '<code class="edit-op">stop</code>'
        else:
            body = f'<code class="edit-op">{c["op"]}</code><span class="edit-role">{c["role"]}</span>{html.escape(name(c))}'
        list_items.append(f'<li style="--p:{c["p"]}; --i:{i + 1}"><span class="edit-body">{body}</span><span class="edit-prob">{prob(c["p"])}</span></li>')
    edits = '<ol class="edits flow-list" aria-hidden="true">\n' + '\n'.join(list_items) + '\n</ol>'

    figure = ('<figure class="flow">\n' + svg + '\n' + edits +
              '\n<figcaption>Vehicle · trained GFlowNet · width ∝ <var>P</var><sub>F</sub>(<b>x</b>′ | <b>x</b>)</figcaption>\n</figure>')
    page = (ROOT / 'index.html').read_text()
    start, end = page.index('<!-- flow:start -->'), page.index('<!-- flow:end -->')
    (ROOT / 'index.html').write_text(page[:start] + '<!-- flow:start -->\n' + figure + '\n        ' + page[end:])
    print('flow figure written')


if __name__ == '__main__':
    main()
