"""Write the Sim-to-Real Held-out Tasks section from design/real_tasks.json (clips made by design/cut_real.py).

    python design/build_real.py

One captioned clip per successful held-out task, in the paper's order; a badge marks sped-up clips. The section goes
between the <!-- real:start --> and <!-- real:end --> markers in index.html.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROBOTS = [('vehicle', 'Vehicle'), ('quadruped', 'Quadruped')]


def fmt(speed):
    return f'{speed:g}'


def main():
    tasks = json.loads((ROOT / 'design' / 'real_tasks.json').read_text())
    tabs, panels = [], []
    for index, (robot, label) in enumerate(ROBOTS):
        tabs.append(f'<button type="button" role="tab" aria-controls="real-{robot}"><img src="./static/images/robots/real-{robot}.jpg" '
                    f'width="240" height="150" alt="" aria-hidden="true">{label}</button>')
        poster_attr = 'poster' if index == 0 else 'data-poster'  # hidden tabs load their posters when opened
        figures = []
        for item in (t for t in tasks if t['robot'] == robot):
            name = f"{robot}-{item['task']}"
            speed = item['speed']
            badge = f'<span class="speed-badge">×{fmt(speed)}</span>' if speed > 1 else ''
            spoken = f', played at {fmt(speed)} times speed' if speed > 1 else ''
            figures.append(
                f'<figure class="experiment real-clip">{badge}'
                f'<video {poster_attr}="./static/images/posters/real/{name}.jpg" controls muted playsinline preload="none" '
                f'width="960" height="540" aria-label="{label} real-world task: {item["label"].lower()}{spoken}">'
                f'<source src="./static/videos/real/{name}.mp4" type="video/mp4"></video>'
                f'<figcaption>{item["label"]}</figcaption></figure>')
        panels.append(f'<div class="robot-panel experiment-grid real-grid" id="real-{robot}" aria-label="{label} real-world tasks">\n'
                      + '\n'.join(figures) + '\n</div>')
    section = ('<section class="paper-section band-sand" id="real" aria-labelledby="real-title">\n<div class="page-container">\n'
               '<h2 id="real-title">Sim-to-Real Held-out Tasks</h2>\n'
               '<div class="robot-tabs" role="tablist" aria-label="Robot" data-robot-tabs hidden>\n' + '\n'.join(tabs) + '\n</div>\n'
               + '\n'.join(panels) + '\n</div>\n</section>')
    page = (ROOT / 'index.html').read_text()
    start, end = page.index('<!-- real:start -->'), page.index('<!-- real:end -->')
    (ROOT / 'index.html').write_text(page[:start] + '<!-- real:start -->\n' + section + '\n    ' + page[end:])
    print(f'{len(tasks)} real-world tasks')


if __name__ == '__main__':
    main()
