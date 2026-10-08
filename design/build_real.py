"""Write the Sim-to-Real Held-out Tasks section from design/real_clips.json (made by the transcode step).

    python design/build_real.py

Clips are listed per robot in recording order; a badge shows the playback speed of sped-up clips. The section
goes between the <!-- real:start --> and <!-- real:end --> markers in index.html.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROBOTS = [('vehicle', 'Vehicle'), ('quadruped', 'Quadruped')]


def main():
    clips = json.loads((ROOT / 'design' / 'real_clips.json').read_text())
    tabs, panels = [], []
    for robot, label in ROBOTS:
        mine = [c for c in clips if c['robot'] == robot]
        tabs.append(f'<button type="button" role="tab" aria-controls="real-{robot}"><img src="./static/images/robots/real-{robot}.jpg" '
                    f'width="240" height="150" alt="" aria-hidden="true">{label}</button>')
        figures = []
        for n, c in enumerate(mine, 1):
            speed = c['speed']
            badge = f'<span class="speed-badge">×{speed}</span>' if speed > 1 else ''
            spoken = f', played at {speed} times speed' if speed > 1 else ''
            figures.append(
                f'<figure class="experiment real-clip">{badge}'
                f'<video poster="./static/images/posters/real/{c["name"]}.jpg" controls muted playsinline preload="none" '
                f'width="960" height="540" aria-label="{label} real-world trial {n}{spoken}">'
                f'<source src="./static/videos/real/{c["name"]}.mp4" type="video/mp4"></video></figure>')
        panels.append(f'<div class="robot-panel real-grid" id="real-{robot}" aria-label="{label} real-world trials">\n'
                      + '\n'.join(figures) + '\n</div>')
    section = ('<section class="paper-section band-sand" id="real" aria-labelledby="real-title">\n<div class="page-container">\n'
               '<h2 id="real-title">Sim-to-Real Held-out Tasks</h2>\n'
               '<div class="robot-tabs" role="tablist" aria-label="Robot" data-robot-tabs hidden>\n' + '\n'.join(tabs) + '\n</div>\n'
               + '\n'.join(panels) + '\n</div>\n</section>')
    page = (ROOT / 'index.html').read_text()
    start, end = page.index('<!-- real:start -->'), page.index('<!-- real:end -->')
    (ROOT / 'index.html').write_text(page[:start] + '<!-- real:start -->\n' + section + '\n    ' + page[end:])
    print(f'{len(clips)} real clips')


if __name__ == '__main__':
    main()
