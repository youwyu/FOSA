"""Cut the successful real-world task segments listed in design/real_tasks.json from the original iPhone footage.

    python design/cut_real.py --footage ~/Downloads

Every segment comes from a FOSA (Ours) trial, as identified in the real-world demo slides (the vehicle's four tasks
from its single course run; the quadruped's grass and mug segments use the slides' trims). Barrier balance is not
listed: FOSA's trial there was not a success (Table 2). HLG/BT.2020 footage is tone-mapped to SDR BT.709; output is 960x540 at 30 fps.
Runs one ffmpeg at a time at the lowest CPU priority.
"""
import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HDR = ('zscale=w=960:h=540:filter=lanczos,zscale=t=linear:npl=203,format=gbrpf32le,zscale=p=bt709,'
       'tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p')
SDR = 'scale=960:540:flags=lanczos,format=yuv420p'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--footage', type=Path, required=True, help='folder with vehicle/ and quadruped/')
    args = parser.parse_args()
    videos, posters = ROOT / 'static/videos/real', ROOT / 'static/images/posters/real'
    videos.mkdir(parents=True, exist_ok=True)
    posters.mkdir(parents=True, exist_ok=True)
    for item in json.loads((ROOT / 'design/real_tasks.json').read_text()):
        src = args.footage / item['source']
        transfer = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=color_transfer',
                                   '-of', 'csv=p=0', str(src)], capture_output=True, text=True).stdout
        chain = HDR if ('arib-std-b67' in transfer or 'smpte2084' in transfer) else SDR
        name = f"{item['robot']}-{item['task']}"
        dst = videos / f'{name}.mp4'
        subprocess.run(['nice', '-n', '19', 'ffmpeg', '-nostdin', '-loglevel', 'error', '-y', '-threads', '6',
                        '-ss', str(item['t0']), '-t', str(item['t1'] - item['t0']), '-i', str(src), '-an',
                        '-vf', f"{chain},setpts=PTS/{item['speed']},fps=30", '-c:v', 'libx264', '-preset', 'slow',
                        '-crf', '24', '-maxrate', '2200k', '-bufsize', '4400k', '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart', '-threads', '6', str(dst)], check=True)
        subprocess.run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-y', '-ss', '2', '-i', str(dst), '-frames:v', '1',
                        '-q:v', '5', str(posters / f'{name}.jpg')], check=True)
        print(name, f"{dst.stat().st_size // 1024} KB", flush=True)


if __name__ == '__main__':
    main()
