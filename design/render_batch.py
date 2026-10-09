"""Render preview clips of trained FOSA task policies for the project website.

Read-only with respect to the discovery runs: specs and checkpoints are only loaded, and clips are
written under --out. Each task row may carry a `render_spec`: a copy of the task spec whose render
scene was authored by scripts/render_scenes.py (as for the held-out videos); the training spec
itself is never modified. Each clip renders in its own process.

    python render_batch.py --tasks tasks.jsonl --out clips --shard 0 --num-shards 2 --handvla <HandVLA>
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HANDVLA = Path(os.environ.get('FOSA_HANDVLA', '/N/slate/youwyu/HandVLA'))
sys.path.insert(0, str(HANDVLA))
sys.path.insert(0, str(HANDVLA / 'learning'))


def latest_checkpoint(log_dir: Path) -> Path:
    steps = sorted((p for p in (log_dir / 'checkpoints').iterdir() if p.name.isdigit()), key=lambda p: int(p.name))
    return steps[-1]


def render_one(row: dict, out: Path, width: int, height: int, duration: float, gpu: str = '0') -> dict:
    """Render one clip. Runs in its own process: compiled environments are never freed in-process,
    so a long-lived worker eventually exhausts GPU memory."""
    import ppo  # HandVLA/learning/ppo.py
    ppo.configure_gpu(gpu)
    from _src.mjp.compat import patch_numpy_for_mujoco_warp
    patch_numpy_for_mujoco_warp()
    from _src.mjp.registry import make_env
    from _src.mjp.rendering import render_policy_video

    spec = ppo.ensure_scene_spec_ready_for_training(Path(row['spec']))
    render_spec = Path(row.get('render_spec') or spec).resolve()
    checkpoint = latest_checkpoint(Path(row['log_dir']))
    env = make_env(row['task'], impl='warp', scene_spec=spec, contact_capacity=ppo.RENDER_CONTACT_CAPACITY)
    policy = ppo.load_checkpoint_policy(checkpoint)
    with tempfile.TemporaryDirectory() as tmp:
        outputs = render_policy_video(
            env=env, robot=row['robot'],
            make_policy=lambda _params, deterministic=True: policy, params=None,
            output_dir=Path(tmp), scene_spec=render_spec, seed=10_001,
            duration_s=duration, width=width, height=height,
        )
        if 'video' not in outputs:
            raise RuntimeError(outputs.get('error', 'no video'))
        shutil.copy(outputs['video'], out / f"{row['key']}.mp4")
    return {'checkpoint': checkpoint.name}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=360)
    parser.add_argument('--duration', type=float, default=6.0)
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--gpu', default='0', help='GPU index within the job for this shard')
    parser.add_argument('--one', help=argparse.SUPPRESS)  # internal: render this JSON row in-process
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.one:
        print(json.dumps(render_one(json.loads(args.one), args.out, args.width, args.height, args.duration, args.gpu)), flush=True)
        return 0

    rows = [json.loads(line) for line in args.tasks.read_text().splitlines() if line.strip()]
    rows = rows[args.shard::args.num_shards]
    if args.limit:
        rows = rows[:args.limit]
    status = (args.out / f'status-{args.shard}.jsonl').open('a')
    for row in rows:
        if (args.out / f"{row['key']}.mp4").exists():
            continue
        start = time.time()
        record = {'key': row['key']}
        command = [sys.executable, __file__, '--out', str(args.out), '--width', str(args.width),
                   '--height', str(args.height), '--duration', str(args.duration), '--gpu', args.gpu, '--one', json.dumps(row)]
        try:
            done = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
            if done.returncode != 0:
                raise RuntimeError(done.stderr.strip().splitlines()[-1] if done.stderr.strip() else f'exit {done.returncode}')
            record.update(json.loads(done.stdout.strip().splitlines()[-1]))
        except Exception as exc:
            record['error'] = repr(exc)[:500]
        record['seconds'] = round(time.time() - start, 1)
        print(json.dumps(record), flush=True)
        status.write(json.dumps(record) + '\n')
        status.flush()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
