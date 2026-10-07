#!/bin/bash
#SBATCH -A r00338 -p gpu --gres=gpu:1 -c 16 --mem=64G -t 03:00:00
#SBATCH -J fosa-web-render --array=0-7
#SBATCH -o /N/slate/youwyu/fosa_web/logs/render-%A_%a.out -e /N/slate/youwyu/fosa_web/logs/render-%A_%a.err
PS1=${PS1:-}
module load conda/26.3.2; conda activate fosa
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl
cd /N/slate/youwyu/fosa_web
python render_batch.py --tasks landscape_tasks.jsonl --out renders --shard ${SLURM_ARRAY_TASK_ID} --num-shards 8
