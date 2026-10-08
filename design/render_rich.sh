#!/bin/bash
#SBATCH -A r00338 -p gpu --gres=gpu:1 -c 16 --mem=64G -t 06:00:00
#SBATCH -J fosa-web-rich --array=0-7
#SBATCH -o /N/slate/youwyu/fosa_web/logs/rich-%A_%a.out -e /N/slate/youwyu/fosa_web/logs/rich-%A_%a.err
PS1=${PS1:-}
module load conda/26.3.2; conda activate fosa
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl FOSA_HANDVLA=/N/slate/youwyu/fosa_web/HandVLA_render
cd /N/slate/youwyu/fosa_web
python render_batch.py --tasks tasks_cluster_run.jsonl --out clips_rich --shard ${SLURM_ARRAY_TASK_ID} --num-shards 8 --width 960 --height 540 --duration 8
