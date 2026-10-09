#!/bin/bash
# Rich-scene clips (1 GPU per job). TASKS/SHARD/NSHARD from --export; the gpu array walks the list in reverse.
#SBATCH -A r00338 --gres=gpu:1 -c 16 --mem=64G
#SBATCH -o /N/slate/youwyu/fosa_web/logs/rich2-%j.out -e /N/slate/youwyu/fosa_web/logs/rich2-%j.err
PS1=${PS1:-}
module load conda/26.3.2; conda activate fosa
export PYTHONNOUSERSITE=1 MUJOCO_GL=egl FOSA_HANDVLA=/N/slate/youwyu/fosa_web/HandVLA_render
cd /N/slate/youwyu/fosa_web
python render_batch.py --tasks ${TASKS:-tasks_cluster_rich.jsonl} --out clips_rich2 --shard ${SHARD:-${SLURM_ARRAY_TASK_ID}} \
    --num-shards ${NSHARD:-5} --width 960 --height 540 --duration 8
