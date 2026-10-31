#!/usr/bin/env bash
# Discovery on Slurm: one job per robot and method, run until the method's history holds ITERATIONS trained tasks.
# The history is a run's only state: a job requeues itself before its walltime runs out, and resubmitting tops it up.
#
# usage: scripts/slurm.sh [--dry-run] ROBOTS METHODS ITERATIONS
#   ROBOTS and METHODS are comma-separated; METHODS may be "all", the paper's seven methods.
#   Methods: curriculum mcts llm_mcts muzero fosa_rnd fosa_llm fosa, and fosa_rnd_bonus (not in the paper).
#   The paper: scripts/slurm.sh go2,jackal,booster,inspire_hand all 256   (4 GPUs per job)
#   Settings: environment variables, listed in scripts/slurm_common.sh.
set -Eeuo pipefail
# shellcheck source=scripts/slurm_common.sh
source "${FOSA_REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}/scripts/slurm_common.sh"

readonly TASKS_PER_GPU=4  # PPO runs per GPU; a planner trains GPUs x this tasks per round
readonly BAD_GPU_EXIT_CODE=75  # fosa.discovery.common.BAD_GPU_EXIT_CODE: every PPO run on this node keeps failing
readonly PAPER_METHODS=curriculum,mcts,llm_mcts,muzero,fosa_rnd,fosa_llm,fosa
declare -rA METHODS=(  # method -> entry point and arguments
  [curriculum]="fosa.discovery.curriculum"
  [mcts]="fosa.discovery.mcts --method mcts"
  [llm_mcts]="fosa.discovery.mcts --method llm_mcts"
  [muzero]="fosa.discovery.mcts --method muzero"
  [fosa_rnd]="fosa.discovery.fosa --method fosa_rnd"
  [fosa_llm]="fosa.discovery.fosa --method fosa_llm"
  [fosa]="fosa.discovery.fosa --method fosa"
  [fosa_rnd_bonus]="fosa.discovery.fosa --method fosa --rnd-bonus"
)

trained() {  # trained ROBOT METHOD: the number of trained tasks in the history
  local history="${LOG_ROOT}/$1/$2_history.jsonl"
  if [[ -f "${history}" ]]; then grep -c '"status": "trained"' "${history}" || true; else echo 0; fi
}

run_worker() {  # run_worker ROBOT METHOD ITERATIONS
  local robot="$1" method="$2" total="$3" parallel status=0
  (($(trained "${robot}" "${method}") < total)) || return 0
  activate_env
  cd "${REPO_ROOT}"
  parallel=$((${SLURM_GPUS_ON_NODE:-1} * TASKS_PER_GPU))
  export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / parallel > 0 ? SLURM_CPUS_PER_TASK / parallel : 1))
  export MKL_NUM_THREADS="${OMP_NUM_THREADS}" OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
  # shellcheck disable=SC2086  # the entry point and its arguments are split on purpose
  run_job python -m ${METHODS[${method}]} --robot "${robot}" --iterations "${total}" --gpu-ids all || status=$?
  if ((status == BAD_GPU_EXIT_CODE)); then  # retry on another node
    scontrol requeue "${SLURM_JOB_ID}"
    scontrol update JobId="${SLURM_JOB_ID}" ExcNodeList="$(hostname)"
    return 0
  fi
  ((status == 0)) || exit "${status}"
  (($(trained "${robot}" "${method}") >= total)) || scontrol requeue "${SLURM_JOB_ID}"
}

main() {
  [[ "${1:-}" != --dry-run ]] || { DRY_RUN=1 && shift; }
  (($# == 3)) || { usage && exit 2; }
  local robots methods robot method list="$2"
  [[ "${list}" != all ]] || list="${PAPER_METHODS}"
  IFS=, read -ra robots <<<"$1"
  IFS=, read -ra methods <<<"${list}"
  for method in "${methods[@]}"; do [[ -v METHODS[${method}] ]] || die "unknown method: ${method}"; done
  for robot in "${robots[@]}"; do
    for method in "${methods[@]}"; do submit "${MODEL_NAME}-${robot}-${method}" "${robot}" "${method}" "$3"; done
  done
}

if [[ "${1:-}" == --worker ]]; then run_worker "${@:2}"; else main "$@"; fi
