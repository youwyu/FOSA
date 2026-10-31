#!/usr/bin/env bash
# Students on Slurm: one job per robot and method distils the method's task policies into one language-conditioned
# student. A student saves a checkpoint every round and resumes from the newest, so a job that runs out of walltime
# requeues and loses at most one round; a finished student is skipped.
#
# usage: scripts/slurm_student.sh [--dry-run] ROBOTS METHODS
#   ROBOTS and METHODS are comma-separated; METHODS may be "all", the paper's seven methods.
#   FOSA_STUDENT_SEED     student seed (default: 1; the paper uses 1, 2 and 3)
#   FOSA_STUDENT_ROOT     students go to <root>/<robot>/<method> (default: <log root>/students/seed<seed>)
#   The paper: for seed in 1 2 3; do FOSA_STUDENT_SEED=$seed scripts/slurm_student.sh go2,jackal,booster,inspire_hand all; done
#   Other settings: environment variables, listed in scripts/slurm_common.sh.
set -Eeuo pipefail
# shellcheck source=scripts/slurm_common.sh
source "${FOSA_REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}/scripts/slurm_common.sh"

readonly SEED="${FOSA_STUDENT_SEED:-1}"
readonly STUDENT_ROOT="${FOSA_STUDENT_ROOT:-${LOG_ROOT}/students/seed${SEED}}"
readonly FINAL=student_round_005.pt  # the last round of fosa.learning.dagger (CONFIG.rounds = 5)
readonly PAPER_METHODS=curriculum,mcts,llm_mcts,muzero,fosa_rnd,fosa_llm,fosa

finished() { [[ -s "${STUDENT_ROOT}/$1/$2/checkpoints/${FINAL}" ]]; }  # finished ROBOT METHOD

newest() { ls "${STUDENT_ROOT}/$1/$2"/checkpoints/student_round_*.pt 2>/dev/null | tail -1 || true; }  # newest ROBOT METHOD

run_worker() {  # run_worker ROBOT METHOD
  local robot="$1" method="$2" before
  ! finished "${robot}" "${method}" || return 0
  activate_env
  cd "${REPO_ROOT}"
  before="$(newest "${robot}" "${method}")"
  run_job python -m fosa.learning.dagger --robot "${robot}" --method "${method}" --seed "${SEED}" \
    --gpu-ids "$(seq -s, 0 $((${SLURM_GPUS_ON_NODE:-1} - 1)))" --resume-dir "${STUDENT_ROOT}/${robot}/${method}" || true
  ! finished "${robot}" "${method}" || return 0
  # No final checkpoint (busy GPUs, a dead rank): requeue, on another node if not even one round was saved here.
  scontrol requeue "${SLURM_JOB_ID}"
  [[ "$(newest "${robot}" "${method}")" != "${before}" ]] ||
    scontrol update JobId="${SLURM_JOB_ID}" ExcNodeList="$(hostname)"
}

main() {
  [[ "${1:-}" != --dry-run ]] || { DRY_RUN=1 && shift; }
  (($# == 2)) || { usage && exit 2; }
  local robots methods robot method list="$2"
  [[ "${list}" != all ]] || list="${PAPER_METHODS}"
  IFS=, read -ra robots <<<"$1"
  IFS=, read -ra methods <<<"${list}"
  for robot in "${robots[@]}"; do
    for method in "${methods[@]}"; do
      if finished "${robot}" "${method}"; then
        printf '%s %s already distilled\n' "${robot}" "${method}"
      else
        submit "student${SEED}-${robot}-${method}" "${robot}" "${method}"
      fi
    done
  done
}

if [[ "${1:-}" == --worker ]]; then run_worker "${@:2}"; else main "$@"; fi
