#!/usr/bin/env bash
# Held-out evaluation on Slurm: runs one `python -m fosa.evaluation.heldout STEP ROBOT [ARGS...]` as a job, from the
# directory you submit from. A job requeues itself shortly before its walltime runs out; every step then resumes or
# reruns.
#
# usage: scripts/slurm_eval.sh [--dry-run] STEP ROBOT [ARGS...]
#   scripts/slurm_eval.sh prepare go2     # compile the held-out scenes (needs the LLM)
#   scripts/slurm_eval.sh train go2       # train their reference policies, one per GPU at a time
#   scripts/slurm_eval.sh evaluate go2 --checkpoint checkpoints/go2/student.pt
#   One GPU is enough for evaluate: FOSA_SLURM_GRES=gpu:1.
#   Settings: environment variables, listed in scripts/slurm_common.sh.
set -Eeuo pipefail
# shellcheck source=scripts/slurm_common.sh
source "${FOSA_REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}/scripts/slurm_common.sh"

if [[ "${1:-}" == --worker ]]; then
  activate_env
  run_job python -m fosa.evaluation.heldout "${@:2}"
else
  [[ "${1:-}" != --dry-run ]] || { DRY_RUN=1 && shift; }
  (($# >= 2)) || { usage && exit 2; }
  submit "${MODEL_NAME}-heldout-$1-$2" "$@"
fi
