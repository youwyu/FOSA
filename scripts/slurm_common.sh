# Sourced by the Slurm launchers: site settings, job submission and the job environment.
#
# Settings (environment variables; all optional):
#   FOSA_SLURM_ACCOUNT    account to charge (default: your default account)
#   FOSA_SLURM_PARTITION  GPU partition (default: gpu)
#   FOSA_SLURM_GRES       GPUs per job (default: gpu:4, the paper's setting)
#   FOSA_SLURM_TIME       walltime; a job requeues itself shortly before it runs out (default: 2-00:00:00)
#   FOSA_SLURM_DEPENDENCY, FOSA_SLURM_EXCLUDE   passed to sbatch --dependency and --exclude
#   FOSA_CPUS_PER_JOB     (default: 16 per GPU)
#   FOSA_CONDA_ENV        (default: fosa)
#   FOSA_CONDA_MODULE     environment module that provides conda, if your site needs one
#   FOSA_LLM_MODEL, FOSA_LOG_ROOT and the other FOSA_LLM_* variables reach the jobs unchanged.

REPO_ROOT="${FOSA_REPO_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
LLM_MODEL="${FOSA_LLM_MODEL:-gemma-4-31B-it}"  # fosa.llm.DEFAULT_MODEL
MODEL_NAME="$(sed -E 's/[^A-Za-z0-9._-]+/_/g; s/^[._-]+|[._-]+$//g' <<<"${LLM_MODEL}")"  # fosa.llm.model_log_name
LOG_ROOT="${FOSA_LOG_ROOT:-${REPO_ROOT}/logs/${MODEL_NAME}}"  # fosa.llm.model_log_root
GRES="${FOSA_SLURM_GRES:-gpu:4}"
DRY_RUN=0
export FOSA_REPO_ROOT="${REPO_ROOT}" FOSA_LLM_MODEL="${LLM_MODEL}" FOSA_LOG_ROOT="${LOG_ROOT}"

die() { printf 'Error: %s\n' "$*" >&2; exit 1; }

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }  # the launcher's header

submit() {  # submit NAME ARGS...: run this launcher with --worker ARGS as one Slurm job (printed with --dry-run)
  local name="$1"
  shift
  local command=(
    sbatch --parsable ${FOSA_SLURM_ACCOUNT:+--account="${FOSA_SLURM_ACCOUNT}"} --partition="${FOSA_SLURM_PARTITION:-gpu}"
    --nodes=1 --ntasks=1 --gres="${GRES}" --cpus-per-task="${FOSA_CPUS_PER_JOB:-$((${GRES##*:} * 16))}"
    --time="${FOSA_SLURM_TIME:-2-00:00:00}" --signal=B:USR1@300 --requeue --open-mode=append
    ${FOSA_SLURM_DEPENDENCY:+--dependency="${FOSA_SLURM_DEPENDENCY}"} ${FOSA_SLURM_EXCLUDE:+--exclude="${FOSA_SLURM_EXCLUDE}"}
    --job-name="${name}" --output="${LOG_ROOT}/slurm/${name}-%j.out" --error="${LOG_ROOT}/slurm/${name}-%j.err"
    "$0" --worker "$@"
  )
  if ((DRY_RUN)); then
    printf '%q ' "${command[@]}"
    printf '\n'
  else
    mkdir -p "${LOG_ROOT}/slurm"
    printf '%s %s\n' "${name}" "$("${command[@]}")"
  fi
}

activate_env() {  # in a job: the conda environment
  set +u
  [[ ! -f /etc/profile.d/modules.sh ]] || source /etc/profile.d/modules.sh
  [[ -z "${FOSA_CONDA_MODULE}" ]] || module load "${FOSA_CONDA_MODULE}"
  conda activate "${FOSA_CONDA_ENV:-fosa}" || die "missing conda environment ${FOSA_CONDA_ENV:-fosa}"
  set -u
  export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1 MUJOCO_GL="${MUJOCO_GL:-egl}"
}

run_job() {  # run_job COMMAND...: run COMMAND; when Slurm signals that the walltime is near, stop it and requeue
  # Bash runs a trap only between foreground commands, so COMMAND runs in the background and is waited on.
  trap 'kill "${job}" 2>/dev/null; scontrol requeue "${SLURM_JOB_ID}"; exit 0' USR1
  "$@" &
  job=$!
  wait "${job}"
}
