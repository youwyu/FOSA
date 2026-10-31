"""PPO teacher training (fosa.learning.ppo) and DAgger student distillation (fosa.learning.dagger)."""

import os
from pathlib import Path


def use_scratch_tmpdir() -> None:
    """Point TMPDIR/TEMP/TMP at FOSA_TMPDIR (default /dev/shm/fosa-<uid>), node-local scratch for runtime files."""
    path = Path(os.environ.get("FOSA_TMPDIR", f"/dev/shm/fosa-{os.getuid()}")).expanduser()
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    resolved = str(path.resolve())
    for name in ("TMPDIR", "TEMP", "TMP"):
        os.environ[name] = resolved
