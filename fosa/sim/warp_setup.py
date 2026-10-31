"""MuJoCo Warp setup: kernels cached in the repository, quiet logging."""

from __future__ import annotations

import os
import warnings
from fosa import REPO_ROOT


def configure_warp() -> None:
    os.environ.setdefault("WARP_CACHE_PATH", str(REPO_ROOT / "cache" / "warp"))
    warnings.filterwarnings("ignore", message=r".*warp\.types\.warp_type_to_np_dtype.*")
    import warp

    warp.config.quiet = True
    warp.config.verbose = False
    warp.config.verbose_warnings = False
