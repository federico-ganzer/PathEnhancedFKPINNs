import os

# Keep CPU tests deterministic and quiet; the package turns on float64 on import.
os.environ.setdefault("JAX_PLATFORMS", "cpu" if not os.environ.get("PEFK_TEST_GPU") else "")

import pathenhancedfkpinn  # noqa: E402,F401
