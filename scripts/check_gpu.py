"""GPU check. Run on a cluster GPU node (or anywhere, to see what JAX finds).

    uv run --extra cuda13 python scripts/check_gpu.py [--out results/u17_env.txt]

Passes if JAX lists a GPU and a float32 matmul and a float64 reduction both run on it.
Exit code 0 = pass, 1 = no GPU or an op did not run on the GPU.
"""

from __future__ import annotations

import argparse
import io
import sys
from contextlib import redirect_stdout


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    import pathenhancedfkpinn  # noqa: F401  (enables float64)
    import jax
    import jax.numpy as jnp

    buf = io.StringIO()
    with redirect_stdout(buf):
        jax.print_environment_info()
    devs = jax.devices()
    gpus = [d for d in devs if d.platform == "gpu"]
    lines = [buf.getvalue(), f"devices: {devs}"]
    ok = bool(gpus)
    if gpus:
        g = gpus[0]
        a32 = jax.device_put(jnp.ones((2048, 2048), jnp.float32), g)
        m = (a32 @ a32).block_until_ready()
        a64 = jax.device_put(jnp.arange(10**6, dtype=jnp.float64), g)
        s = jnp.sum(a64).block_until_ready()
        on32, on64 = list(m.devices())[0], list(s.devices())[0]
        lines += [f"float32 matmul on {on32}, dtype {m.dtype}, value {float(m[0, 0])}",
                  f"float64 sum on {on64}, dtype {s.dtype}, value {float(s)} (expect {10**6 * (10**6 - 1) / 2})"]
        ok = on32.platform == "gpu" and on64.platform == "gpu" and s.dtype == jnp.float64
    lines.append("GPU check PASS" if ok else "GPU check FAIL")
    text = "\n".join(lines)
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
