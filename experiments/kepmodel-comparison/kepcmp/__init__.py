"""harv vs kepmodel periodogram comparison.

Research experiment. See ``../README.md`` for the design; ``docs/spec.md`` in the
repo root remains authoritative for harv itself. Nothing here is public API.

Run modules with *this package's parent directory* on ``PYTHONPATH``, in an
environment where ``harv`` and ``kepmodel`` are importable::

    EXP=path/to/kepmodel-comparison
    PYTHONPATH=$EXP uv run python -m kepcmp.identity --adapter gaia

Nothing here hardcodes a location, so the directory can be copied into another
repository as-is; ``launch_full_grid.sh`` resolves its own path. See "Porting this
code" in ``../README.md`` for what the host environment must provide.

**float64 is enabled here, at import, before anything touches JAX.** harv does not
enable it, so JAX defaults to float32 with ``eps ~ 1e-7``. The identity gate in
``../README.md`` asserts agreement to ``1e-8`` in nats, which is simply unreachable
in single precision --- and it would fail in a way that looks exactly like a wiring
bug. kepmodel/spleaf are numpy and already double precision, so without this the two
sides are not even computing in the same precision.
"""

__all__ = ()

import jax

jax.config.update("jax_enable_x64", True)
