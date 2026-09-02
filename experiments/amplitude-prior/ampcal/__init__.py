"""Does a period-dependent amplitude prior belong in harv's periodogram?

Research experiment. See ``../README.md`` for the design; ``docs/spec.md`` in the harv
repo remains authoritative for harv itself. Nothing here is public API.

Run modules with *this package's parent directory* on ``PYTHONPATH``, in an
environment where ``harv`` is importable::

    EXP=path/to/amplitude-prior
    PYTHONPATH=$EXP uv run python -m ampcal.identity --adapter gaia

Nothing here hardcodes a location, so the directory can be copied into another
repository as-is, provided it is on ``PYTHONPATH``.

**float64 is enabled here, at import, before anything touches JAX.** harv does not
enable it, so JAX defaults to float32 with ``eps ~ 1e-7``. The correctness gate in
:mod:`ampcal.identity` asserts agreement at the ``1e-11`` relative level, which is
simply unreachable in single precision --- and it would fail in a way that looks
exactly like a wiring bug.
"""

__all__ = ()

import jax

jax.config.update("jax_enable_x64", True)
