"""Ensure ``ampcal`` is importable and float64 is on before anything touches JAX.

The package's own ``__init__`` enables x64, so importing it first is what matters.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ampcal  # noqa: F401  (import for the x64 side effect)
