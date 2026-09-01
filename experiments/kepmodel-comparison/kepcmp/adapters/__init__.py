"""Data-type adapters.

The adapter is the single seam that keeps the harness data-type agnostic: everything
above this layer --- ``identity``, ``run``, ``artifact``, ``reduce/*`` --- talks only
to the protocol in ``base.py``, and the plumbing both adapters share lives in
``common.py``.
"""

__all__ = ("ADAPTERS", "Adapter", "GaiaAdapter", "RVAdapter", "get_adapter")

from kepcmp.adapters.base import Adapter
from kepcmp.adapters.gaia import GaiaAdapter
from kepcmp.adapters.rv import RVAdapter

ADAPTERS: dict[str, type] = {"rv": RVAdapter, "gaia": GaiaAdapter}


def get_adapter(name: str) -> Adapter:
    """Instantiate an adapter by name (``"rv"`` or ``"gaia"``)."""
    try:
        cls = ADAPTERS[name]
    except KeyError:
        raise ValueError(
            f"unknown adapter {name!r}; choose from {sorted(ADAPTERS)}"
        ) from None
    return cls()  # type: ignore[return-value]
