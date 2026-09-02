"""Load the artifact once and hand every analysis the same table.

:mod:`ampcal.reduce.calibrate` already returns ``list[dict]`` and already owns the
metric definitions. This module runs it, turns it into a frame, and caches it as CSV in
the bundle -- so the table the report cites is literally the table it computed from, and
a second invocation can reuse it.
"""

from __future__ import annotations

__all__ = ("ReportData", "load", "write_provenance")

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ampcal.adapters import get_adapter
from ampcal.artifact import ArtifactReader
from ampcal.reduce.calibrate import reduce_calibrate


def _scalarize(value: Any) -> Any:
    """HDF5 attrs come back as numpy scalars *and* arrays (``merged_from`` is a list of
    shard paths), so neither ``.item()`` nor ``.tolist()`` alone is safe."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, bytes):
        return value.decode()
    return value


def _fingerprint(path: Path) -> dict[str, Any]:
    """Identify an artifact without hashing gigabytes.

    Size plus the sorted simulation list pins the dataset: two runs of the same grid
    produce the same ``sim_id`` set, and a different grid or a partial run does not.
    """
    with ArtifactReader(path) as art:
        sim_ids = art.sim_ids()
        digest = hashlib.sha256("\n".join(sim_ids).encode()).hexdigest()[:16]
        meta = dict(art.meta)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "n_sims": len(sim_ids),
        "sim_id_sha256_16": digest,
        "attrs": {k: _scalarize(v) for k, v in meta.items()},
    }


@dataclass
class ReportData:
    """Everything the analyses read, loaded once."""

    adapter_name: str
    adapter: Any
    signal_path: Path
    meta: dict[str, Any]
    calibrate: pd.DataFrame
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def exponents(self) -> list[float]:
        return sorted(float(e) for e in self.calibrate["exponent"].unique())

    @property
    def levels(self) -> list[float]:
        return sorted(float(m) for m in self.calibrate["level"].unique())

    @property
    def period_ratios(self) -> list[float]:
        return sorted(float(p) for p in self.calibrate["period_ratio"].unique())


def load(
    artifact_dir: Path,
    *,
    adapter_name: str | None = None,
    cache_dir: Path | None = None,
    reuse: bool = False,
) -> ReportData:
    """Load ``signal.h5`` from ``artifact_dir`` and run the reduction."""
    signal = artifact_dir / "signal.h5"
    if not signal.exists():
        raise FileNotFoundError(
            f"{signal} not found; --artifact-dir should be the directory "
            "slurm/run_grid.sh wrote (it contains signal.h5)"
        )

    with ArtifactReader(signal) as art:
        meta = {k: _scalarize(v) for k, v in art.meta.items()}
        stored_adapter = art.adapter_name
    name = adapter_name or stored_adapter
    if adapter_name and adapter_name != stored_adapter:
        raise ValueError(
            f"--adapter {adapter_name!r} but the artifact was written by "
            f"{stored_adapter!r}; the grids are not interchangeable"
        )

    cached = cache_dir / "calibrate.csv" if cache_dir else None
    if reuse and cached is not None and cached.exists():
        calibrate = pd.read_csv(cached)
    else:
        calibrate = pd.DataFrame(reduce_calibrate(signal))
        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            calibrate.to_csv(cached, index=False)

    return ReportData(
        adapter_name=name,
        adapter=get_adapter(name),
        signal_path=signal,
        meta=meta,
        calibrate=calibrate,
        provenance={"adapter": name, "signal": _fingerprint(signal)},
    )


def write_provenance(data: ReportData, path: Path, *, command: str) -> None:
    payload = dict(data.provenance)
    payload["command"] = command
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
