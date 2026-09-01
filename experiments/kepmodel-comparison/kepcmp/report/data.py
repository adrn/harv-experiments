"""Load the artifacts once and hand every analysis the same tables.

The five reductions already return ``list[dict]`` and already own the metric
definitions (`kepcmp.reduce.common`). This module runs them, turns them into frames,
and caches them as CSV in the bundle -- so the tables the report cites are literally
the tables it computed from, and a second invocation can reuse them.
"""

from __future__ import annotations

__all__ = ("ReportData", "load")

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from kepcmp.adapters import get_adapter
from kepcmp.artifact import ArtifactReader
from kepcmp.reduce.calibrate import reduce_calibrate
from kepcmp.reduce.decompose import reduce_decompose
from kepcmp.reduce.regime_map import reduce_regime_map
from kepcmp.reduce.roc import reduce_roc


def _scalarize(value: Any) -> Any:
    """HDF5 attrs come back as numpy scalars *and* arrays (``merged_from`` is a list
    of shard paths), so neither ``.item()`` nor ``.tolist()`` alone is safe."""
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
    null_path: Path | None
    meta: dict[str, Any]
    calibrate: pd.DataFrame
    regime: pd.DataFrame
    decompose: pd.DataFrame
    roc: pd.DataFrame | None
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def sigma_amp_mults(self) -> list[float]:
        return sorted(
            float(m)
            for m in self.calibrate["sigma_amp_mult"].dropna().unique()
        )

    @property
    def period_ratios(self) -> list[float]:
        return sorted(float(p) for p in self.calibrate["period_ratio"].unique())

    def delta_rows(self) -> pd.DataFrame:
        """`calibrate` rows for harv's statistic only (drops the ``z0`` reference row)."""
        return self.calibrate[
            self.calibrate["statistic"].str.startswith("delta")
        ].copy()

    def z0_rows(self) -> pd.DataFrame:
        return self.calibrate[self.calibrate["statistic"] == "z0"].copy()


def _reduce_cached(
    name: str,
    fn: Any,
    cache_dir: Path | None,
    reuse: bool,
) -> pd.DataFrame:
    path = cache_dir / f"{name}.csv" if cache_dir else None
    if reuse and path is not None and path.exists():
        return pd.read_csv(path)
    frame = pd.DataFrame(fn())
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    return frame


def load(
    artifact_dir: Path,
    *,
    adapter_name: str | None = None,
    cache_dir: Path | None = None,
    reuse: bool = False,
    fpr: float = 0.01,
) -> ReportData:
    """Load ``signal.h5`` / ``null.h5`` from ``artifact_dir`` and run the reductions."""
    signal = artifact_dir / "signal.h5"
    if not signal.exists():
        raise FileNotFoundError(
            f"{signal} not found; --artifact-dir should be the directory "
            "slurm/run_grid.sh wrote (it contains signal.h5 and null.h5)"
        )
    null = artifact_dir / "null.h5"
    null_path = null if null.exists() else None

    with ArtifactReader(signal) as art:
        meta = {k: _scalarize(v) for k, v in art.meta.items()}
        stored_adapter = art.adapter_name
    name = adapter_name or stored_adapter
    if adapter_name and adapter_name != stored_adapter:
        raise ValueError(
            f"--adapter {adapter_name!r} but the artifact was written by "
            f"{stored_adapter!r}; the grids are not interchangeable"
        )

    calibrate = _reduce_cached(
        "calibrate", lambda: reduce_calibrate(signal), cache_dir, reuse
    )
    regime = _reduce_cached(
        "regime_map",
        lambda: reduce_regime_map(signal, null_path, fpr=fpr),
        cache_dir,
        reuse,
    )
    decompose = _reduce_cached(
        "decompose", lambda: reduce_decompose(signal), cache_dir, reuse
    )
    roc = None
    if null_path is not None:
        roc = _reduce_cached(
            "roc", lambda: reduce_roc(signal, null_path, fpr=fpr), cache_dir, reuse
        )

    provenance = {
        "adapter": name,
        "signal": _fingerprint(signal),
        "null": _fingerprint(null_path) if null_path else None,
        "fpr": fpr,
    }
    return ReportData(
        adapter_name=name,
        adapter=get_adapter(name),
        signal_path=signal,
        null_path=null_path,
        meta=meta,
        calibrate=calibrate,
        regime=regime,
        decompose=decompose,
        roc=roc,
        provenance=provenance,
    )


def write_provenance(data: ReportData, path: Path, *, command: str) -> None:
    payload = dict(data.provenance)
    payload["command"] = command
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
