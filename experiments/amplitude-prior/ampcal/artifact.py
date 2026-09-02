"""HDF5 artifact: the one shared table every reduction reads.

Layout (see ``../README.md``, "The artifact")::

    /                       attrs: harv version, created_utc, samples_per_peak,
                                   period_min/max, t_span, sigma_n, exponents
    /frequency              (n_freq,) shared grid, cycles / day
    /sims/<sim_id>/         attrs: cell axes (population included), seed, injected
                                   truth, data_rms, integrated_snr
             data/          the adapter's per-epoch arrays (RV: time, rv, rv_err;
                            Gaia: time, al_position, al_position_err, scan_angle,
                            parallax_factor) --- kept so the report's nuisance sweep
                            can replay a dataset without re-simulating
             arms/<name>/   delta, occam, shrinkage, cond (n_freq,)
                            attrs: n_terms_requested, n_terms_effective, exponent,
                                   level, sigma_amp, ln_z_base
             reference/     R (n_freq,);  attrs: n_mc, seed

Nothing here imports an adapter, so reductions run in a plain dev environment.
"""

from __future__ import annotations

__all__ = ("ArmRecord", "ArtifactReader", "ArtifactWriter", "SimRecord")

import datetime as dt
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Self

import h5py
import numpy as np
from numpy.typing import NDArray


def _pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "absent"


@dataclass
class ArmRecord:
    """One arm's periodogram plus its Occam / shrinkage decomposition.

    ``name`` is carried explicitly rather than derived, because it is the artifact's
    group key and the join key every reduction uses; deriving it in two places is how
    a rename silently orphans half a grid.
    """

    name: str
    n_terms_requested: int
    n_terms_effective: int
    exponent: float
    level: float
    sigma_amp: float
    """The arm's scale at ``P = P0``, in the adapter's observation unit."""

    ln_z_base: float
    delta: NDArray[np.float64]
    occam: NDArray[np.float64]
    shrinkage: NDArray[np.float64]
    cond: NDArray[np.float64]


@dataclass
class SimRecord:
    """Everything computed for one simulated dataset."""

    sim_id: str
    cell_id: str
    seed: int
    population: str
    period_ratio: float
    snr: float
    eccentricity: float
    n_obs: int
    integrated_snr: float
    data_rms: float
    truth: dict[str, float]

    data: dict[str, NDArray[np.float64]]
    """Per-epoch arrays from ``adapter.data_arrays``; the names differ by data type,
    which is why this is a dict rather than named fields."""

    arms: list[ArmRecord] = field(default_factory=list)
    reference: NDArray[np.float64] | None = None
    reference_n_mc: int = 0
    reference_seed: int = 0


class ArtifactWriter:
    """Incremental writer. Each simulation is flushed as it completes, so a long
    run that dies partway through still leaves a readable artifact.
    """

    def __init__(self, path: str | Path, *, frequency: NDArray[np.float64], **meta: Any):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._h5 = h5py.File(self.path, "w")
        self._h5.create_dataset("frequency", data=np.asarray(frequency, dtype=float))
        self._h5.create_group("sims")
        self._h5.attrs["created_utc"] = dt.datetime.now(dt.UTC).isoformat()
        self._h5.attrs["harv_version"] = _pkg_version("harv")
        for k, v in meta.items():
            self._h5.attrs[k] = v

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._h5.close()

    def write(self, rec: SimRecord) -> None:
        g = self._h5["sims"].create_group(rec.sim_id)
        for name in (
            "cell_id", "seed", "population", "period_ratio", "snr", "eccentricity",
            "n_obs", "integrated_snr", "data_rms",
        ):
            g.attrs[name] = getattr(rec, name)
        for k, v in rec.truth.items():
            g.attrs[f"truth_{k}"] = v

        d = g.create_group("data")
        for name, arr in rec.data.items():
            d.create_dataset(name, data=arr)

        arms = g.create_group("arms")
        for arm in rec.arms:
            ag = arms.create_group(arm.name)
            for field_name in ("delta", "occam", "shrinkage", "cond"):
                ag.create_dataset(field_name, data=getattr(arm, field_name))
            for field_name in (
                "n_terms_requested", "n_terms_effective", "exponent", "level",
                "sigma_amp", "ln_z_base",
            ):
                ag.attrs[field_name] = getattr(arm, field_name)

        if rec.reference is not None:
            rg = g.create_group("reference")
            rg.create_dataset("R", data=rec.reference)
            rg.attrs["n_mc"] = rec.reference_n_mc
            rg.attrs["seed"] = rec.reference_seed
        self._h5.flush()


class ArtifactReader:
    """Read-only accessor used by the reductions."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._h5 = h5py.File(self.path, "r")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self._h5.close()

    def close(self) -> None:
        self._h5.close()

    @property
    def meta(self) -> dict[str, Any]:
        return dict(self._h5.attrs)

    @property
    def frequency(self) -> NDArray[np.float64]:
        return np.asarray(self._h5["frequency"])

    @property
    def period(self) -> NDArray[np.float64]:
        return 1.0 / self.frequency

    def sim_ids(self) -> list[str]:
        return sorted(self._h5["sims"].keys())

    def attrs(self, sim_id: str) -> dict[str, Any]:
        out = dict(self._h5[f"sims/{sim_id}"].attrs)
        # h5py hands strings back as bytes when written from numpy scalars; the
        # reductions group on `population`, so a b"physical" key would split the axis
        # in two without ever raising.
        for k, v in out.items():
            if isinstance(v, bytes):
                out[k] = v.decode()
        return out

    def arm_names(self, sim_id: str) -> list[str]:
        return sorted(self._h5[f"sims/{sim_id}/arms"].keys())

    def arm(self, sim_id: str, name: str) -> dict[str, Any]:
        g = self._h5[f"sims/{sim_id}/arms/{name}"]
        out: dict[str, Any] = dict(g.attrs)
        for field_name in ("delta", "occam", "shrinkage", "cond"):
            out[field_name] = np.asarray(g[field_name])
        return out

    def reference(self, sim_id: str) -> NDArray[np.float64] | None:
        g = self._h5[f"sims/{sim_id}"]
        if "reference" not in g:
            return None
        return np.asarray(g["reference/R"])

    def has_reference(self, sim_id: str) -> bool:
        return "reference" in self._h5[f"sims/{sim_id}"]

    def data(self, sim_id: str) -> dict[str, NDArray[np.float64]]:
        """Every stored per-epoch array, keyed as the adapter wrote it."""
        g = self._h5[f"sims/{sim_id}/data"]
        return {k: np.asarray(g[k]) for k in g}

    @property
    def adapter_name(self) -> str:
        """Which adapter produced this artifact."""
        return str(self._h5.attrs["adapter"])
