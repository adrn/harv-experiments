"""HDF5 artifact: the one shared table all four reductions read.

Layout (see ``../README.md``, "Component 4")::

    /                       attrs: harv/kepmodel/spleaf versions, created_utc,
                                   samples_per_peak, period_min/max, t_span, sigma_n
    /frequency              (n_freq,) shared grid, cycles / day
    /sims/<sim_id>/         attrs: cell axes, seed, injected truth, data_rms,
                                   integrated_snr, is_null
             data/          the adapter's per-epoch arrays (RV: time, rv, rv_err;
                            Gaia: time, al_position, al_position_err, scan_angle,
                            parallax_factor) --- kept so `prior_quality` can run the
                            sampler on the identical dataset without re-simulating
             kepmodel/      z0 (n_freq,);  attrs: chi2_base
             harv/H<h>_s<i>/ delta, occam, shrinkage, cond (n_freq,)
                            attrs: n_terms_requested, n_terms_effective, sigma_amp,
                                   sigma_amp_mult, ln_z_base
             reference/     R (n_freq,);  attrs: n_mc, seed

Nothing here imports kepmodel, so reductions can run in a plain dev environment.
"""

from __future__ import annotations

__all__ = ("ArtifactReader", "ArtifactWriter", "SimRecord", "StatRecord")

import datetime as dt
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray


def _pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "absent"


@dataclass
class StatRecord:
    """One harv configuration's periodogram plus its decomposition terms."""

    n_terms_requested: int
    n_terms_effective: int
    sigma_amp: float
    sigma_amp_mult: float
    ln_z_base: float
    delta: NDArray[np.float64]
    occam: NDArray[np.float64]
    shrinkage: NDArray[np.float64]
    cond: NDArray[np.float64]

    @property
    def key(self) -> str:
        return f"H{self.n_terms_requested}_s{self.sigma_amp_mult:.4g}"


@dataclass
class SimRecord:
    """Everything computed for one simulated dataset."""

    sim_id: str
    cell_id: str
    seed: int
    period_ratio: float
    snr: float
    eccentricity: float
    n_obs: int
    integrated_snr: float
    is_null: bool
    data_rms: float
    truth: dict[str, float]

    data: dict[str, NDArray[np.float64]]
    """Per-epoch arrays from ``adapter.data_arrays``; the names differ by data type,
    which is why this is a dict rather than named fields."""

    z0: NDArray[np.float64]
    chi2_base: float

    z0_degenerate: bool = False
    """``z0`` carries no period information here (``n_obs <= 3``) or was uncomputable."""

    z0_reason: str = ""
    """Why ``z0`` is degenerate, empty when it is fine. Recorded so the regime map can
    say *which* failure occurred rather than just omitting the cell."""

    stats: list[StatRecord] = field(default_factory=list)
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
        self._h5.attrs["kepmodel_version"] = _pkg_version("kepmodel")
        self._h5.attrs["spleaf_version"] = _pkg_version("spleaf")
        for k, v in meta.items():
            self._h5.attrs[k] = v

    def __enter__(self) -> ArtifactWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._h5.close()

    def write(self, rec: SimRecord) -> None:
        g = self._h5["sims"].create_group(rec.sim_id)
        for name in (
            "cell_id", "seed", "period_ratio", "snr", "eccentricity", "n_obs",
            "integrated_snr", "is_null", "data_rms",
        ):
            g.attrs[name] = getattr(rec, name)
        for k, v in rec.truth.items():
            g.attrs[f"truth_{k}"] = v

        d = g.create_group("data")
        for name, arr in rec.data.items():
            d.create_dataset(name, data=arr)

        km = g.create_group("kepmodel")
        km.create_dataset("z0", data=rec.z0)
        km.attrs["chi2_base"] = rec.chi2_base
        km.attrs["z0_degenerate"] = rec.z0_degenerate
        km.attrs["z0_reason"] = rec.z0_reason

        hv = g.create_group("harv")
        for st in rec.stats:
            sg = hv.create_group(st.key)
            sg.create_dataset("delta", data=st.delta)
            sg.create_dataset("occam", data=st.occam)
            sg.create_dataset("shrinkage", data=st.shrinkage)
            sg.create_dataset("cond", data=st.cond)
            sg.attrs["n_terms_requested"] = st.n_terms_requested
            sg.attrs["n_terms_effective"] = st.n_terms_effective
            sg.attrs["sigma_amp"] = st.sigma_amp
            sg.attrs["sigma_amp_mult"] = st.sigma_amp_mult
            sg.attrs["ln_z_base"] = st.ln_z_base

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

    def __enter__(self) -> ArtifactReader:
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
        return dict(self._h5[f"sims/{sim_id}"].attrs)

    def z0(self, sim_id: str) -> NDArray[np.float64]:
        return np.asarray(self._h5[f"sims/{sim_id}/kepmodel/z0"])

    def kepmodel_attrs(self, sim_id: str) -> dict[str, Any]:
        """``chi2_base`` plus the ``z0_degenerate`` / ``z0_nonphysical`` flags."""
        return dict(self._h5[f"sims/{sim_id}/kepmodel"].attrs)

    def z0_usable(self, sim_id: str) -> bool:
        """Whether ``z0`` is a measurement rather than a singular-matrix artifact.

        Reductions must gate on this: at ``n_obs <= 3`` kepmodel's ``z0`` is flat to
        numerical noise (or outright non-physical), and scoring it against truth as if
        it were a real periodogram would silently report nonsense.
        """
        a = self.kepmodel_attrs(sim_id)
        return not bool(a.get("z0_degenerate", False))

    def stat_keys(self, sim_id: str) -> list[str]:
        return sorted(self._h5[f"sims/{sim_id}/harv"].keys())

    def stat(self, sim_id: str, key: str) -> dict[str, Any]:
        g = self._h5[f"sims/{sim_id}/harv/{key}"]
        out: dict[str, Any] = dict(g.attrs)
        for name in ("delta", "occam", "shrinkage", "cond"):
            out[name] = np.asarray(g[name])
        return out

    def reference(self, sim_id: str) -> NDArray[np.float64] | None:
        g = self._h5[f"sims/{sim_id}"]
        if "reference" not in g:
            return None
        return np.asarray(g["reference/R"])

    def data(self, sim_id: str) -> dict[str, NDArray[np.float64]]:
        """Every stored per-epoch array, keyed as the adapter wrote it."""
        g = self._h5[f"sims/{sim_id}/data"]
        return {k: np.asarray(g[k]) for k in g}

    @property
    def adapter_name(self) -> str:
        """Which adapter produced this artifact. Defaults to ``rv`` for artifacts
        written before the harness became multi-adapter."""
        return str(self._h5.attrs.get("adapter", "rv"))

    def has_reference(self, sim_id: str) -> bool:
        return "reference" in self._h5[f"sims/{sim_id}"]
