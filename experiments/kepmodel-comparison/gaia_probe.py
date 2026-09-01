"""Feasibility probe: does the kepcmp seam actually reach Gaia astrometry?

Not a gate, not a test. Answers three yes/no questions:
  1. does harv's periodogram run on GaiaAstrometryData?
  2. does kepmodel's AstroModel._chi2ogram run with harv's own 5 base columns?
  3. do the two peak at the same period?
"""
import harv.periodogram as hp
import kepcmp  # noqa: F401  (float64)
import numpy as np
from harv.models.astrometry import GaiaAstrometryModel
from harv.models.parameterizations import FourierGaiaAstrometry
from harv.samplers._prior_resolution import effective_linear_prior_from_prior
from harv.simulate import simulate_gaia_epoch_astrometry
from unxt import Q, ustrip

T_SPAN = Q(5.0, "yr")
data, truth = simulate_gaia_epoch_astrometry(
    seed=42, n_obs=60, baseline=T_SPAN, period=Q(500.0, "day"),
    eccentricity=0.3, semi_major_axis=Q(1.0, "mas"), parallax=Q(10.0, "mas"),
    al_error=Q(0.1, "mas"),
)
print("data fields:", type(data).__name__, {k: getattr(data, k).shape for k in
      ("time", "al_position", "al_position_err", "scan_angle", "parallax_factor")})
print("truth period:", truth["period"])

freq = hp.frequency_grid(data, period_min=Q(20.0, "day"), period_max=Q(4.0, "yr"),
                         samples_per_peak=10)
print("n_freq:", freq.shape)

# --- 1. harv marginal periodogram --------------------------------------------
prior = FourierGaiaAstrometry(n_terms=1).default_prior(
    period_min=Q(20.0, "day"), period_max=Q(4.0, "yr"),
    sigma_amp=Q(5.0, "mas"), sigma_pos=Q(100.0, "mas"),
    sigma_pm=Q(100.0, "mas/yr"), sigma_parallax=Q(100.0, "mas"),
)
res = hp.periodogram(data, freq, prior=prior, n_terms=1)
delta = np.asarray(res.delta_ln_likelihood)
p_delta = float(ustrip("day", 1 / freq[np.nanargmax(delta)]))
print(f"harv Delta: peak {p_delta:.2f} d, range {np.ptp(delta):.1f} nats")

# --- 2. kepmodel profile periodogram, harv's own base columns ----------------
from kepmodel.astro import AstroModel
from spleaf.term import Error

base_model = GaiaAstrometryModel(parameterization=FourierGaiaAstrometry(n_terms=0))
base_prior = FourierGaiaAstrometry(n_terms=0).default_prior(
    period_min=Q(20.0, "day"), period_max=Q(4.0, "yr"),
    sigma_pos=Q(100.0, "mas"), sigma_pm=Q(100.0, "mas/yr"),
    sigma_parallax=Q(100.0, "mas"),
)
lp = effective_linear_prior_from_prior(base_prior, base_model) or {}
lp = {n: lp[n] for n in base_model._all_linear_names()}
nl = {"period": Q(500.0, "day"), "eccentricity": 0.0}
X_base = np.asarray(base_model._full_design_matrix(nl, data), dtype=float)
print("base columns:", X_base.shape, base_model._all_linear_names())

t = np.asarray(ustrip("day", data.time - data.t_ref), dtype=float)
s = np.asarray(ustrip("mas", data.al_position), dtype=float)
err = np.asarray(ustrip("mas", data.al_position_err), dtype=float)
psi = np.asarray(ustrip("rad", data.scan_angle), dtype=float)

km = AstroModel(t, s, np.cos(psi), np.sin(psi), err=Error(err))
for i, name in enumerate(base_model._all_linear_names()):
    km.add_lin(X_base[:, i], name=name)

f = np.asarray(ustrip("1/day", freq), dtype=float)
nu0, dnu = 2 * np.pi * f[0], 2 * np.pi * float(np.diff(f)[0])
chi2_h, chi2_k = km._chi2ogram(nu0, dnu, f.size)
_, power = km.periodogram(nu0, dnu, f.size)
print("power == 1 - chi2K/chi2H:",
      bool(np.allclose(power, 1 - chi2_k / chi2_h, rtol=1e-10, atol=1e-12)))
z0 = 0.5 * (chi2_h - chi2_k)
p_z0 = float(ustrip("day", 1 / freq[np.nanargmax(z0)]))
print(f"kepmodel z0: peak {p_z0:.2f} d, range {np.ptp(z0):.1f} nats, chi2_H {chi2_h:.3e}")

# --- 3. do they agree? -------------------------------------------------------
print(f"\ntruth {float(ustrip('day', truth['period'])):.2f} d | "
      f"Delta {p_delta:.2f} d | z0 {p_z0:.2f} d")
print("Occam offset (z0 - Delta) range:", f"{np.ptp(z0 - delta):.4f} nats",
      "median", f"{np.median(z0 - delta):.2f}")
