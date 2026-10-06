"""Sample the actual candidate state before numerical bounds can hide growth."""
import numpy as np

from .backend import to_cpu
from .physics import A_EARTH


def stability_report(m, ps, fields, ps_tendency, interface_flux,
                     pressure_candidates=()):
    xp, p, o = m.xp, m.cfg.physics, m.ops
    u, v, t, q = fields
    def scalar(x):
        return float(to_cpu(x))
    def per_layer(x):
        return to_cpu(x).tolist()
    report = {"step": m.step_count, "simulated_hours": (m.step_count + 1) * m.dt / 3600}
    inputs = (ps, *fields, ps_tendency, interface_flux, *pressure_candidates)
    report["finite"] = all(bool(to_cpu(xp.isfinite(x).all())) for x in inputs)
    if not report["finite"]:
        report["status"] = "nonfinite"
        return report
    bounds = p.bounds
    limits = {"u": (u, -float(p.umax), float(p.umax)),
              "v": (v, -float(p.umax), float(p.umax)),
              "T": (t, float(bounds.air_temp_min_k), float(bounds.air_temp_max_k)),
              "q": (q, float(bounds.humidity_min), float(bounds.humidity_max)),
              "ps": (ps, float(p.dynamics.min_surface_pressure_pa),
                     float(p.dynamics.max_surface_pressure_pa))}
    fractions = {key: scalar(((x < lo) | (x > hi)).mean())
                 for key, (x, lo, hi) in limits.items()}
    plo, phi = limits["ps"][1:]
    fractions["ps_earlier_stages"] = max(
        [scalar(((x < plo) | (x > phi)).mean()) for x in pressure_candidates] or [0.0])
    report["outside_bounds_fraction"] = fractions
    speed = xp.sqrt(u ** 2 + v ** 2)
    report["max_wind_ms_by_layer"] = per_layer(speed.max(axis=(1, 2)))
    latitude_indices = to_cpu(speed.max(axis=-1).argmax(axis=-1)).astype(int)
    report["max_wind_latitude_deg_by_layer"] = m.lats[latitude_indices].tolist()
    div = o.divergence(u.astype(xp.float32), v.astype(xp.float32))
    polar_weights = o.coslat * (xp.abs(o.lat) >= np.deg2rad(float(m.cfg.numerics.polar_filter_lat)))
    denominator = xp.maximum(polar_weights.sum() * m.nlon, 1e-30)
    report["polar_divergence_rms_s1_by_layer"] = per_layer(
        xp.sqrt((div.astype(xp.float64) ** 2 * polar_weights).sum(axis=(1, 2)) / denominator))
    report["polar_zonal_mean_divergence_rms_s1_by_layer"] = per_layer(
        xp.sqrt((div.mean(axis=-1, keepdims=True).astype(xp.float64) ** 2
                 * polar_weights).sum(axis=(1, 2))
                / xp.maximum(polar_weights.sum(), 1e-30)))
    horizontal = m.dt * (xp.abs(u) * o.invdx + xp.abs(v) * o.invdy)
    layer_pressure_mass = ((m.surface_pressure - float(p.dynamics.top_pressure_pa))[None]
                           * m._sigma_thickness_xp[:, None, None])
    vertical = m.dt * (xp.abs(interface_flux[:-1]) + xp.abs(interface_flux[1:])) / layer_pressure_mass
    report["advective_cfl_max"] = scalar((horizontal + vertical).max())
    # This is the explicit scalar Laplacian diffusion number, not a full
    # gravity-wave or nonlinear stability guarantee.
    max_diff = max(float(p.visc), float(p.diff_T), float(p.diff_q))
    report["diffusion_number_max"] = m.dt * max_diff * scalar((o.invdx ** 2 + o.invdy ** 2).max())
    report["T_range_k"] = [scalar(t.min()), scalar(t.max())]
    report["ps_range_pa"] = [scalar(ps.min()), scalar(ps.max())]
    report["ps_tendency_max_pas"] = scalar(xp.abs(ps_tendency).max())
    report["status"] = ("bounds_exceeded" if any(fractions.values()) else
                        "cfl_exceeded" if report["advective_cfl_max"] > 1
                        or report["diffusion_number_max"] > 0.5 else "no_limit_exceeded")
    report["stabilization_power_wm2"] = {
        key: value["energy_j"] / (4 * np.pi * A_EARTH ** 2 * m.dt)
        for key, value in m.stabilization_budget.items()}
    return report
