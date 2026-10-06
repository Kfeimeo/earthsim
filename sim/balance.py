"""Discrete balanced initial conditions and conservative polar stabilization."""
import numpy as np

from .backend import to_cpu
from .physics import A_EARTH, CP, LV, RD, qsat


def initialize_balanced(model):
    """Specify T(phi,p), integrate hydrostatically, then solve gradient wind.

    A flat lower boundary and constant ps make the axisymmetric pressure
    surfaces and sigma surfaces coincide. Real terrain cannot be combined
    with this zonal, v=0 equilibrium without a separate balance adjustment.
    """
    m = model
    xp, p, o = m.xp, m.cfg.physics, m.ops
    ic, dyn = p.initial_conditions, p.dynamics
    m.surface_height = xp.zeros_like(m.elev)
    m.surface_geopotential = xp.zeros_like(m.elev)
    m.terrain_slope_x = xp.zeros_like(m.elev)
    m.terrain_slope_y = xp.zeros_like(m.elev)
    m.terrain_slope = xp.zeros_like(m.elev)
    m.surface_pressure = xp.full_like(
        m.elev, float(dyn.reference_surface_pressure_pa))
    if not (float(dyn.top_pressure_pa) < float(dyn.min_surface_pressure_pa)
            <= float(dyn.reference_surface_pressure_pa)
            <= float(dyn.max_surface_pressure_pa)):
        raise ValueError("invalid top/reference/surface pressure bounds")
    pressure = (float(dyn.top_pressure_pa)
                + m._sigma_centres_xp[:, None, None]
                * (m.surface_pressure - float(dyn.top_pressure_pa)))
    height = -float(p.vertical.scale_height) * xp.log(
        pressure / float(dyn.reference_surface_pressure_pa))
    # Disabling winds selects a resting, horizontally uniform atmosphere.
    contrast = (float(ic.equilibrium_temp_pole_delta_k)
                if p.ideal_wind_enabled else 0.0)
    base = (float(ic.equilibrium_temp_base_k)
            - contrast * xp.sin(o.lat) ** 2)
    m.T_layers = xp.clip(m._equilibrium_temperature_profile(base, height),
                         float(p.bounds.air_temp_min_k),
                         float(p.bounds.air_temp_max_k)).astype(xp.float32)
    rh = (float(p.init_upper_rh)
          + (float(p.init_surface_rh) - float(p.init_upper_rh))
          * xp.exp(-height / float(ic.humidity_decay_height_m)))
    m.q_layers = (xp.clip(rh * qsat(xp, m.T_layers, pressure),
                          float(p.bounds.humidity_min),
                          float(p.bounds.humidity_max)) if p.moisture
                  else xp.zeros_like(m.T_layers)).astype(xp.float32)
    m._refresh_primitive_diagnostics()
    _, gy = m._primitive_pressure_gradient()
    # Rationalized root connected continuously to the geostrophic branch.
    # tan(phi)/a * u^2 + f*u + gy = 0. Symmetry gives u=0 at
    # an exact equatorial grid point, where both f and gy vanish.
    f, b = o.f.astype(xp.float64), o.tanl.astype(xp.float64)
    c = gy.astype(xp.float64)
    disc = f * f - 4.0 * b * c
    if bool(to_cpu(xp.any(disc < -1.0e-18))):
        raise ValueError("temperature profile has no real gradient-wind root")
    denominator = f + xp.where(f >= 0, 1.0, -1.0) * xp.sqrt(
        xp.maximum(disc, 0.0))
    equator = xp.abs(f) < 1.0e-12
    u = -2.0 * c / xp.where(equator, 1.0, denominator)
    m.u_layers = xp.where(equator, 0.0, u).astype(xp.float32)
    if float(to_cpu(xp.max(xp.abs(m.u_layers)))) > float(p.umax):
        raise ValueError("balanced wind exceeds physics.umax; adjust T or umax")
    m.v_layers = xp.zeros_like(m.u_layers)
    m.w_layers = xp.zeros_like(m.u_layers)
    m.omega_layers = xp.zeros_like(m.u_layers)
    m.sigma_dot_interfaces = xp.zeros(
        (m.nz + 1, m.nlat, m.nlon), dtype=xp.float32)
    amplitude = float(ic.balanced_perturbation_k)
    if not np.isfinite(amplitude) or abs(amplitude) > 0.01:
        raise ValueError("balanced_perturbation_k must be finite and <= 0.01 K")
    m.T_layers += (amplitude * xp.cos(o.lat) ** 2
                   * xp.sin(3.0 * o.lon_rad)
                   * xp.exp(-height / 7000.0)).astype(xp.float32)
    m._refresh_primitive_diagnostics()
    m._sync_surface_views()
    print("[startup] balanced ideal atmosphere: flat dynamical terrain, "
          "constant ps, hydrostatic gradient wind", flush=True)


def balance_report(m):
    """Evaluate the exact discrete inviscid meridional momentum residual."""
    from .primitive import sigma_mass_flux
    xp, o = m.xp, m.ops
    _, gy = m._primitive_pressure_gradient()
    residual = -gy - o.f * m.u_layers - o.tanl * m.u_layers ** 2
    rv = np.asarray(to_cpu(residual))
    ps_t, _, _ = sigma_mass_flux(
        xp, o, m.u_layers, m.v_layers, m.surface_pressure,
        m._sigma_thickness_xp, float(m.cfg.physics.dynamics.top_pressure_pa))
    diffusivity = float(m.cfg.physics.dynamics.surface_pressure_diffusivity)
    if diffusivity:
        ps_t += diffusivity * o.lap(m.surface_pressure)
    pt = np.asarray(to_cpu(ps_t))
    max_each = np.max(np.abs(rv), axis=(1, 2))
    lat_indices = np.argmax(np.max(np.abs(rv), axis=-1), axis=-1)
    zonal = rv.mean(axis=-1)
    report = dict(max_residual_ms2=max_each.tolist(),
                  latitude_deg=m.lats[lat_indices].tolist(),
                  zonal_mean_residual_ms2=zonal.tolist(),
                  ps_tendency_max_pas=float(np.max(np.abs(pt))),
                  ps_tendency_mean_pas=float(np.average(
                      pt.mean(axis=-1), weights=np.cos(np.deg2rad(m.lats)))))
    tolerance = float(m.cfg.physics.initial_conditions.balance_tolerance_ms2)
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("balance_tolerance_ms2 must be finite and positive")
    report["passed"] = bool(np.isfinite(rv).all() and np.isfinite(pt).all()
                            and np.max(max_each) <= tolerance)
    for k in range(m.nz):
        print(f"[balance] layer={k} max|Rv|={max_each[k]:.6e} m/s^2 "
              f"lat={report['latitude_deg'][k]:.4f} deg "
              f"max|zonal_mean Rv|={np.max(np.abs(zonal[k])):.6e}", flush=True)
    print(f"[balance] initial dynamical ps tendency: "
          f"max={report['ps_tendency_max_pas']:.6e}, "
          f"area_mean={report['ps_tendency_mean_pas']:.6e} Pa/s; "
          f"passed={report['passed']}", flush=True)
    return report


def polar_divergence_damping(m, v, dt, surface_pressure=None):
    """Negative adjoint of zonal-mean spherical divergence (energy dissipative).

    Use forward differences of cos(phi)*v at meridional faces, including
    the polar boundary faces. This also damps the alternating-grid mode.
    The adjoint uses the instantaneous layer mass and true spherical area.
    """
    xp, o = m.xp, m.ops
    coefficient = float(m.cfg.numerics.polar_divergence_damping_m2s)
    if not np.isfinite(coefficient) or coefficient < 0:
        raise ValueError("polar_divergence_damping_m2s must be non-negative")
    if coefficient == 0:
        return v
    ps = m.surface_pressure if surface_pressure is None else surface_pressure
    mass = (ps
            - float(m.cfg.physics.dynamics.top_pressure_pa))[None, :, :]
    mean_mass = mass.mean(axis=-1, keepdims=True)
    vm = (v * mass).mean(axis=-1, keepdims=True) / mean_mass
    cos = o.coslat
    # Closed polar flux, evaluated as differences of latitude-centred v.
    weighted = vm * cos
    zero = xp.zeros_like(weighted[..., :1, :])
    difference = xp.diff(xp.concatenate([zero, weighted, zero], axis=-2), axis=-2)
    face_cos = xp.concatenate([cos[:1], 0.5 * (cos[:-1] + cos[1:]), cos[-1:]])
    face_w = xp.concatenate([o.pf_w[:1], 0.5 * (o.pf_w[:-1] + o.pf_w[1:]), o.pf_w[-1:]])
    face_mass = xp.concatenate([mean_mass[:, :1],
                                0.5 * (mean_mass[:, :-1] + mean_mass[:, 1:]),
                                mean_mass[:, -1:]], axis=-2)
    # Limit explicit damping strength using a conservative matrix row bound.
    rate_bound = 4.0 * coefficient * float(o.invdy) ** 2 * float(
        to_cpu(xp.max(face_mass) / xp.min(mean_mass)))
    effective = min(coefficient, coefficient * 0.5 / max(dt * rate_bound, 0.5))
    flux = face_w * face_mass * difference / face_cos
    tendency = effective * xp.diff(flux, axis=-2) * o.invdy ** 2 / mean_mass
    return v + dt * tendency


def filter_increments(m, ps, fields, old_ps, old_fields):
    """Jointly filter mass and mass-weighted increments, preserving zonal sums.

    Filtering increments leaves the balanced background intact. No field is
    smoothed independently of ps; all use the same zonal linear operator.
    """
    xp, o = m.xp, m.ops
    top = float(m.cfg.physics.dynamics.top_pressure_pa)
    mu, mu0 = ps - top, old_ps - top
    mu_filtered = mu0 + o.polar_filter(mu - mu0)
    if bool(to_cpu(xp.any(mu_filtered <= 0))):
        raise ValueError("polar mass filtering produced a non-positive column")
    out = []
    for field, old in zip(fields, old_fields):
        conserved0 = mu0[None] * old
        increment = mu[None] * field - conserved0
        out.append((conserved0 + o.polar_filter(increment)) / mu_filtered[None])
    return mu_filtered + top, out


def budgets(m, ps, fields):
    """Global mass, relative momentum and discrete hydrostatic total energy.

    Integrate Phi exactly in pressure within each piecewise-isothermal layer:
    integral(Phi dp) = mu*Phi_s + Rd*integral(Tv dp)
                       - p_top*(Phi_top-Phi_s).
    This includes the finite-pressure model-top correction. Internal energy
    uses the model's constant dry Cv; latent energy is Lv*q.
    """
    xp = m.xp
    u, v, t, q = fields
    area = (2 * np.pi * A_EARTH ** 2 / m.nlon
            * (2 * np.sin(np.pi / (2 * m.nlat))) * m.ops.coslat)
    mass = ((ps.astype(xp.float64)
             - float(m.cfg.physics.dynamics.top_pressure_pa))[None]
            * m._sigma_thickness_xp[:, None, None]
            * area / float(m.cfg.physics.g_eff))
    td, qd = t.astype(xp.float64), q.astype(xp.float64)
    tv = td * (1.0 + 0.608 * qd)
    energy = ((CP - RD) * td + RD * tv + LV * qd
              + 0.5 * (u.astype(xp.float64) ** 2 + v.astype(xp.float64) ** 2)
              + m.surface_geopotential)
    top = float(m.cfg.physics.dynamics.top_pressure_pa)
    mu = ps.astype(xp.float64) - top
    phi_depth = xp.zeros_like(mu)
    for k in range(m.nz):
        bottom_p = top + float(m.sigma_interfaces[k]) * mu
        upper_p = top + float(m.sigma_interfaces[k + 1]) * mu
        phi_depth += RD * tv[k] * xp.log(bottom_p / upper_p)
    top_correction = (top * phi_depth * area / float(m.cfg.physics.g_eff)).sum()
    return dict(mass_kg=float(to_cpu(mass.sum())),
                axial_momentum_kgm2s=float(to_cpu(
                    (mass * u * A_EARTH * m.ops.coslat).sum())),
                meridional_momentum_kgms=float(to_cpu((mass * v).sum())),
                energy_j=float(to_cpu((mass * energy).sum() - top_correction)))


def budget_changes(m, ps0, fields0, ps1, fields1):
    """Sum local changes, avoiding subtraction of two huge global totals.

    Product differences are expanded before integration. The kinetic energy
    difference uses (v1-v0)*(v1+v0)/2; the top-pressure correction uses log1p
    for small pressure changes. Identical states therefore give exact zeros.
    """
    xp = m.xp
    top, g = float(m.cfg.physics.dynamics.top_pressure_pa), float(m.cfg.physics.g_eff)
    area = (4 * np.pi * A_EARTH ** 2 / m.nlon
            * np.sin(np.pi / (2 * m.nlat)) * m.ops.coslat.astype(xp.float64))
    mu0 = ps0.astype(xp.float64) - top
    dmu = ps1.astype(xp.float64) - ps0.astype(xp.float64)
    ds = m._sigma_thickness_xp.astype(xp.float64)[:, None, None]
    weight = ds * area / g
    if ps0 is ps1 and all(fields0[k] is fields1[k] for k in (0, 2, 3)):
        # Divergence damping changes only v. Avoid allocating full double
        # precision temperature/humidity copies just to subtract zeros.
        v0, v1 = fields0[1].astype(xp.float64), fields1[1].astype(xp.float64)
        dv = v1 - v0
        kinetic = float(to_cpu((weight * mu0 * (0.5 * dv * (v1 + v0))).sum()))
        return {"mass_kg": 0.0, "axial_momentum_kgm2s": 0.0,
                "meridional_momentum_kgms": float(to_cpu((weight * mu0 * dv).sum())),
                "kinetic_energy_j": kinetic, "energy_j": kinetic}
    a, b = [x.astype(xp.float64) for x in fields0], [x.astype(xp.float64) for x in fields1]
    u0, v0, t0, q0 = a
    u1, v1, t1, q1 = b
    du, dv, dt, dq = [y - x for x, y in zip(a, b)]
    dke = 0.5 * (du * (u1 + u0) + dv * (v1 + v0))
    ke1 = 0.5 * (u1 ** 2 + v1 ** 2)
    dtv = dt + 0.608 * (t0 * dq + q1 * dt)
    tv1 = t1 * (1 + 0.608 * q1)
    de = (CP - RD) * dt + RD * dtv + LV * dq + dke
    e1 = ((CP - RD) * t1 + RD * tv1 + LV * q1
          + ke1 + m.surface_geopotential)
    dphi = xp.zeros_like(mu0)
    for k in range(m.nz):
        sb, st = float(m.sigma_interfaces[k]), float(m.sigma_interfaces[k + 1])
        pb0, pt0 = top + sb * mu0, top + st * mu0
        dlog = xp.log1p(sb * dmu / pb0) - xp.log1p(st * dmu / pt0)
        dphi += RD * (dtv[k] * xp.log(pb0 / pt0) + tv1[k] * dlog)
    def total(x):
        return float(to_cpu(x.sum(dtype=xp.float64)))
    return {
        "mass_kg": total(weight * dmu),
        "axial_momentum_kgm2s": total(
            weight * (mu0 * du + dmu * u1) * A_EARTH * m.ops.coslat),
        "meridional_momentum_kgms": total(weight * (mu0 * dv + dmu * v1)),
        "kinetic_energy_j": total(weight * (mu0 * dke + dmu * ke1)),
        "energy_j": total(weight * (mu0 * de + dmu * e1))
                    - total(top * dphi * area / g),
    }
