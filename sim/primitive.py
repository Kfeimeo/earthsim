"""Hydrostatic primitive-equation utilities in a hybrid sigma-pressure grid.

Arrays are ordered from the lowest atmospheric layer to the highest.  Interface
pressure is ``p = A + B * ps`` (CAM ``hyai``/``hybi`` convention, with ``A`` in
Pa).  ``B`` is one at the ground, so the lowest interfaces follow terrain, and
zero at and above the pure-pressure transition, so the model top is the fixed
pressure ``A[-1]``.  A pure sigma grid is the special case
``A = p_top * (1 - sigma)``, ``B = sigma``.

The prognostic mass of a layer is its pseudo-density ``dp = dA + dB * ps``.
Zero vertical mass flux at both boundaries is the exact impermeability
condition, including over terrain.
"""
import numpy as np

from .physics import RD, CP, KAPPA, P0

# U.S. Standard Atmosphere 1976 layers: base height (m), base temperature (K),
# lapse rate (K/m, positive means warming with height) and base pressure (Pa).
_STANDARD_ATMOSPHERE = (
    (0.0, 288.15, -0.0065, 101325.0),
    (11000.0, 216.65, 0.0, 22632.06),
    (20000.0, 216.65, 0.001, 5474.889),
    (32000.0, 228.65, 0.0028, 868.0187),
    (47000.0, 270.65, 0.0, 110.9063),
    (51000.0, 270.65, -0.0028, 66.93887),
)
_G0 = 9.80665
_R_STANDARD = 287.053


def standard_atmosphere_height(pressure_pa):
    """Geometric height of a pressure in the U.S. Standard Atmosphere 1976."""
    p = np.atleast_1d(np.asarray(pressure_pa, dtype=np.float64))
    height = np.zeros_like(p)
    for index, (h_base, t_base, lapse, p_base) in enumerate(_STANDARD_ATMOSPHERE):
        if index + 1 < len(_STANDARD_ATMOSPHERE):
            p_next = _STANDARD_ATMOSPHERE[index + 1][3]
            selected = (p <= p_base) & (p > p_next)
        else:
            selected = p <= p_base
        if index == 0:
            selected |= p > p_base
        if not np.any(selected):
            continue
        ratio = p[selected] / p_base
        if lapse:
            exponent = -_R_STANDARD * lapse / _G0
            height[selected] = h_base + t_base / lapse * (ratio ** exponent - 1.0)
        else:
            height[selected] = h_base - _R_STANDARD * t_base / _G0 * np.log(ratio)
    return height


def validate_sigma_interfaces(values, nz):
    sigma = np.asarray(values, dtype=np.float32)
    if sigma.shape != (nz + 1,):
        raise ValueError(
            f"physics.vertical.sigma_interfaces must contain {nz + 1} values")
    if not np.isclose(sigma[0], 1.0) or not np.isclose(sigma[-1], 0.0):
        raise ValueError(
            "sigma interfaces must start at 1 (surface) and end at 0 (top)")
    if np.any(np.diff(sigma) >= 0):
        raise ValueError("sigma interfaces must be strictly decreasing")
    return sigma


def sigma_to_hybrid(sigma_interfaces, top_pressure):
    """Express a pure sigma grid as hybrid coefficients."""
    sigma = np.asarray(sigma_interfaces, dtype=np.float64)
    hyai = float(top_pressure) * (1.0 - sigma)
    hybi = sigma.copy()
    hyai[0], hybi[0] = 0.0, 1.0
    hyai[-1], hybi[-1] = float(top_pressure), 0.0
    return hyai.astype(np.float32), hybi.astype(np.float32)


def generate_hybrid_coefficients(nz, top_pressure, reference_surface_pressure,
                                 transition_pressure, *, top_layers=7,
                                 top_stretch=5.0, surface_layers=6,
                                 surface_ratio=0.7, sigma_exponent=1.0):
    """Build a CAM-like hybrid grid from a few shape parameters.

    Interfaces are spaced in ``ln p`` at the reference surface pressure: the
    uppermost ``top_layers`` are stretched by up to ``top_stretch`` so the
    stratosphere is cheap and acts as a sponge, the troposphere is uniform in
    ``ln p``, and the lowest ``surface_layers`` shrink geometrically by
    ``surface_ratio`` to resolve the boundary layer.  Interfaces above
    ``transition_pressure`` are pure pressure surfaces (``B = 0``); below it
    ``B`` grows as ``((p - p_t) / (p_s0 - p_t)) ** sigma_exponent`` to one.
    """
    nz = int(nz)
    top_pressure = float(top_pressure)
    reference = float(reference_surface_pressure)
    transition = float(transition_pressure)
    top_layers, surface_layers = int(top_layers), int(surface_layers)
    if nz < 2:
        raise ValueError("a hybrid grid needs at least two layers")
    if not 0.0 < top_pressure < transition < reference:
        raise ValueError(
            "hybrid grid needs 0 < top_pressure < transition_pressure "
            "< reference_surface_pressure")
    if top_layers < 0 or surface_layers < 0:
        raise ValueError("hybrid top_layers and surface_layers must be >= 0")
    if nz - top_layers - surface_layers < 1:
        raise ValueError(
            "hybrid levels must exceed top_layers + surface_layers")
    if float(top_stretch) < 1.0:
        raise ValueError("hybrid top_stretch must be >= 1")
    if not 0.0 < float(surface_ratio) <= 1.0:
        raise ValueError("hybrid surface_ratio must lie in (0, 1]")
    if float(sigma_exponent) < 1.0:
        raise ValueError(
            "hybrid sigma_exponent must be >= 1 to keep layers positive")

    weights = []
    for k in range(top_layers):
        weights.append(1.0 + (float(top_stretch) - 1.0)
                       * (top_layers - k) / top_layers)
    weights.extend([1.0] * (nz - top_layers - surface_layers))
    weights.extend(float(surface_ratio) ** j
                   for j in range(1, surface_layers + 1))
    weights = np.asarray(weights, dtype=np.float64)
    delta = np.log(reference / top_pressure) / weights.sum()
    log_p = np.concatenate(
        [[np.log(top_pressure)], np.log(top_pressure) + np.cumsum(delta * weights)])
    pressure = np.exp(log_p)[::-1]          # surface -> top
    pressure[0], pressure[-1] = reference, top_pressure

    fraction = np.clip((pressure - transition) / (reference - transition), 0.0, 1.0)
    hybi = fraction ** float(sigma_exponent)
    hyai = pressure - hybi * reference
    hybi[0], hyai[0] = 1.0, 0.0
    hybi[-1], hyai[-1] = 0.0, top_pressure
    return hyai.astype(np.float32), hybi.astype(np.float32)


def validate_hybrid_coefficients(hyai, hybi, reference_surface_pressure,
                                 min_surface_pressure=None):
    """Check CAM-style hybrid coefficients ordered surface -> top."""
    hyai = np.asarray(hyai, dtype=np.float64)
    hybi = np.asarray(hybi, dtype=np.float64)
    if hyai.ndim != 1 or hyai.shape != hybi.shape or hyai.size < 2:
        raise ValueError("hyai and hybi must be equally long 1-D sequences")
    if not np.isclose(hybi[0], 1.0, atol=1e-6) or abs(hyai[0]) > 1e-3:
        raise ValueError("hybrid coefficients must start at the surface "
                         "(hyai=0, hybi=1)")
    if abs(hybi[-1]) > 1e-6 or hyai[-1] <= 0.0:
        raise ValueError("hybrid coefficients must end at a fixed-pressure "
                         "model top (hybi=0, hyai>0)")
    if np.any(np.diff(hybi) > 1e-9) or np.any(hybi < -1e-9) or np.any(hybi > 1 + 1e-9):
        raise ValueError("hybi must decrease monotonically from 1 to 0")
    reference = float(reference_surface_pressure)
    pressure = hyai + hybi * reference
    if np.any(np.diff(pressure) >= 0.0):
        raise ValueError("hybrid interface pressures must strictly decrease "
                         "with height at the reference surface pressure")
    if min_surface_pressure is not None:
        low = hyai + hybi * float(min_surface_pressure)
        if np.any(np.diff(low) >= 0.0):
            raise ValueError(
                "hybrid layers collapse at physics.dynamics."
                "min_surface_pressure_pa; raise it or lower sigma_exponent")
    return hyai.astype(np.float32), hybi.astype(np.float32)


def hybrid_pressures(xp, hyai, hybi, surface_pressure):
    """Interface and full-level pressure for a surface-pressure field."""
    a = xp.asarray(hyai, dtype=xp.float32)[:, None, None]
    b = xp.asarray(hybi, dtype=xp.float32)[:, None, None]
    p_i = a + b * surface_pressure[None, :, :]
    # Full levels use fixed arithmetic centres.  Keeping p linear in surface
    # pressure is essential for the transformed pressure-gradient term to
    # cancel terrain-following coordinate slopes discretely.
    return p_i, 0.5 * (p_i[:-1] + p_i[1:])


def exner(xp, pressure):
    return (pressure / P0) ** KAPPA


def hydrostatic_state(xp, temperature, humidity, surface_pressure,
                      hyai, hybi, surface_geopotential):
    """Diagnose interface pressure, full-level pressure and geopotential."""
    if getattr(xp, "__name__", "") == "cupy":
        from . import cuda_kernels
        if cuda_kernels.load():
            return cuda_kernels.hydrostatic_state(
                temperature, humidity, surface_pressure, hyai, hybi,
                surface_geopotential, RD)

    p_i, p_full = hybrid_pressures(xp, hyai, hybi, surface_pressure)
    tv = temperature * (1.0 + 0.608 * humidity)

    phi_interfaces = [surface_geopotential]
    phi_full = []
    for k in range(temperature.shape[0]):
        p_bottom = p_i[k]
        p_top = p_i[k + 1]
        p_mid = p_full[k]
        phi_bottom = phi_interfaces[-1]
        phi_full.append(
            phi_bottom + RD * tv[k] * xp.log(p_bottom / p_mid))
        phi_interfaces.append(
            phi_bottom + RD * tv[k] * xp.log(p_bottom / p_top))
    return (p_i, p_full, xp.stack(phi_full, axis=0),
            xp.stack(phi_interfaces, axis=0), tv)


def hybrid_mass_flux(xp, ops, u, v, surface_pressure, hyai, hybi):
    """Return ps tendency, layer mass divergence, interface flux and dp.

    The recurrence integrates the hybrid-coordinate continuity equation from
    the impermeable lower boundary.  Its final interface is zero to roundoff
    whenever the vertically integrated divergence and ps tendency use the
    same discrete operator.  Positive interface flux is downward.
    """
    a = xp.asarray(hyai, dtype=xp.float32)
    b = xp.asarray(hybi, dtype=xp.float32)
    da = (a[:-1] - a[1:])[:, None, None]
    db = (b[:-1] - b[1:])[:, None, None]
    dp = da + db * surface_pressure[None, :, :]
    mass_u = dp * u
    mass_v = dp * v
    if ops.advection_scheme == "muscl_tvd":
        # The pressure continuity equation and tracer transport share these
        # control-volume face mass fluxes, preserving a constant tracer.
        div_mass = ops.finite_volume_divergence(mass_u, mass_v)
    else:
        div_mass = ops.divergence(mass_u, mass_v)
    ps_tendency = -div_mass.sum(axis=0)
    fluxes = [xp.zeros_like(surface_pressure)]
    for k in range(u.shape[0]):
        fluxes.append(fluxes[-1] + db[k] * ps_tendency + div_mass[k])
    flux = xp.stack(fluxes, axis=0)
    # Remove accumulated floating-point residue without changing the two
    # boundary values.  This is normally tiny but matters in long integrations.
    if u.shape[0] > 1:
        ramp = xp.asarray(
            np.linspace(0.0, 1.0, u.shape[0] + 1, dtype=np.float32)
        )[:, None, None]
        flux = flux - ramp * flux[-1:]
    return ps_tendency, div_mass, flux, dp


def sigma_mass_flux(xp, ops, u, v, surface_pressure, sigma_thickness,
                    top_pressure):
    """Pure-sigma wrapper around :func:`hybrid_mass_flux`."""
    thickness = np.asarray(sigma_thickness, dtype=np.float64)
    sigma = np.concatenate([[1.0], 1.0 - np.cumsum(thickness)])
    sigma[-1] = 0.0
    hyai, hybi = sigma_to_hybrid(sigma, top_pressure)
    ps_tendency, div_mass, flux, _ = hybrid_mass_flux(
        xp, ops, u, v, surface_pressure, hyai, hybi)
    return ps_tendency, div_mass, flux


def vertical_face_values(xp, ops, field, interface_flux):
    """Upstream-biased values at the interior interfaces ``1 .. nz-1``.

    With the MUSCL scheme the face value is the upstream cell plus half of
    its limited slope, which is second order for smooth profiles and keeps
    the layer-by-layer transport monotone at the small vertical Courant
    numbers of a hydrostatic model.  Otherwise the first-order upstream
    value is used.
    """
    flux = interface_flux[1:-1]
    if ops.advection_scheme != "muscl_tvd" or field.shape[0] < 3:
        return xp.where(flux >= 0, field[1:], field[:-1])
    slope = xp.zeros_like(field)
    slope[1:-1] = ops._limited_slope(
        field[1:-1] - field[:-2], field[2:] - field[1:-1])
    return xp.where(flux >= 0,
                    field[1:] - 0.5 * slope[1:],
                    field[:-1] + 0.5 * slope[:-1])


def mass_consistent_transport(xp, ops, field, u, v, dp_old, dp_new,
                              div_mass, interface_flux, diffusivity, dt):
    """Flux-form horizontal/vertical transport of a layer scalar.

    ``dp_old``/``dp_new`` are the layer pseudo-densities before and after
    the surface-pressure update; the same interface flux is shared by the
    two adjacent layers so vertical exchange is exactly conservative.
    """
    if ops.cuda_adv is not None:
        return ops.cuda_adv.mass_transport(
            field, u, v, dp_old, dp_new, div_mass, interface_flux,
            ops.invdx_flat, float(ops.invdy), diffusivity, dt,
            scheme=ops.advection_scheme, limiter=ops.advection_limiter,
            coslat=ops.coslat_flat, invcoslat=ops.invcoslat_flat)

    numerator = dp_old * field
    if ops.advection_scheme == "muscl_tvd":
        mass_u = dp_old * u
        mass_v = dp_old * v
        scalar_divergence = ops.muscl_flux_divergence(field, mass_u, mass_v)
        face_mass_divergence = ops.finite_volume_divergence(mass_u, mass_v)
        # The correction form retains exact constant-field preservation even
        # if a repeated floating-point divergence differs in its last bit.
        numerator += dt * (
            -field * div_mass - scalar_divergence
            + field * face_mass_divergence)
    else:
        advective = ops.upwind_adv(field, u, v)
        numerator += dt * (-field * div_mass + dp_old * advective)

    if field.shape[0] > 1:
        transported = dt * interface_flux[1:-1] * vertical_face_values(
            xp, ops, field, interface_flux)
        numerator[:-1] += transported
        numerator[1:] -= transported

    out = numerator / xp.maximum(dp_new, xp.float32(1.0))
    if float(diffusivity):
        out += dt * float(diffusivity) * ops.lap(field)
    return out


def pressure_vertical_velocity(xp, ops, u, v, surface_pressure,
                               ps_tendency, interface_flux, hybi):
    """Diagnose pressure velocity omega = Dp/Dt at full levels."""
    b = xp.asarray(hybi, dtype=xp.float32)
    b_full = (0.5 * (b[:-1] + b[1:]))[:, None, None]
    ps_x, ps_y = ops.gradient(surface_pressure)
    material_ps = (ps_tendency[None, :, :]
                   + u * ps_x[None, :, :]
                   + v * ps_y[None, :, :])
    return (b_full * material_ps
            + 0.5 * (interface_flux[:-1] + interface_flux[1:]))


def sponge_profile(reference_pressure_layers, sponge_base_pressure, top_pressure):
    """Smooth 0..1 weight that switches on above ``sponge_base_pressure``."""
    p = np.asarray(reference_pressure_layers, dtype=np.float64)
    depth = max(float(sponge_base_pressure) - float(top_pressure), 1.0)
    fraction = np.clip((float(sponge_base_pressure) - p) / depth, 0.0, 1.0)
    return (np.sin(0.5 * np.pi * fraction) ** 2).astype(np.float32)


def column_total_energy(xp, u, v, temperature, humidity, layer_mass,
                        surface_geopotential, column_mass, latent_heat):
    """Vertically integrated total energy in J m-2.

    ``sum(dp/g * (cp T + K + L q)) + Phi_s * M`` is the quantity conserved by
    adiabatic, frictionless hydrostatic dynamics under a fixed-pressure top.
    """
    # Accumulate in float64: a column holds ~2.6e9 J m-2, whose float32
    # resolution (~256 J m-2) is as large as one step's energy change.
    f64 = xp.float64
    u64, v64 = u.astype(f64), v.astype(f64)
    kinetic = 0.5 * (u64 * u64 + v64 * v64)
    column = (layer_mass.astype(f64)
              * (CP * temperature.astype(f64) + kinetic
                 + float(latent_heat) * humidity.astype(f64))).sum(axis=0)
    return column + surface_geopotential.astype(f64) * column_mass.astype(f64)


def implicit_vertical_diffusion(xp, field, layer_mass, exchange, dt):
    """Backward-Euler vertical diffusion with conservative interface exchange.

    ``exchange[k]`` (kg m-2 s-1) couples layers ``k`` and ``k+1``; the
    column integral of ``layer_mass * field`` is conserved to roundoff and
    the scheme is unconditionally stable.
    """
    nz = field.shape[0]
    if nz < 2:
        return field
    lower = [None] + [-dt * exchange[k - 1] for k in range(1, nz)]
    upper = [-dt * exchange[k] for k in range(nz - 1)] + [None]
    diag = []
    rhs = []
    for k in range(nz):
        d = layer_mass[k].copy()
        if lower[k] is not None:
            d = d - lower[k]
        if upper[k] is not None:
            d = d - upper[k]
        diag.append(d)
        rhs.append(layer_mass[k] * field[k])
    for k in range(1, nz):
        w = lower[k] / diag[k - 1]
        diag[k] = diag[k] - w * upper[k - 1]
        rhs[k] = rhs[k] - w * rhs[k - 1]
    out = [None] * nz
    out[-1] = rhs[-1] / diag[-1]
    for k in range(nz - 2, -1, -1):
        out[k] = (rhs[k] - upper[k] * out[k + 1]) / diag[k]
    return xp.stack(out, axis=0)


def dry_convective_adjustment(xp, temperature, humidity, layer_mass,
                              exner_layers, passes=2, threshold_k=0.05):
    """Remove dry static instability with enthalpy-conserving pair mixing.

    Adjacent layers whose potential temperature decreases upward are reset to
    a common potential temperature chosen so ``sum(dp cp T)`` is unchanged;
    specific humidity is mixed with mass weights in the same pairs.
    """
    nz = temperature.shape[0]
    if nz < 2:
        return temperature, humidity
    T = list(temperature)
    q = None if humidity is None else list(humidity)
    for _ in range(int(passes)):
        for k in range(nz - 1):
            theta_low = T[k] / exner_layers[k]
            theta_high = T[k + 1] / exner_layers[k + 1]
            unstable = theta_low > theta_high + float(threshold_k)
            m_low, m_high = layer_mass[k], layer_mass[k + 1]
            theta_mixed = ((m_low * T[k] + m_high * T[k + 1])
                           / (m_low * exner_layers[k]
                              + m_high * exner_layers[k + 1]))
            T[k] = xp.where(unstable, theta_mixed * exner_layers[k], T[k])
            T[k + 1] = xp.where(unstable, theta_mixed * exner_layers[k + 1],
                                T[k + 1])
            if q is not None:
                q_mixed = (m_low * q[k] + m_high * q[k + 1]) / (m_low + m_high)
                q[k] = xp.where(unstable, q_mixed, q[k])
                q[k + 1] = xp.where(unstable, q_mixed, q[k + 1])
    return (xp.stack(T, axis=0),
            None if q is None else xp.stack(q, axis=0))
