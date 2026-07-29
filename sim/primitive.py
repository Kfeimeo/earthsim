"""Hydrostatic primitive-equation utilities in a terrain-following sigma grid.

Arrays are ordered from the lowest atmospheric layer to the highest.  Sigma is
one at the ground and zero at the fixed-pressure model top.  The prognostic
column mass is ``mu = ps - p_top``; zero sigma mass flux at both boundaries is
therefore the exact impermeability condition, including over terrain.
"""
import numpy as np

from .physics import RD


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


def hydrostatic_state(xp, temperature, humidity, surface_pressure,
                      sigma_interfaces, top_pressure, surface_geopotential):
    """Diagnose interface pressure, full-level pressure and geopotential."""
    if getattr(xp, "__name__", "") == "cupy":
        from . import cuda_kernels
        if cuda_kernels.load():
            return cuda_kernels.hydrostatic_state(
                temperature, humidity, surface_pressure, sigma_interfaces,
                top_pressure, surface_geopotential, RD)

    sigma_i = xp.asarray(sigma_interfaces, dtype=xp.float32)[:, None, None]
    mu = surface_pressure - float(top_pressure)
    p_i = float(top_pressure) + sigma_i * mu[None, :, :]
    # Full levels use fixed arithmetic sigma centres.  Keeping p linear in
    # surface pressure is essential for the transformed pressure-gradient
    # term to cancel terrain-following coordinate slopes discretely.
    p_full = 0.5 * (p_i[:-1] + p_i[1:])
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


def sigma_mass_flux(xp, ops, u, v, surface_pressure, sigma_thickness,
                    top_pressure):
    """Return ps tendency and interface ``mu * sigma_dot`` mass flux.

    The recurrence integrates the sigma-coordinate continuity equation from
    the impermeable lower boundary.  Its final interface is zero to roundoff
    whenever the vertically integrated divergence and ps tendency use the
    same discrete operator.
    """
    mu = surface_pressure - float(top_pressure)
    mass_u = mu[None, :, :] * u
    mass_v = mu[None, :, :] * v
    if ops.advection_scheme == "muscl_tvd":
        # The pressure continuity equation and tracer transport share these
        # control-volume face mass fluxes, preserving a constant tracer.
        div_mass = ops.finite_volume_divergence(mass_u, mass_v)
    else:
        div_mass = ops.divergence(mass_u, mass_v)
    ds = xp.asarray(sigma_thickness, dtype=xp.float32)[:, None, None]
    ps_tendency = -(ds * div_mass).sum(axis=0)
    fluxes = [xp.zeros_like(surface_pressure)]
    for k in range(u.shape[0]):
        fluxes.append(
            fluxes[-1] + ds[k] * (ps_tendency + div_mass[k]))
    flux = xp.stack(fluxes, axis=0)
    # Remove accumulated floating-point residue without changing the two
    # boundary values.  This is normally tiny but matters in long integrations.
    if u.shape[0] > 1:
        ramp = xp.asarray(
            np.linspace(0.0, 1.0, u.shape[0] + 1, dtype=np.float32)
        )[:, None, None]
        flux = flux - ramp * flux[-1:]
    return ps_tendency, div_mass, flux


def mass_consistent_transport(xp, ops, field, u, v, mu_old, mu_new,
                              sigma_thickness, div_mass, interface_flux,
                              diffusivity, dt):
    """Flux-form horizontal/vertical transport of a sigma-level scalar."""
    if ops.cuda_adv is not None:
        return ops.cuda_adv.mass_transport(
            field, u, v, mu_old, mu_new, sigma_thickness,
            div_mass, interface_flux, ops.invdx_flat, float(ops.invdy),
            diffusivity, dt, scheme=ops.advection_scheme,
            limiter=ops.advection_limiter, coslat=ops.coslat_flat,
            invcoslat=ops.invcoslat_flat)

    ds = xp.asarray(sigma_thickness, dtype=xp.float32)[:, None, None]
    numerator = mu_old[None, :, :] * ds * field
    if ops.advection_scheme == "muscl_tvd":
        mass_u = mu_old[None, :, :] * u
        mass_v = mu_old[None, :, :] * v
        scalar_divergence = ops.muscl_flux_divergence(
            field, mass_u, mass_v)
        face_mass_divergence = ops.finite_volume_divergence(mass_u, mass_v)
        # The correction form retains exact constant-field preservation even
        # if a repeated floating-point divergence differs in its last bit.
        numerator += dt * ds * (
            -field * div_mass - scalar_divergence
            + field * face_mass_divergence)
    else:
        advective = ops.upwind_adv(field, u, v)
        numerator += dt * ds * (
            -field * div_mass + mu_old[None, :, :] * advective)

    # One flux is shared by the two adjacent layers, making exchange exactly
    # conservative. Positive flux is toward increasing sigma (downward).
    for k in range(field.shape[0] - 1):
        flux = interface_flux[k + 1]
        upstream = xp.where(flux >= 0, field[k + 1], field[k])
        transported = dt * flux * upstream
        numerator[k] += transported
        numerator[k + 1] -= transported

    out = numerator / xp.maximum(
        mu_new[None, :, :] * ds, xp.float32(1.0))
    if float(diffusivity):
        out += dt * float(diffusivity) * ops.lap(field)
    return out


def pressure_vertical_velocity(xp, ops, u, v, surface_pressure,
                               ps_tendency, interface_flux, sigma_centres,
                               top_pressure):
    """Diagnose pressure velocity omega = Dp/Dt at full sigma levels."""
    mu = surface_pressure - float(top_pressure)
    ps_x, ps_y = ops.gradient(surface_pressure)
    material_ps = (ps_tendency[None, :, :]
                   + u * ps_x[None, :, :]
                   + v * ps_y[None, :, :])
    sigma = xp.asarray(sigma_centres, dtype=xp.float32)[:, None, None]
    sigma_dot = (0.5 * (interface_flux[:-1] + interface_flux[1:])
                 / xp.maximum(mu[None, :, :], 1.0))
    return sigma * material_ps + mu[None, :, :] * sigma_dot
