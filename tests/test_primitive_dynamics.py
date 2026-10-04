import numpy as np

from sim.config import load_config
from sim.model import EarthModel
from sim.physics import A_EARTH, Ops, RD, qsat
from sim.primitive import sigma_mass_flux, mass_consistent_transport


def primitive_config():
    cfg = load_config()
    cfg["backend"] = "cpu"
    cfg["grid"].update(nlat=18, nlon=36, topo_file="")
    cfg["grid"]["topo_files"] = []
    cfg["data"]["init_mode"] = "ideal"
    cfg["time"]["dt"] = 20.0
    cfg["physics"]["dynamics"]["core"] = "primitive_equations"
    cfg["physics"].update(
        moisture=False,
        radiation=False,
        ocean=False,
        ideal_wind_enabled=False,
        ideal_wave_amp_K=0.0,
        ideal_humidity_wave=0.0,
        drag_ocean_atmosphere=0.0,
        drag_land_atmosphere=0.0,
    )
    return cfg


def test_hydrostatic_pressure_and_height_are_monotone():
    model = EarthModel(primitive_config())

    assert np.all(np.diff(model.pressure_layers_pa, axis=0) < 0)
    assert np.all(np.diff(model.h_layers, axis=0) > 0)
    assert np.all(model.pressure_interfaces_pa[0] == model.surface_pressure)
    assert np.allclose(
        model.pressure_interfaces_pa[-1],
        model.cfg.physics.dynamics.top_pressure_pa,
    )


def test_sigma_boundaries_are_impermeable_and_transport_is_mass_consistent():
    model = EarthModel(primitive_config())
    xp, top = model.xp, float(model.cfg.physics.dynamics.top_pressure_pa)
    ps_t, div_mass, flux = sigma_mass_flux(
        xp, model.ops, model.u_layers, model.v_layers,
        model.surface_pressure, model.sigma_thickness, top)

    np.testing.assert_allclose(flux[0], 0.0)
    np.testing.assert_allclose(flux[-1], 0.0, atol=2e-6)

    ps_new = model.surface_pressure + model.dt * ps_t
    constant = np.full_like(model.T_layers, 7.0)
    thickness = model.sigma_thickness[:, None, None]
    transported = mass_consistent_transport(
        xp, model.ops, constant, model.u_layers, model.v_layers,
        thickness * (model.surface_pressure - top)[None],
        thickness * (ps_new - top)[None],
        div_mass, flux, 0.0, model.dt)
    np.testing.assert_allclose(transported, constant, rtol=2e-6, atol=2e-6)


def test_muscl_tvd_spatial_transport_is_second_order():
    """The smooth-wave error should fall by about four when dx is halved."""
    errors = []
    for nlon in (32, 64, 128):
        ops = Ops(
            np, np.array([0.0], np.float32), nlon,
            advection_scheme="muscl_tvd", advection_limiter="mc")
        longitude = 2.0 * np.pi * np.arange(nlon) / nlon
        field = np.sin(longitude)[None, :].astype(np.float32)
        u = np.full_like(field, 10.0)
        actual = ops.muscl_adv(field, u, np.zeros_like(field))
        exact = (-10.0 / A_EARTH * np.cos(longitude))[None, :]
        errors.append(float(np.mean(np.abs(actual - exact))))

    assert errors[0] / errors[1] > 3.5
    assert errors[1] / errors[2] > 3.5


def test_muscl_tvd_preserves_bounds_and_integrated_tracer_mass():
    nlon = 64
    ops = Ops(
        np, np.array([0.0], np.float32), nlon,
        advection_scheme="muscl_tvd", advection_limiter="mc")
    field = np.zeros((1, 1, nlon), np.float32)
    field[..., 20:40] = 1.0
    u = np.full_like(field, 20.0)
    v = np.zeros_like(field)
    mu = np.full((1, 1, nlon), 90000.0, np.float32)
    div_mass = ops.finite_volume_divergence(mu * u, mu * v)
    vertical_flux = np.zeros((2, 1, nlon), np.float32)
    dt = 0.4 / (20.0 * float(ops.invdx[0, 0]))
    initial_mass = float((mu * field).sum())

    for _ in range(20):
        field = mass_consistent_transport(
            np, ops, field, u, v, mu, mu, div_mass,
            vertical_flux, 0.0, dt)

    assert float(field.min()) >= -2.0e-7
    assert float(field.max()) <= 1.0 + 2.0e-7
    final_mass = float((mu * field).sum())
    np.testing.assert_allclose(final_mass, initial_mass, rtol=2e-7)


def test_isothermal_atmosphere_remains_balanced_over_terrain():
    model = EarthModel(primitive_config())
    temperature = 280.0
    model.T_layers[...] = temperature
    model.q_layers[...] = 0.0
    model.surface_pressure[...] = (
        101300.0 * np.exp(
            -np.asarray(model.surface_geopotential) / (RD * temperature)))
    model.u_layers[...] = 0.0
    model.v_layers[...] = 0.0
    model._refresh_primitive_diagnostics()

    pgf_x, pgf_y = model._primitive_pressure_gradient()
    assert float(np.hypot(pgf_x, pgf_y).max()) < 2.0e-4

    model.step(20)
    assert float(np.hypot(model.u_layers, model.v_layers).max()) < 0.1
    np.testing.assert_allclose(
        model.sigma_dot_interfaces[[0, -1]], 0.0, atol=1e-8)


def test_saturation_humidity_decreases_with_height_pressure():
    temperature = np.array([280.0, 280.0], np.float32)
    saturation = qsat(np, temperature, np.array([100000.0, 50000.0]))

    assert saturation[1] > saturation[0]
    # At fixed T, lower ambient pressure corresponds to a larger saturation
    # mixing ratio. This guards against the former fixed-1000-hPa calculation.
    np.testing.assert_allclose(
        saturation,
        0.622 * 610.78 * np.exp(
            17.27 * (temperature - 273.15) / (temperature - 35.85))
        / (np.array([100000.0, 50000.0])
           - 0.378 * 610.78 * np.exp(
               17.27 * (temperature - 273.15) / (temperature - 35.85))),
        rtol=1e-6,
    )


def test_primitive_step_uses_two_hydrostatic_integrations(monkeypatch):
    import sim.model as model_module

    model = EarthModel(primitive_config())
    original = model_module.hydrostatic_state
    calls = []

    def counted(*args, **kwargs):
        calls.append(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(model_module, "hydrostatic_state", counted)
    model.step(1)

    assert len(calls) == 2
