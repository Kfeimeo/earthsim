import numpy as np
import pytest

from sim.config import load_config
from sim.model import EarthModel
from sim.physics import CP, RD
from sim.primitive import (
    dry_convective_adjustment,
    exner,
    generate_hybrid_coefficients,
    hybrid_mass_flux,
    implicit_vertical_diffusion,
    mass_consistent_transport,
    sigma_to_hybrid,
    sponge_profile,
    standard_atmosphere_height,
    validate_hybrid_coefficients,
)


def hybrid_config(nlat=12, nlon=24, dt=60.0):
    cfg = load_config()
    cfg["backend"] = "cpu"
    cfg["grid"].update(nlat=nlat, nlon=nlon, topo_file="")
    cfg["grid"]["topo_files"] = []
    cfg["data"]["init_mode"] = "ideal"
    cfg["time"]["dt"] = dt
    cfg["physics"]["dynamics"].update(
        core="primitive_equations", top_pressure_pa=300.0,
        min_surface_pressure_pa=45000.0)
    cfg["physics"]["vertical"]["coordinate"] = "hybrid"
    cfg["physics"]["vertical"]["hybrid"]["levels"] = 20
    return cfg


def adiabatic(cfg, winds=True):
    cfg["physics"].update(
        moisture=False, radiation=False, ocean=False,
        ideal_wind_enabled=winds, ideal_wave_amp_K=0.0,
        ideal_humidity_wave=0.0,
        drag_ocean_atmosphere=0.0, drag_land_atmosphere=0.0)
    cfg["physics"]["surface_flux"]["sensible_heat_coeff"] = 0.0
    cfg["physics"]["column_physics"].update(
        vertical_diffusion=False, dry_adjustment=False)
    return cfg


# --------------------------------------------------------------- coefficients
def test_generated_grid_is_cam_like():
    hyai, hybi = generate_hybrid_coefficients(26, 300.0, 101300.0, 10000.0)
    pressure = hyai + hybi * 101300.0

    assert hyai.shape == hybi.shape == (27,)
    assert pressure[0] == pytest.approx(101300.0)
    assert pressure[-1] == pytest.approx(300.0)
    assert np.all(np.diff(pressure) < 0)
    # Pure pressure surfaces above the transition, terrain following below.
    assert np.all(hybi[pressure <= 10000.0 + 1e-3] == 0.0)
    assert hybi[0] == 1.0 and np.all(np.diff(hybi) <= 0)
    # The lowest layer resolves the boundary layer, the top layer is thick.
    assert pressure[0] - pressure[1] < 2500.0
    assert pressure[-2] - pressure[-1] > 200.0
    # Layers stay positive over the highest smoothed terrain.
    low = hyai + hybi * 45000.0
    assert np.all(np.diff(low) < 0)


def test_hybrid_validation_rejects_bad_coefficients():
    with pytest.raises(ValueError):
        validate_hybrid_coefficients([0.0, 300.0, 500.0], [1.0, 0.0, 0.0], 1e5)
    with pytest.raises(ValueError):
        validate_hybrid_coefficients([0.0, 0.0], [1.0, 0.0], 1e5)
    with pytest.raises(ValueError):
        validate_hybrid_coefficients([0.0, 100.0, 300.0], [1.0, 0.2, 0.5], 1e5)
    hyai, hybi = sigma_to_hybrid([1.0, 0.5, 0.0], 10000.0)
    np.testing.assert_allclose(hyai, [0.0, 5000.0, 10000.0])
    np.testing.assert_allclose(hybi, [1.0, 0.5, 0.0])


def test_standard_atmosphere_heights():
    heights = standard_atmosphere_height([101325.0, 22632.0, 5474.9, 300.0])
    assert heights[0] == pytest.approx(0.0, abs=1.0)
    assert heights[1] == pytest.approx(11000.0, abs=10.0)
    assert heights[2] == pytest.approx(20000.0, abs=10.0)
    assert 37000.0 < heights[3] < 41000.0


# --------------------------------------------------------------------- model
def test_hybrid_model_geometry():
    model = EarthModel(hybrid_config())

    assert model.vertical_coordinate == "hybrid"
    assert model.nz == 20
    assert model.top_pressure_pa == pytest.approx(300.0)
    np.testing.assert_allclose(model.pressure_interfaces_pa[-1], 300.0)
    np.testing.assert_allclose(model.pressure_interfaces_pa[0],
                               model.surface_pressure)
    assert np.all(np.diff(model.pressure_layers_pa, axis=0) < 0)
    assert np.all(np.diff(model.h_layers, axis=0) > 0)
    assert np.all(np.diff(model.levels_m) > 0)
    # Pure-pressure layers have the same pressure in every column.
    top = model.pressure_layers_pa[-1]
    assert float(np.ptp(top)) < 1e-3
    assert model.sponge_active
    assert model.sponge_weights[-1] > 0.9 and model.sponge_weights[0] == 0.0


def test_hybrid_mass_flux_boundaries_and_constant_tracer():
    model = EarthModel(adiabatic(hybrid_config()))
    xp = model.xp
    ps_t, div_mass, flux, dp0 = hybrid_mass_flux(
        xp, model.ops, model.u_layers, model.v_layers,
        model.surface_pressure, model.hyai, model.hybi)

    np.testing.assert_allclose(flux[0], 0.0)
    np.testing.assert_allclose(flux[-1], 0.0, atol=2e-6)
    # Pressure layers carry mass flux only through horizontal divergence.
    np.testing.assert_allclose(
        dp0[-1], float(model.hyai[-2] - model.hyai[-1]), rtol=1e-6)

    ps_new = model.surface_pressure + model.dt * ps_t
    dp1 = (model._d_hyai_xp + model._d_hybi_xp * ps_new[None]).astype(xp.float32)
    constant = np.full_like(model.T_layers, 7.0)
    transported = mass_consistent_transport(
        xp, model.ops, constant, model.u_layers, model.v_layers,
        dp0, dp1, div_mass, flux, 0.0, model.dt)
    np.testing.assert_allclose(transported, constant, rtol=2e-6, atol=2e-6)


def test_uniform_potential_temperature_is_preserved_by_adiabatic_step():
    cfg = adiabatic(hybrid_config())
    cfg["physics"].update(visc=0.0, diff_T=0.0)
    cfg["physics"]["dynamics"].update(energy_fixer=False)
    cfg["physics"]["dynamics"]["upper_damping"]["enabled"] = False
    # A 100 hPa top keeps T = theta * Exner inside the temperature bounds,
    # and the polar filter is off because it smooths T, not theta.
    cfg["physics"]["dynamics"]["top_pressure_pa"] = 10000.0
    cfg["physics"]["vertical"]["hybrid"]["transition_pressure_pa"] = 30000.0
    cfg["numerics"]["polar_filter_passes"] = 0
    model = EarthModel(cfg)
    theta = 320.0
    model.T_layers[...] = theta * np.asarray(model.exner_layers)
    model._refresh_primitive_diagnostics()

    model.step(10)

    recovered = model.T_layers / model.exner_layers
    # Flux-form theta transport with a divergent wind keeps a uniform theta
    # uniform; a T formulation with an explicit omega term would not.
    np.testing.assert_allclose(recovered, theta, rtol=5e-5)


def test_isothermal_atmosphere_is_balanced_over_terrain_in_hybrid_grid():
    model = EarthModel(adiabatic(hybrid_config(), winds=False))
    temperature = 280.0
    model.T_layers[...] = temperature
    model.q_layers[...] = 0.0
    model.surface_pressure[...] = (
        101300.0 * np.exp(
            -np.asarray(model.surface_geopotential) / (RD * temperature)))
    model._refresh_primitive_diagnostics()

    pgf_x, pgf_y = model._primitive_pressure_gradient()
    assert float(np.hypot(pgf_x, pgf_y).max()) < 2.0e-4

    model.step(20)
    assert float(np.hypot(model.u_layers, model.v_layers).max()) < 0.1
    np.testing.assert_allclose(model.sigma_dot_interfaces[[0, -1]], 0.0,
                               atol=1e-8)


def test_energy_fixer_closes_the_adiabatic_budget():
    cfg = adiabatic(hybrid_config())
    increment = 1.0e-4
    cfg["physics"]["dynamics"]["energy_fixer_min_increment_k"] = increment
    model = EarthModel(cfg)
    before = model.energy_diagnostics()
    column_mass = (before["mean_surface_pressure_pa"] - 300.0) / 9.8
    # One pending fixer increment, plus float32 rounding of the applied
    # increments, bounds the residual drift.
    budget = 3.0 * increment * CP * column_mass
    applied = 0
    for _ in range(25):
        model.step(4)
        after = model.energy_diagnostics()
        # Horizontal diffusion and the discrete adiabatic step lose energy;
        # the fixer returns it so the dry adiabatic total is conserved.
        assert abs(after["total_energy_j_m2"]
                   - before["total_energy_j_m2"]) < budget
        applied += after["energy_fixer_dT_k"] != 0.0
    assert applied > 0
    assert after["limiter_counts"] == {
        "wind": 0, "temperature": 0, "humidity": 0, "surface_pressure": 0}
    assert after["mean_surface_pressure_pa"] == pytest.approx(
        before["mean_surface_pressure_pa"], abs=0.05)

    cfg["physics"]["dynamics"]["energy_fixer"] = False
    unfixed = EarthModel(cfg)
    unfixed.step(100)
    drift = unfixed.energy_diagnostics()
    assert drift["energy_fixer_dT_k"] == 0.0
    assert abs(drift["total_energy_j_m2"]
               - before["total_energy_j_m2"]) > budget


def test_step_without_physics_keeps_water_and_mass():
    cfg = hybrid_config()
    cfg["physics"].update(radiation=False, ocean=False)
    cfg["physics"]["moisture_transport"]["subsidence_drying_coeff"] = 0.0
    cfg["physics"].update(c_evap=0.0, rh_crit=5.0)   # no evaporation or rain
    model = EarthModel(cfg)
    before = model.energy_diagnostics()
    model.step(10)
    after = model.energy_diagnostics()
    assert after["total_water_kg_m2"] == pytest.approx(
        before["total_water_kg_m2"], rel=1e-5)
    assert after["mean_surface_pressure_pa"] == pytest.approx(
        before["mean_surface_pressure_pa"], abs=0.05)


# ----------------------------------------------------------- column physics
def test_implicit_vertical_diffusion_conserves_and_smooths():
    rng = np.random.default_rng(3)
    nz, shape = 6, (6, 3, 4)
    field = rng.uniform(280.0, 320.0, size=shape).astype(np.float64)
    layer_mass = rng.uniform(500.0, 2000.0, size=shape)
    exchange = rng.uniform(0.0, 5.0, size=(nz - 1, 3, 4))
    before = (layer_mass * field).sum(axis=0)

    mixed = implicit_vertical_diffusion(np, field, layer_mass, exchange, 600.0)

    np.testing.assert_allclose((layer_mass * mixed).sum(axis=0), before,
                               rtol=1e-10)
    assert float(np.ptp(mixed, axis=0).mean()) < float(np.ptp(field, axis=0).mean())
    assert mixed.min() >= field.min() - 1e-9 and mixed.max() <= field.max() + 1e-9


def test_dry_adjustment_removes_instability_and_conserves_enthalpy():
    pressure = np.array([95000.0, 85000.0, 75000.0, 65000.0])[:, None, None]
    pi = exner(np, pressure) * np.ones((4, 2, 2))
    layer_mass = np.full((4, 2, 2), 1000.0)
    theta = np.array([310.0, 300.0, 305.0, 312.0])[:, None, None]
    temperature = theta * pi
    humidity = np.array([0.012, 0.004, 0.003, 0.002])[:, None, None] * np.ones_like(pi)
    enthalpy = (layer_mass * CP * temperature).sum(axis=0)
    water = (layer_mass * humidity).sum(axis=0)

    adjusted, q = dry_convective_adjustment(
        np, temperature, humidity, layer_mass, pi, passes=4)

    theta_new = adjusted / pi
    assert np.all(np.diff(theta_new, axis=0) >= -0.06)
    np.testing.assert_allclose((layer_mass * CP * adjusted).sum(axis=0),
                               enthalpy, rtol=1e-10)
    np.testing.assert_allclose((layer_mass * q).sum(axis=0), water, rtol=1e-10)


def test_surface_fluxes_warm_a_boundary_layer_not_one_layer():
    cfg = hybrid_config()
    cfg["physics"].update(ocean=False, radiation=False, moisture=False,
                          ideal_wind_enabled=False)
    cfg["physics"]["column_physics"].update(
        vertical_diffusion=False, dry_adjustment=False)
    cfg["physics"]["dynamics"].update(energy_fixer=False)
    model = EarthModel(cfg)
    model.Ts[...] = model.T_layers[0] + 20.0     # strong sensible heating
    before = model.T_layers.copy()
    model.step(1)
    warming = np.asarray(model.T_layers - before)
    column = warming.mean(axis=(1, 2))
    assert column[0] > 0.0
    # Several layers above the lowest one warm as well.
    assert np.count_nonzero(column[:8] > 0.05 * column[0]) >= 3
    assert float(np.abs(warming).max()) < 5.0


def test_rayleigh_sponge_damps_eddies_only():
    model = EarthModel(adiabatic(hybrid_config()))
    k0 = model._sponge_first_level
    u = np.zeros_like(model.u_layers)
    v = np.zeros_like(model.v_layers)
    u[k0:] = 30.0
    u[k0:, :, ::2] += 10.0
    u[k0:, :, 1::2] -= 10.0
    zonal_mean = u.mean(axis=-1, keepdims=True).copy()

    model.dt = 3.0 * 86400.0
    du, dv = model._rayleigh_sponge(u.copy(), v.copy())

    np.testing.assert_allclose(du.mean(axis=-1, keepdims=True), zonal_mean,
                               rtol=1e-6)
    tau = float(model.cfg.physics.dynamics.upper_damping.rayleigh_tau_s)
    expected = 1.0 / (1.0 + model.dt * model.sponge_weights[-1] / tau)
    eddy_before = np.abs(u[-1] - zonal_mean[-1]).mean()
    eddy_after = np.abs(du[-1] - zonal_mean[-1]).mean()
    assert eddy_after == pytest.approx(expected * eddy_before, rel=1e-5)
    assert eddy_after < 0.3 * eddy_before
    np.testing.assert_allclose(du[:k0], 0.0)


def test_sponge_profile_shape():
    weights = sponge_profile([90000.0, 10000.0, 4000.0, 1000.0, 300.0],
                             5000.0, 300.0)
    assert weights[0] == 0.0 and weights[1] == 0.0
    assert 0.0 < weights[2] < weights[3] < weights[4] <= 1.0


def test_divergence_damping_opposes_divergence():
    model = EarthModel(adiabatic(hybrid_config()))
    lon = np.radians(model.lons)[None, None, :]
    u = (10.0 * np.sin(4 * lon) * np.ones_like(model.u_layers)).astype(np.float32)
    v = np.zeros_like(u)
    div = model.ops.divergence(u, v)
    damp_x, damp_y = model._divergence_damping(u, v)
    new_div = model.ops.divergence(u + model.dt * damp_x, v + model.dt * damp_y)
    assert float(np.abs(new_div).mean()) < float(np.abs(div).mean())


def test_energy_diagnostics_report_expected_keys():
    model = EarthModel(hybrid_config())
    model.step(2)
    out = model.energy_diagnostics()
    for key in ("total_energy_j_m2", "kinetic_energy_j_m2", "enthalpy_j_m2",
                "latent_energy_j_m2", "dynamics_energy_error_w_m2",
                "energy_fixer_dT_k", "physics_energy_change_w_m2",
                "physics_forcing_w_m2", "mean_surface_pressure_pa",
                "total_water_kg_m2", "advective_cfl", "gravity_wave_cfl",
                "limiter_counts"):
        assert key in out
    assert out["gravity_wave_cfl"] < 1.0
    column = model.atmosphere_column_cpu(3, 5)
    assert column["theta_k"].shape == (model.nz,)
    assert np.all(np.diff(column["pressure_pa"]) < 0)


def test_sigma_coordinate_still_available():
    cfg = hybrid_config()
    cfg["physics"]["vertical"]["coordinate"] = "sigma"
    cfg["physics"]["dynamics"]["top_pressure_pa"] = 10000.0
    model = EarthModel(cfg)
    assert model.vertical_coordinate == "sigma"
    assert model.nz == len(cfg["physics"]["vertical"]["levels_m"])
    np.testing.assert_allclose(model.hybi, cfg["physics"]["vertical"]["sigma_interfaces"])
    model.step(2)
    model.check_health()
