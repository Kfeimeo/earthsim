import numpy as np
import pytest

from sim.balance import budgets, filter_increments, polar_divergence_damping
from sim.config import load_config
from sim.model import EarthModel
from sim.primitive import sigma_mass_flux


def config(nlat=36, nlon=32):
    cfg = load_config()
    cfg["backend"] = "cpu"
    cfg["data"]["init_mode"] = "ideal"
    cfg["grid"].update(nlat=nlat, nlon=nlon, topo_files=[])
    cfg["time"]["dt"] = 5.0
    cfg["physics"]["initial_conditions"]["balanced_perturbation_k"] = 0.0
    return cfg


@pytest.mark.parametrize("nlat", [35, 36, 720])
def test_gradient_wind_and_mass_balance(nlat):
    m = EarthModel(config(nlat))
    assert max(m.initial_balance["max_residual_ms2"]) < 1e-8
    assert m.initial_balance["ps_tendency_max_pas"] == 0
    np.testing.assert_array_equal(m.v_layers, 0)
    np.testing.assert_array_equal(m.surface_geopotential, 0)
    assert np.max(np.abs(m.u_layers)) > 1
    np.testing.assert_allclose(m.u_layers, m.u_layers[:, ::-1], atol=1e-3)


def test_unbalanced_initial_state_cannot_start():
    m = EarthModel(config())
    m.u_layers[:] = 100.0
    with pytest.raises(ValueError, match="initial meridional momentum residual"):
        m.step()
    assert m.step_count == 0


def test_temperature_edit_is_included_in_startup_gate():
    m = EarthModel(config())
    m.T_layers[:, m.nlat // 2:] += 30
    with pytest.raises(ValueError, match="initial meridional momentum residual"):
        m.step()


def test_nonfinite_initial_state_cannot_start():
    m = EarthModel(config())
    m.T_layers[0, 2, 3] = np.nan
    with pytest.raises(ValueError, match="initial meridional momentum residual"):
        m.step()


def test_hydrostatic_energy_includes_finite_top_pressure():
    from sim.physics import CP, RD, A_EARTH
    m = EarthModel(config())
    zero = np.zeros_like(m.T_layers)
    temperature = np.full_like(zero, 280)
    value = budgets(m, m.surface_pressure, (zero, zero, temperature, zero))
    ps, top, g = float(m.surface_pressure[0, 0]), 10000, float(m.cfg.physics.g_eff)
    expected = 4 * np.pi * A_EARTH ** 2 / g * (
        (ps - top) * CP * 280 - top * RD * 280 * np.log(ps / top))
    np.testing.assert_allclose(value["energy_j"], expected, rtol=2e-7)


def test_joint_filter_preserves_mass_and_mass_weighted_zonal_sums():
    m = EarthModel(config())
    rng = np.random.default_rng(12)
    ps0 = m.surface_pressure.copy()
    ps0 += rng.normal(0, 1000, ps0.shape).astype(np.float32)
    ps = ps0 + rng.normal(0, 20, ps0.shape).astype(np.float32)
    old = (m.u_layers, m.v_layers, m.T_layers, m.q_layers)
    fields = [x + rng.normal(0, 0.01, x.shape).astype(np.float32) for x in old]
    filtered_ps, filtered = filter_increments(m, ps, fields, ps0, old)
    np.testing.assert_allclose(filtered_ps.sum(-1), ps.sum(-1), rtol=2e-7)
    top = float(m.cfg.physics.dynamics.top_pressure_pa)
    for a, b in zip(fields, filtered):
        np.testing.assert_allclose(((ps - top) * a).sum(-1),
                                   ((filtered_ps - top) * b).sum(-1),
                                   rtol=3e-6, atol=2)
    unchanged_ps, unchanged = filter_increments(m, ps0, old, ps0, old)
    np.testing.assert_array_equal(unchanged_ps, ps0)
    for a, b in zip(old, unchanged):
        np.testing.assert_allclose(a, b, rtol=1e-7)


def test_high_latitude_zonal_mean_damping_removes_kinetic_energy():
    m = EarthModel(config(720, 16))
    polar = np.abs(m.lats) > 70
    v = np.zeros_like(m.v_layers)
    v[:, polar] = (np.arange(m.nlat)[polar] % 2 * 2 - 1)[None, :, None]
    ps = m.surface_pressure.copy()
    ps *= 1 + 0.02 * np.sin(m.ops.lon_rad)
    after = polar_divergence_damping(m, v, 5, ps)
    weight = (ps - 10000) * m.ops.coslat
    assert np.sum(weight * after ** 2) < np.sum(weight * v ** 2)
    np.testing.assert_array_equal(after[:, np.abs(m.lats) < 60], 0)


def test_continuity_conserves_true_spherical_mass_with_cos_clamp():
    m = EarthModel(config(180, 32))
    rng = np.random.default_rng(9)
    u = rng.normal(size=m.u_layers.shape).astype(np.float32)
    v = rng.normal(size=m.v_layers.shape).astype(np.float32)
    ps_t, _, _ = sigma_mass_flux(np, m.ops, u, v, m.surface_pressure,
                                m.sigma_thickness, 10000)
    weighted = ps_t.astype(np.float64) * m.ops.coslat
    assert abs(weighted.sum()) / np.abs(weighted).sum() < 1e-7


def test_unforced_balanced_jet_stays_close_for_100_steps():
    cfg = config(72, 32)
    p = cfg["physics"]
    p.update(moisture=False, radiation=False, ocean=False,
             visc=0, diff_T=0, diff_q=0,
             drag_ocean_atmosphere=0, drag_land_atmosphere=0)
    p["surface_flux"]["sensible_heat_coeff"] = 0
    m = EarthModel(cfg)
    initial = m.u_layers.copy()
    initial_mass = budgets(m, m.surface_pressure,
                           (m.u_layers, m.v_layers, m.T_layers, m.q_layers))["mass_kg"]
    m.step(100)
    assert np.max(np.abs(m.v_layers)) < 1e-3
    assert np.max(np.abs(m.u_layers - initial)) < 1e-3
    final_mass = budgets(m, m.surface_pressure,
                         (m.u_layers, m.v_layers, m.T_layers, m.q_layers))["mass_kg"]
    assert abs(final_mass / initial_mass - 1) < 1e-7


def test_cuda_balanced_moist_step_and_double_precision_filter():
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("no CUDA device")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA unavailable")
    cfg = config()
    cpu = EarthModel(cfg)
    cpu.step(3)
    cfg["backend"] = "cuda"
    gpu = EarthModel(cfg)
    assert gpu.initial_balance["passed"]
    gpu.step(3)
    for name in ("T_layers", "q_layers", "u_layers", "v_layers", "surface_pressure"):
        actual = cp.asnumpy(getattr(gpu, name))
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, getattr(cpu, name), rtol=1e-5, atol=2e-3)
    field = np.random.default_rng(1).normal(size=(36, 32))
    actual = gpu.ops.polar_filter(cp.asarray(field))
    assert actual.dtype == cp.float64
    np.testing.assert_allclose(cp.asnumpy(actual), cpu.ops.polar_filter(field), atol=1e-12)
