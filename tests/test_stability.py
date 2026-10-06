import numpy as np

from sim.balance import budgets, budget_changes
from sim.config import load_config
from sim.diagnostics import stability_report
from sim.model import EarthModel


def model():
    cfg = load_config()
    cfg["backend"] = "cpu"
    cfg["data"]["init_mode"] = "ideal"
    cfg["grid"].update(nlat=18, nlon=36, topo_files=[])
    cfg["time"]["dt"] = 5
    return EarthModel(cfg)


def test_tiny_kinetic_loss_is_not_lost_in_global_thermal_energy():
    m = model()
    shape = m.u_layers.shape
    z = np.zeros(shape, np.float64)
    v0 = np.ones(shape, np.float64)
    v1 = v0 - 1e-12
    t = np.full(shape, 280.0)
    ps = m.surface_pressure.astype(np.float64)
    actual = budget_changes(m, ps, (z, v0, t, z), ps, (z, v1, t, z))
    mass = budgets(m, ps, (z, v0, t, z))["mass_kg"]
    expected = mass * 0.5 * (v1[0, 0, 0] - 1) * (v1[0, 0, 0] + 1)
    assert actual["energy_j"] < 0
    np.testing.assert_allclose(actual["energy_j"], expected, rtol=2e-7)
    assert actual["energy_j"] == actual["kinetic_energy_j"]
    assert actual["mass_kg"] == 0
    assert actual["axial_momentum_kgm2s"] == 0


def test_unchanged_state_has_exact_zero_budget_changes():
    m = model()
    fields = (m.u_layers, m.v_layers, m.T_layers, m.q_layers)
    result = budget_changes(m, m.surface_pressure, fields, m.surface_pressure, fields)
    assert all(x == 0 for x in result.values())


def test_local_energy_difference_includes_pressure_and_moisture_changes():
    m = model()
    f0 = tuple(x.astype(np.float64) for x in (m.u_layers, m.v_layers, m.T_layers, m.q_layers))
    f1 = tuple(x + delta for x, delta in zip(f0, (0.1, 0.2, 0.3, 0.0001)))
    p0 = m.surface_pressure.astype(np.float64)
    p1 = p0 + 100
    before, after = budgets(m, p0, f0), budgets(m, p1, f1)
    actual = budget_changes(m, p0, f0, p1, f1)
    for key in before:
        # Old totals use float32 area factors; the new differences compute
        # quadrature weights in float64, hence this metric-rounding tolerance.
        np.testing.assert_allclose(actual[key], after[key] - before[key], rtol=3e-6)


def test_candidate_bounds_and_cfl_are_reported_before_clipping():
    m = model()
    u = np.full_like(m.u_layers, 1e6)
    fields = (u, m.v_layers, m.T_layers, m.q_layers)
    report = stability_report(m, m.surface_pressure, fields,
                              np.zeros_like(m.surface_pressure), m.sigma_dot_interfaces,
                              pressure_candidates=(m.surface_pressure * 2,))
    assert report["finite"]
    assert report["advective_cfl_max"] > 1
    assert report["outside_bounds_fraction"]["u"] == 1
    assert report["outside_bounds_fraction"]["ps_earlier_stages"] == 1
    assert report["status"] == "bounds_exceeded"
    assert np.min(report["max_wind_ms_by_layer"]) >= 1e6


def test_nonfinite_candidate_is_reported():
    m = model()
    t = m.T_layers.copy()
    t[-1, 0, 0] = np.nan
    report = stability_report(m, m.surface_pressure,
        (m.u_layers, m.v_layers, t, m.q_layers),
        np.zeros_like(m.surface_pressure), m.sigma_dot_interfaces)
    assert report["status"] == "nonfinite"
