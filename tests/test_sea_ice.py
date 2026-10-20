import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

from sim.config import load_config
from sim.model import EarthModel


def _config(nlat=8, nlon=16, **overrides):
    cfg = load_config()
    cfg["backend"] = "cpu"
    cfg["grid"].update(nlat=nlat, nlon=nlon, topo_file="")
    cfg["time"]["dt"] = 60.0
    cfg["time"]["start"] = "2026-05-01T00:00:00"
    cfg["data"].update(init_mode="ideal", atmosphere_file="",
                       surface_file="", ocean_file="")
    # The shipped config.yaml freezes at 271.2 K; DEFAULTS still carry an
    # older 268 K floor, which would let the water under ice cool below the
    # temperature the ideal state starts from.
    cfg["physics"]["bounds"]["ocean_freezing_min_temp_k"] = 271.2
    for key, value in overrides.items():
        cfg["physics"][key] = value
    return cfg


def _real_files(td, *, with_ice=True, polar_skin=255.0):
    """One ERA5-style pressure-level file plus a single-level surface file."""
    lat = np.arange(-90.0, 90.1, 10.0)
    lon = np.array([0.0, 120.0, 240.0])
    level = np.array([1000.0, 500.0])
    shape = (1, len(level), len(lat), len(lon))
    temp = np.empty(shape, np.float32)
    temp[0, 0] = 280.0
    temp[0, 1] = 250.0
    pressure = xr.Dataset(
        {"t": (("time", "level", "latitude", "longitude"), temp),
         "q": (("time", "level", "latitude", "longitude"),
               np.full(shape, 0.003, np.float32))},
        coords={"time": ["2026-05-01T00:00:00"], "level": level,
                "latitude": lat, "longitude": lon},
    )
    pressure["t"].attrs["units"] = "K"
    pressure["q"].attrs["units"] = "kg kg-1"
    pressure["level"].attrs["units"] = "hPa"

    polar = (np.abs(lat) >= 60.0)[:, None] * np.ones((1, len(lon)), bool)
    skt = np.where(polar, polar_skin, 290.0).astype(np.float32)[None]
    sst = np.where(polar, 271.46, 290.0).astype(np.float32)[None]
    data = {
        "skt": (("time", "latitude", "longitude"), skt),
        "sst": (("time", "latitude", "longitude"), sst),
    }
    if with_ice:
        data["siconc"] = (("time", "latitude", "longitude"),
                          np.where(polar, 0.9, 0.0).astype(np.float32)[None])
    surface = xr.Dataset(
        data, coords={"time": ["2026-05-01T00:00:00"],
                      "latitude": lat, "longitude": lon})
    surface["skt"].attrs["units"] = "K"
    surface["sst"].attrs["units"] = "K"
    pressure_path = Path(td) / "pressure.nc"
    surface_path = Path(td) / "surface.nc"
    pressure.to_netcdf(pressure_path, engine="scipy")
    surface.to_netcdf(surface_path, engine="scipy")
    return str(pressure_path), str(surface_path)


def _polar_ocean(model):
    lat = np.asarray(model.ops.lat)[:, 0]
    polar = (np.abs(lat) >= np.radians(60.0))[:, None]
    return polar & (np.asarray(model.ocean) > 0.5)


def _low_latitude(model):
    """Rows the bilinear remap cannot blend with the polar input rows."""
    lat = np.asarray(model.ops.lat)[:, 0]
    return (np.abs(lat) < np.radians(50.0))[:, None] & np.ones(
        (1, model.nlon), bool)


def test_real_initialization_keeps_sea_ice_and_skin_temperature():
    with tempfile.TemporaryDirectory() as td:
        pressure_path, surface_path = _real_files(td)
        cfg = _config()
        cfg["data"].update(init_mode="real", atmosphere_file=pressure_path,
                           surface_file=surface_path)
        model = EarthModel(cfg)
        assert model.initialization_source == "real"
        polar = _polar_ocean(model)
        assert polar.any()
        ice = np.asarray(model.ice)
        # The analysed concentration survives _diag_surface(): the old
        # temperature diagnosis gave zero ice for an SST of 271.46 K.
        np.testing.assert_allclose(ice[polar], 0.9, atol=1e-5)
        assert ice[_low_latitude(model)].max() == 0.0
        # Water under the ice keeps the SST; the ice skin is the analysed
        # skin temperature, which is what the atmosphere must see.
        np.testing.assert_allclose(np.asarray(model.Ts)[polar], 271.46, atol=1e-3)
        np.testing.assert_allclose(np.asarray(model.ice_temp)[polar], 255.0, atol=1e-3)
        open_water = (np.asarray(model.ocean) > 0.5) & (ice == 0.0)
        np.testing.assert_allclose(
            np.asarray(model.ice_temp)[open_water],
            np.asarray(model.Ts)[open_water])


def test_real_initialization_infers_ice_from_sub_freezing_skin():
    with tempfile.TemporaryDirectory() as td:
        pressure_path, surface_path = _real_files(td, with_ice=False)
        cfg = _config()
        cfg["data"].update(init_mode="real", atmosphere_file=pressure_path,
                           surface_file=surface_path)
        model = EarthModel(cfg)
        polar = _polar_ocean(model)
        ice = np.asarray(model.ice)
        # A 255 K skin over 271 K water can only be ice.
        np.testing.assert_allclose(ice[polar], 1.0, atol=1e-5)
        assert ice[_low_latitude(model)].max() == 0.0

        cfg = _config()
        cfg["physics"]["sea_ice"]["infer_from_skin_temp"] = False
        cfg["data"].update(init_mode="real", atmosphere_file=pressure_path,
                           surface_file=surface_path)
        model = EarthModel(cfg)
        # Without a concentration field and without inference the real
        # state keeps the ice of the idealized equilibrium it replaced.
        np.testing.assert_allclose(
            np.asarray(model.ice), np.asarray(EarthModel(_config()).ice))


def test_diagnostic_mode_reproduces_temperature_threshold_ice():
    cfg = _config()
    cfg["physics"]["sea_ice"]["prognostic"] = False
    model = EarthModel(cfg)
    assert not model.sea_ice_prognostic
    bounds = model.cfg.physics.bounds
    expected = np.asarray(model.ocean) * np.clip(
        (float(bounds.sea_ice_threshold_k) - np.asarray(model.Ts))
        / float(bounds.sea_ice_transition_k), 0.0, 1.0)
    np.testing.assert_allclose(np.asarray(model.ice), expected, atol=1e-6)
    model.step(1)
    expected = np.asarray(model.ocean) * np.clip(
        (float(bounds.sea_ice_threshold_k) - np.asarray(model.Ts))
        / float(bounds.sea_ice_transition_k), 0.0, 1.0)
    np.testing.assert_allclose(np.asarray(model.ice), expected, atol=1e-6)


def test_ideal_initialization_freezes_cold_ocean():
    cfg = _config()
    model = EarthModel(cfg)
    ice = np.asarray(model.ice)
    ocean = np.asarray(model.ocean) > 0.5
    lat = np.abs(np.asarray(model.ops.lat)[:, 0])
    cold_rows = lat >= np.radians(70.0)
    assert ice[cold_rows][ocean[cold_rows]].min() == 1.0
    assert ice[~cold_rows][ocean[~cold_rows]].max() == 0.0
    # The ice skin sits below the water temperature, which is at the
    # freezing point.
    assert (np.asarray(model.ice_temp)[ice > 0.5]
            < np.asarray(model.Ts)[ice > 0.5]).all()


def _run_dark(model, steps):
    """Step with the sun off so only longwave and turbulent fluxes act."""
    # cfg.physics returns a copy; the stored mapping is cfg["physics"].
    model.cfg["physics"]["S0"] = 0.0
    model.step(steps)


def test_ice_skin_cools_while_water_stays_at_freezing():
    cfg = _config(radiation=True, diurnal_cycle=False)
    model = EarthModel(cfg)
    ocean = np.asarray(model.ocean) > 0.5
    ice = np.asarray(model.ice) > 0.5
    cells = ocean & ice
    assert cells.any()
    freezing = float(model.cfg.physics.bounds.ocean_freezing_min_temp_k)
    _run_dark(model, 60)
    Ts = np.asarray(model.Ts)
    skin = np.asarray(model.ice_temp)
    # Water under ice cannot drop below the freezing point and is not an
    # energy source: it stays exactly there with no sun.
    np.testing.assert_allclose(Ts[cells], freezing, atol=1e-3)
    # The ice skin radiates and cools below the water temperature.
    assert (skin[cells] < freezing - 1.0).all()
    assert (np.asarray(model.ice)[cells] >= 0.5).all()


def test_open_water_cooling_below_freezing_forms_ice():
    cfg = _config(radiation=True, diurnal_cycle=False)
    model = EarthModel(cfg)
    xp = model.xp
    ocean = np.asarray(model.ocean) > 0.5
    freezing = float(model.cfg.physics.bounds.ocean_freezing_min_temp_k)
    # Start from ice-free water just above freezing under very cold air.
    model.ice = xp.zeros_like(model.Ts)
    model.Ts = xp.where(model.ocean > 0.5, xp.float32(freezing + 0.001),
                        model.Ts).astype(xp.float32)
    model.ice_temp = model.Ts.copy()
    model.T_layers = xp.full_like(model.T_layers, 240.0)
    _run_dark(model, 120)
    ice = np.asarray(model.ice)
    assert (ice[ocean] > 0.0).all()
    assert (ice[~ocean] == 0.0).all()
    np.testing.assert_allclose(np.asarray(model.Ts)[ocean], freezing, atol=1e-3)
    # New ice forms at the freezing point and then cools as a skin.
    assert (np.asarray(model.ice_temp)[ocean] <= freezing + 1e-3).all()


def test_warm_air_melts_ice_from_the_top():
    cfg = _config(radiation=True, diurnal_cycle=False)
    model = EarthModel(cfg)
    xp = model.xp
    ocean = np.asarray(model.ocean) > 0.5
    melt = float(model.cfg.physics.sea_ice.melt_temp_k)
    freezing = float(model.cfg.physics.bounds.ocean_freezing_min_temp_k)
    model.ice = (model.ocean * 1.0).astype(xp.float32)
    # Water at the freezing point, so only melt from the top can act.
    model.Ts = xp.where(model.ocean > 0.5, xp.float32(freezing),
                        model.Ts).astype(xp.float32)
    model.ice_temp = xp.where(model.ocean > 0.5, xp.float32(melt - 0.01),
                              model.Ts).astype(xp.float32)
    model.T_layers = xp.full_like(model.T_layers, 300.0)
    model.q_layers = xp.full_like(model.q_layers, 0.01)
    ice0 = np.asarray(model.ice).copy()
    _run_dark(model, 240)
    ice = np.asarray(model.ice)
    assert (ice[ocean] < ice0[ocean]).all()
    assert (ice[ocean] > 0.0).all()
    # The skin is capped at the melting point; the excess melts ice.
    assert (np.asarray(model.ice_temp)[ocean] <= melt + 1e-3).all()


def test_no_ice_tile_matches_open_water_fluxes():
    """Ice-free cells see exactly the open-water surface budget."""
    cfg = _config(radiation=True, diurnal_cycle=False)
    model = EarthModel(cfg)
    xp = model.xp
    model.ice = xp.zeros_like(model.Ts)
    model.ice_temp = model.Ts.copy()
    before = np.asarray(model.Ts).copy()
    model.step(1)
    after = np.asarray(model.Ts)
    assert np.isfinite(after).all()
    assert np.abs(after - before).max() < 5.0
    ice = np.asarray(model.ice)
    # Water already at the freezing point may start to freeze at night;
    # everything else stays ice free and its skin follows the water.
    assert ice.max() < 1e-2
    free = ice == 0.0
    np.testing.assert_allclose(
        np.asarray(model.ice_temp)[free], after[free], atol=1e-3)


def test_frame_round_trip_keeps_ice_state():
    cfg = _config()
    model = EarthModel(cfg)
    model.step(2)
    fields = model.fields_cpu(include_layers=True)
    assert "sea_ice" in fields and "ice_temp" in fields
    restored = EarthModel(cfg)
    restored.load_frame_state(fields, model.t)
    np.testing.assert_allclose(np.asarray(restored.ice), np.asarray(model.ice))
    np.testing.assert_allclose(
        np.asarray(restored.ice_temp), np.asarray(model.ice_temp), atol=1e-3)
    restored.step(1)
    model.step(1)
    np.testing.assert_allclose(
        np.asarray(restored.ice), np.asarray(model.ice), atol=1e-5)
