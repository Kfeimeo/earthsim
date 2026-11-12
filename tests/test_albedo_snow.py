"""Cloud/surface albedo, zenith dependence, sea-ice albedo and the snow pack."""
import numpy as np
import pytest

from sim.config import load_config
from sim.model import EarthModel
from sim.physics import (C_LAND, daylight_mean_cos_zenith, ocean_direct_albedo,
                         snow_direct_albedo)


def small_model(**physics):
    cfg = load_config()
    cfg["backend"] = "cpu"
    cfg["grid"].update(nlat=12, nlon=24, topo_file="", topo_files=[])
    cfg["data"]["init_mode"] = "ideal"
    cfg["time"]["dt"] = 60.0
    cfg["physics"].update(physics)
    return EarthModel(cfg)


def test_ocean_direct_albedo_follows_zenith_angle():
    mu = np.array([1.0, 0.5, 0.1, 0.0], np.float32)
    albedo = ocean_direct_albedo(np, mu)
    assert albedo[0] == pytest.approx(0.0244, abs=1e-3)
    assert albedo[1] == pytest.approx(0.070, abs=2e-3)
    assert 0.25 < albedo[2] < 0.35
    assert np.all(np.diff(albedo) > 0)


def test_snow_brightens_only_at_low_sun():
    mu = np.array([1.0, 0.5, 0.2, 0.0], np.float32)
    albedo = snow_direct_albedo(np, 0.8, mu, 0.4)
    np.testing.assert_allclose(albedo[:2], 0.8)
    assert albedo[2] > 0.8
    assert albedo[3] == pytest.approx(0.8 + 0.4 * 0.2)


def test_daylight_mean_cos_zenith_at_equinox_equator():
    lat = np.radians(np.array([[0.0], [89.0]], np.float32))
    mu = daylight_mean_cos_zenith(np, lat, 0.0)
    assert float(mu[0, 0]) == pytest.approx(2 / np.pi, rel=1e-5)
    assert 0 <= float(mu[1, 0]) < 0.05


def test_clouds_shade_bright_surfaces_less_than_dark_ones():
    m = small_model(ocean_albedo_zenith=False, snow_zenith_factor=0.0)
    shape = m.Ts.shape
    a_c = float(m.cfg.physics.alb_cloud)

    m.cloud = np.zeros(shape, np.float32)
    clear = m._cloudy_albedo(np.float32(0.62), np.float32(0.62))
    np.testing.assert_allclose(clear, 0.62)  # clear sky = surface albedo

    m.cloud = np.ones(shape, np.float32)
    for surface in (0.07, 0.62):
        overcast = m._cloudy_albedo(np.float32(surface), np.float32(surface))
        expected = 1 - (1 - a_c) * (1 - surface) / (1 - a_c * surface)
        np.testing.assert_allclose(overcast, expected, rtol=1e-6)
    ice_kept = (1 - expected) / (1 - 0.62)
    water_kept = (1 - a_c) * (1 - 0.07) / (1 - a_c * 0.07) / 0.93
    # Linear addition left ice only 0.15/0.38 = 39 % of its clear-sky sunlight.
    assert ice_kept > 0.75 > water_kept > 0.55


def test_sea_ice_albedo_drops_as_skin_approaches_melting():
    m = small_model()
    melting = float(np.float32(m.cfg.physics.sea_ice.freezing_temp_k))
    m.ice_skin_temp = np.full_like(m.Ts, melting - 5.0)
    np.testing.assert_allclose(m._sea_ice_albedo(), 0.75)
    m.ice_skin_temp = np.full_like(m.Ts, melting)
    np.testing.assert_allclose(m._sea_ice_albedo(), 0.50)
    m.ice_skin_temp = np.full_like(m.Ts, melting - 0.5)
    np.testing.assert_allclose(m._sea_ice_albedo(), 0.625, rtol=1e-5)


def _all_land(m):
    m.land[...] = 1.0
    m.ocean[...] = 0.0
    m.ice_sheet = np.zeros_like(m.Ts)
    m.snow_mass = np.zeros_like(m.Ts)
    m._snowmelt_mm = np.zeros_like(m.Ts)
    m.ground_water[...] = 10.0


def test_cold_precipitation_builds_snow_and_warm_falls_as_rain():
    m = small_model(ground_runoff_tau=1.0e30)
    _all_land(m)
    zero = np.zeros_like(m.Ts)
    precip = np.full_like(m.Ts, 2.0)

    m.Ta[...] = 263.0
    m._update_ground_water(zero, precip)
    np.testing.assert_allclose(m.snow_mass, 2.0)
    np.testing.assert_allclose(m.ground_water, 10.0)

    m.Ta[...] = 290.0
    m._update_ground_water(zero, precip)
    np.testing.assert_allclose(m.snow_mass, 2.0)
    np.testing.assert_allclose(m.ground_water, 12.0)


def test_snow_melt_conserves_energy_and_water_and_persists_at_night():
    m = small_model(ground_runoff_tau=1.0e30)
    _all_land(m)
    latent = float(m.cfg.physics.sea_ice.latent_heat_j_kg)
    melt_t = float(m.cfg.physics.snow.melt_temp_k)
    m.snow_mass[...] = 20.0

    # Night: far below melting, the pack stays (no diurnal flicker).
    m.Ts[...] = 255.0
    m._diag_surface()
    np.testing.assert_allclose(m.snow_mass, 20.0)
    np.testing.assert_allclose(m.snow, 1.0)

    # Day: 1 K above melting melts C_LAND * 1 K / Lf (7.5 mm) of snow.
    m.Ts[...] = melt_t + 1.0
    before = C_LAND * m.Ts.astype(np.float64) - latent * m.snow_mass
    m._diag_surface()
    after = C_LAND * m.Ts.astype(np.float64) - latent * m.snow_mass
    np.testing.assert_allclose(after, before, rtol=1e-6)
    melted = C_LAND / latent
    np.testing.assert_allclose(m.snow_mass, 20.0 - melted, rtol=1e-5)
    np.testing.assert_allclose(m.Ts, melt_t, atol=1e-4)

    m.Ta[...] = 290.0
    m._update_ground_water(np.zeros_like(m.Ts), np.zeros_like(m.Ts))
    np.testing.assert_allclose(m.ground_water, 10.0 + melted, rtol=1e-5)


def test_ice_sheet_is_always_snow_covered_and_capped_at_melting():
    m = small_model()
    m.land[...] = 1.0
    m.ocean[...] = 0.0
    m.ice_sheet = m._ice_sheet_mask()
    south = m.lats <= -60
    assert np.all(m.ice_sheet[south] == 1)
    assert np.all(m.ice_sheet[(m.lats > -60) & (m.lats < 50)] == 0)

    m.snow_mass = np.zeros_like(m.Ts)
    m.Ts[...] = 280.0
    m._diag_surface()
    melt_t = float(m.cfg.physics.snow.melt_temp_k)
    np.testing.assert_allclose(m.Ts[south], melt_t)
    np.testing.assert_allclose(m.snow[south], 1.0)
    assert np.all(m.Ts[(m.lats > -60) & (m.lats < 50)] == 280.0)


def test_frame_roundtrip_keeps_snow_pack():
    m = small_model()
    m.snow_mass = (m.land * 37.0).astype(np.float32)
    fields = m.fields_cpu(include_layers=True)
    m.snow_mass = np.zeros_like(m.snow_mass)
    m.load_frame_state(fields, m.t)
    land = m.land > 0.5
    melted = fields["sst"][land] + 273.15 > float(m.cfg.physics.snow.melt_temp_k)
    np.testing.assert_allclose(m.snow_mass[land][~melted], 37.0)
    m.step(2)
    m.check_health()


def test_legacy_constant_sea_ice_albedo_is_accepted(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("physics:\n  alb_ice: 0.6\n", encoding="utf-8")
    cfg = load_config(path)
    assert cfg.physics.alb_ice_cold == 0.6
    assert cfg.physics.alb_ice_melt == 0.6
