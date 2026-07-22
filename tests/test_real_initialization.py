import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

from sim.config import load_config
from sim.data_loader import (
    _profile,
    _saturation_specific_humidity,
    load_real_initialization,
)
from sim.model import EarthModel


def test_profile_copies_read_only_pressure_levels():
    lat = np.array([-60.0, 0.0, 60.0])
    lon = np.array([0.0, 120.0, 240.0])
    level = np.array([1000.0, 700.0, 300.0])
    level.setflags(write=False)
    values = np.ones((len(level), len(lat), len(lon)), np.float32)
    ds = xr.Dataset(
        {"t": (("level", "latitude", "longitude"), values)},
        coords={"level": level, "latitude": lat, "longitude": lon},
    )
    ds["level"].attrs["units"] = "hPa"

    profile = _profile(ds["t"], ds, "", lat, lon, [100.0, 1000.0])

    assert profile.shape == (2, len(lat), len(lon))


def test_real_pressure_level_initialization():
    lat = np.array([-60.0, 0.0, 60.0])
    lon = np.array([0.0, 120.0, 240.0])
    level = np.array([1000.0, 700.0, 300.0, 100.0])
    shape = (1, len(level), len(lat), len(lon))
    temp = np.empty(shape, np.float32)
    for k, pressure in enumerate(level):
        temp[0, k] = 285.0 - 0.02 * (1000.0 - pressure)
    ds = xr.Dataset(
        {
            "t": (("time", "level", "latitude", "longitude"), temp),
            "q": (("time", "level", "latitude", "longitude"),
                  np.full(shape, 0.008, np.float32)),
            "u": (("time", "level", "latitude", "longitude"),
                  np.full(shape, 12.0, np.float32)),
            "v": (("time", "level", "latitude", "longitude"),
                  np.full(shape, -4.0, np.float32)),
            "msl": (("time", "latitude", "longitude"),
                    np.full((1, len(lat), len(lon)), 101500.0, np.float32)),
            "sst": (("time", "latitude", "longitude"),
                    np.full((1, len(lat), len(lon)), 299.0, np.float32)),
        },
        coords={"time": ["2026-05-01T00:00:00"], "level": level,
                "latitude": lat, "longitude": lon},
    )
    ds["t"].attrs["units"] = "K"
    ds["q"].attrs["units"] = "kg kg-1"
    ds["level"].attrs["units"] = "hPa"
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "real.nc"
        ds.to_netcdf(path, engine="scipy")
        cfg = load_config()
        cfg["backend"] = "cpu"
        cfg["grid"].update(nlat=8, nlon=16, topo_file="")
        cfg["time"]["dt"] = 60.0
        cfg["data"].update(init_mode="real", atmosphere_file=str(path),
                            surface_file="", ocean_file="")
        model = EarthModel(cfg)
        assert model.initialization_source == "real"
        assert model.T_layers.shape == (5, 8, 16)
        np.testing.assert_allclose(np.asarray(model.u_layers)[0], 12.0, atol=1e-5)
        np.testing.assert_allclose(np.asarray(model.v_layers)[-1], -4.0, atol=1e-5)
        ocean = np.asarray(model.ocean) > 0.5
        np.testing.assert_allclose(np.asarray(model.Ts)[ocean], 299.0)
        assert float(np.asarray(model.pressure_hpa()).mean()) > 1013.0


def test_separate_files_keep_independent_snapshot_times():
    lat = np.array([-60.0, 0.0, 60.0])
    lon = np.array([0.0, 120.0, 240.0])
    level = np.array([1000.0, 700.0])
    horizontal_pattern = lat[:, None] / 10.0 + lon[None, :] / 120.0
    temp = np.stack([
        285.0 + horizontal_pattern,
        270.0 + horizontal_pattern,
    ])[None].astype(np.float32)
    sst = (290.0 + horizontal_pattern)[None].astype(np.float32)

    pressure_ds = xr.Dataset(
        {"t": (("valid_time", "pressure_level", "latitude", "longitude"),
               temp)},
        coords={"valid_time": ["2026-06-01T00:00:00"],
                "pressure_level": level, "latitude": lat, "longitude": lon},
    )
    pressure_ds["t"].attrs["units"] = "K"
    pressure_ds["pressure_level"].attrs["units"] = "hPa"
    surface_ds = xr.Dataset(
        {
            "t2m": (("valid_time", "latitude", "longitude"),
                    (288.0 + horizontal_pattern)[None].astype(np.float32)),
            "skt": (("valid_time", "latitude", "longitude"), sst),
            "sst": (("valid_time", "latitude", "longitude"), sst),
        },
        coords={"valid_time": ["2026-05-01T00:00:00"],
                "latitude": lat, "longitude": lon},
    )
    for name in ("t2m", "skt", "sst"):
        surface_ds[name].attrs["units"] = "K"

    with tempfile.TemporaryDirectory() as td:
        pressure_path = Path(td) / "pressure.nc"
        surface_path = Path(td) / "surface.nc"
        pressure_ds.to_netcdf(pressure_path, engine="scipy")
        surface_ds.to_netcdf(surface_path, engine="scipy")
        cfg = load_config()
        cfg["time"]["start"] = "2026-05-01T00:00:00"
        cfg["data"].update(
            atmosphere_file=str(pressure_path),
            surface_file=str(surface_path),
            ocean_file="",
        )

        fields = load_real_initialization(
            cfg, lat, lon, cfg.physics.vertical.levels_m)

    assert np.isfinite(fields["temp"]).all()
    assert np.ptp(fields["temp"][0]) > 1.0
    assert np.isfinite(fields["surface"]).all()
    assert np.ptp(fields["surface"]) > 1.0
    assert np.isfinite(fields["ocean_surface"]).all()
    assert np.ptp(fields["ocean_surface"]) > 1.0


def test_pressure_profile_uses_configured_reference_and_scale_height():
    lat = np.array([-30.0, 30.0])
    lon = np.array([0.0, 180.0])
    pressure = np.array([1000.0, 500.0])
    temp = np.stack([
        np.full((2, 2), 300.0, np.float32),
        np.full((2, 2), 250.0, np.float32),
    ])[None]
    ds = xr.Dataset(
        {"t": (("time", "level", "latitude", "longitude"), temp)},
        coords={"time": ["2026-05-01T00:00:00"], "level": pressure,
                "latitude": lat, "longitude": lon},
    )
    ds["t"].attrs["units"] = "K"
    ds["level"].attrs["units"] = "hPa"

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "pressure.nc"
        ds.to_netcdf(path, engine="scipy")
        cfg = load_config()
        cfg["data"].update(
            atmosphere_file=str(path), surface_file="", ocean_file="")
        cfg["physics"]["dynamics"]["reference_surface_pressure_pa"] = 80000.0
        cfg["physics"]["vertical"]["scale_height"] = 10000.0
        cfg["physics"]["vertical"]["scale_height_min_m"] = 1.0
        cfg["physics"]["initial_conditions"]["real_default_mslp_hpa"] = 987.0

        levels_m = [0.0, 1000.0]
        fields = load_real_initialization(cfg, lat, lon, levels_m)

    expected_pressure = 80000.0 * np.exp(-np.asarray(levels_m) / 10000.0)
    expected_temp = np.interp(
        expected_pressure, [50000.0, 100000.0], [250.0, 300.0])
    np.testing.assert_allclose(
        fields["temp"][:, 0, 0], expected_temp, rtol=0, atol=1e-5)
    np.testing.assert_allclose(fields["mslp_hpa"], 987.0)


def test_surface_only_fallbacks_use_configured_lapse_and_humidity():
    lat = np.array([-30.0, 30.0])
    lon = np.array([0.0, 180.0])
    t2m = np.full((1, 2, 2), 300.0, np.float32)
    ds = xr.Dataset(
        {"t2m": (("time", "latitude", "longitude"), t2m)},
        coords={"time": ["2026-05-01T00:00:00"],
                "latitude": lat, "longitude": lon},
    )
    ds["t2m"].attrs["units"] = "K"

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "surface.nc"
        ds.to_netcdf(path, engine="scipy")
        cfg = load_config()
        cfg["data"].update(
            atmosphere_file=str(path), surface_file="", ocean_file="")
        cfg["physics"]["vertical"]["lapse_rate"] = 0.01
        cfg["physics"]["vertical"]["scale_height"] = 5000.0
        cfg["physics"]["vertical"]["scale_height_min_m"] = 1.0
        cfg["physics"]["dynamics"]["reference_surface_pressure_pa"] = 90000.0
        cfg["physics"]["initial_conditions"]["humidity_decay_height_m"] = 1000.0
        cfg["physics"]["init_surface_rh"] = 0.9
        cfg["physics"]["init_upper_rh"] = 0.1

        levels_m = np.array([100.0, 1100.0])
        fields = load_real_initialization(cfg, lat, lon, levels_m)

    np.testing.assert_allclose(fields["temp"][0], 300.0)
    np.testing.assert_allclose(fields["temp"][1], 290.0)
    pressure = 90000.0 * np.exp(-levels_m / 5000.0)
    rh = 0.1 + 0.8 * np.exp(-levels_m / 1000.0)
    expected_q = rh * _saturation_specific_humidity(
        fields["temp"][:, 0, 0], pressure)
    np.testing.assert_allclose(
        fields["q"][:, 0, 0], expected_q, rtol=1e-6)
