import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

from earthsim.config import load_config
from earthsim.data_loader import _profile
from earthsim.model import EarthModel


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
