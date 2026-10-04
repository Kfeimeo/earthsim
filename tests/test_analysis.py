import numpy as np
import pytest

from sim.analysis import analyze_point


def fields(include_ground_water=True):
    shape = (1, 2)
    values = {
        "u": np.zeros(shape),
        "v": np.zeros(shape),
        "temp": np.full(shape, 20.0),
        "cloud": np.zeros(shape),
        "precip": np.zeros(shape),
        "press": np.full(shape, 1013.0),
        "hum": np.full(shape, 8.0),
        "ice": np.zeros(shape),
        "sst": np.full(shape, 18.0),
        "uo": np.zeros(shape),
        "vo": np.zeros(shape),
    }
    if include_ground_water:
        values["ground_water"] = np.array([[42.34, 0.0]])
    return values


def test_land_analysis_includes_ground_water():
    result = analyze_point(
        fields(), np.array([0.0]), np.array([0.0, 180.0]),
        np.array([[1.0, 0.0]]), 0.0, 0.0,
    )

    assert result["surface"] == "陆地"
    assert result["ground_water"] == 42.3


def test_ocean_and_old_frames_omit_ground_water():
    ocean = analyze_point(
        fields(), np.array([0.0]), np.array([0.0, 180.0]),
        np.array([[1.0, 0.0]]), 0.0, 180.0,
    )
    old_land = analyze_point(
        fields(include_ground_water=False), np.array([0.0]),
        np.array([0.0, 180.0]), np.array([[1.0, 0.0]]), 0.0, 0.0,
    )

    assert "ground_water" not in ocean
    assert "ground_water" not in old_land


def test_column_profile_lists_levels_top_down_with_selected_layer():
    from sim.analysis import column_profile

    column = {
        "levels_m": np.array([80.0, 1500.0, 12000.0]),
        "pressure_pa": np.array([99300.0, 85000.0, 20000.0]),
        "temp_k": np.array([290.0, 280.0, 215.0]),
        "humidity": np.array([0.010, 0.005, 0.0001]),
        "u": np.array([3.0, 10.0, 30.0]),
        "v": np.array([4.0, 0.0, 0.0]),
        "w": np.array([0.0, 0.02, -0.01]),
        "omega_pa_s": np.array([0.0, -0.2, 0.1]),
    }
    profile = column_profile(column, selected_layer=1)

    assert profile["count"] == 3
    assert [e["index"] for e in profile["levels"]] == [2, 1, 0]
    top, mid, low = profile["levels"]
    assert top["pressure"] == 200.0 and low["pressure"] == 993.0
    assert low["wind_speed"] == 5.0 and low["wind_dir"] == 37
    assert low["temp"] == pytest.approx(16.85, abs=0.06)
    assert 0 <= low["rh"] <= 150 and mid["w"] == 2.0
    assert profile["selected_layer"] == 1
    assert profile["selected"]["wind_speed"] == 10.0


def test_column_profile_works_with_wind_only_playback_frames():
    from sim.analysis import column_profile

    profile = column_profile({"u": np.array([1.0, 2.0]), "v": np.zeros(2)})
    assert profile["count"] == 2
    assert "temp" not in profile["levels"][0]
    assert profile["levels"][1]["wind_speed"] == 1.0


def test_analyze_point_attaches_profile_from_model_column():
    from sim.config import load_config
    from sim.model import EarthModel

    cfg = load_config()
    cfg["backend"] = "cpu"
    cfg["grid"].update(nlat=8, nlon=16, topo_file="")
    cfg["grid"]["topo_files"] = []
    cfg["data"]["init_mode"] = "ideal"
    cfg["physics"]["vertical"]["coordinate"] = "hybrid"
    cfg["physics"]["vertical"]["hybrid"]["levels"] = 16
    cfg["physics"]["dynamics"].update(top_pressure_pa=300.0,
                                      min_surface_pressure_pa=45000.0)
    model = EarthModel(cfg)
    fields = model.fields_cpu()
    result = analyze_point(fields, model.lats, model.lons,
                           np.asarray(model.land), 10.0, 20.0,
                           column=model.atmosphere_column_cpu(3, 4),
                           selected_layer=5)

    levels = result["profile"]["levels"]
    assert len(levels) == 16
    pressures = [e["pressure"] for e in levels]
    assert pressures == sorted(pressures)          # top first
    assert pressures[0] == pytest.approx(
        model.reference_pressure_layers_pa[-1] / 100.0, rel=0.2)
    assert all("rh" in e and "temp" in e for e in levels)
    assert result["profile"]["selected"]["index"] == 5
