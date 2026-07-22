import numpy as np

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
