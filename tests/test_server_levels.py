"""Level-dependent display scalars, global statistics and point analysis."""
import json
import struct

import numpy as np
import pytest

from server import (Hub, LAYER_RANGES, create_app, global_field_stats,
                    level_display_ranges)
from sim.config import load_config


def _small_cfg():
    cfg = load_config()
    cfg["backend"] = "cpu"
    cfg["grid"].update(nlat=12, nlon=24, topo_file="")
    cfg["grid"]["topo_files"] = []
    cfg["time"]["dt"] = 60.0
    cfg["time"]["spinup_days"] = 0
    cfg["data"]["init_mode"] = "ideal"
    cfg["server"]["record_enabled"] = False
    return cfg


def _decode(pkt):
    n = struct.unpack("<I", pkt[:4])[0]
    return json.loads(pkt[4:4 + n])


def test_level_display_ranges_follow_the_standard_atmosphere():
    surface = level_display_ranges(0.0)
    assert surface["press"] == LAYER_RANGES["press"]
    assert surface["temp"] == LAYER_RANGES["temp"]
    mid = level_display_ranges(5000.0)
    assert mid["press"][0] < 540 < mid["press"][1]
    assert mid["temp"][0] < -32.5 < mid["temp"][1]
    assert 0 < mid["hum"][1] < LAYER_RANGES["hum"][1]
    # A known reference pressure overrides the nominal height.
    by_pressure = level_display_ranges(100.0, reference_pressure_hpa=500.0)
    assert by_pressure["press"][0] < 500 < by_pressure["press"][1]


def test_global_field_stats_are_area_weighted():
    lats = np.array([-60.0, 0.0, 60.0])
    temp = np.array([[0.0, 0.0], [10.0, 10.0], [0.0, 0.0]])
    zeros = np.zeros_like(temp)
    s = global_field_stats(lats, temp, zeros + 1000, zeros + 3, zeros + 4)
    assert s["temp"]["min"] == 0 and s["temp"]["max"] == 10
    assert s["temp"]["mean"] == pytest.approx(10 / (1 + 2 * np.cos(np.radians(60))), abs=0.01)
    assert s["wind"] == {"min": 5.0, "max": 5.0, "mean": 5.0}


def test_packet_scalars_follow_the_selected_level():
    hub = Hub(_small_cfg())
    hub.wind_layer_index = 0
    m0 = _decode(hub.packet())
    hub.wind_layer_index = 3
    m3 = _decode(hub.packet())
    r0 = {L["name"]: (L["min"], L["max"]) for L in m0["layers"]}
    r3 = {L["name"]: (L["min"], L["max"]) for L in m3["layers"]}
    assert r0["press"] == tuple(LAYER_RANGES["press"])
    assert r3["press"][1] < r0["press"][0]          # a higher level is a lower pressure
    assert r3["sst"] == r0["sst"]                    # surface-only layers are unchanged
    assert m3["stats"]["layer"] == 3
    assert r3["press"][0] <= m3["stats"]["press"]["min"] <= m3["stats"]["press"]["max"] <= r3["press"][1]
    assert m3["stats"]["temp"]["max"] < m0["stats"]["temp"]["max"]
    assert m3["stats"]["wind"]["min"] <= m3["stats"]["wind"]["mean"] <= m3["stats"]["wind"]["max"]
    assert m3["atmosphere_layer_index"] == 3
    assert hub.level_fields["index"] == 3


def test_analyze_reports_the_selected_level():
    pytest.importorskip("fastapi.testclient")
    from fastapi.testclient import TestClient
    app = create_app(_small_cfg())
    hub = app.state.hub
    client = TestClient(app)
    hub.wind_layer_index = 2
    hub.packet()
    d = client.get("/api/analyze?lat=10&lon=120&layer=2").json()
    assert d["level"]["index"] == 2
    assert d["level"]["height_m"] == round(float(hub.model.levels_m[2]))
    assert d["level"]["temp"] < d["temp"]
    assert "level" not in client.get("/api/analyze?lat=10&lon=120&layer=0").json()
    assert "level" not in client.get("/api/analyze?lat=10&lon=120").json()
