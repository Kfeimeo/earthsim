"""Live mode: start the simulation from a saved frame."""
import asyncio
import json
import os

import numpy as np
import pytest

from server import (Hub, LiveRecorder, frame_path, list_frame_runs,
                    resolve_frame_run)
from sim.backend import to_cpu
from sim.config import load_config
from sim.model import EarthModel


def _small_cfg(tmp_path):
    cfg = load_config()
    cfg["backend"] = "cpu"
    cfg["grid"].update(nlat=12, nlon=24, topo_file="")
    cfg["grid"]["topo_files"] = []
    cfg["time"]["dt"] = 60.0
    cfg["time"]["spinup_days"] = 0
    cfg["data"]["init_mode"] = "ideal"
    cfg["server"]["record_enabled"] = False
    cfg["server"]["record_dir"] = str(tmp_path / "recordings")
    cfg["precompute"]["out_dir"] = str(tmp_path / "out" / "run1")
    return cfg


def _record_run(cfg, steps=(3, 4)):
    """Record a short live run; return (run_dir, model after the last frame)."""
    model = EarthModel(cfg)
    recorder = LiveRecorder(cfg)
    recorder.enabled = True
    recorder.start(model)
    for n in steps:
        model.step(n)
        recorder.save(model, force=True)
    return recorder.run_dir, model


class _FakeWs:
    def __init__(self):
        self.texts, self.frames = [], []

    async def send_text(self, text):
        self.texts.append(json.loads(text))

    async def send_bytes(self, data):
        self.frames.append(data)


def test_saved_runs_are_listed_and_resolved_only_under_known_roots(tmp_path):
    cfg = _small_cfg(tmp_path)
    run_dir, _ = _record_run(cfg)
    os.makedirs(tmp_path / "out" / "not_a_run")  # no manifest: ignored

    hub = Hub(cfg)
    runs = list_frame_runs(hub.run_roots)
    assert len(runs) == 1
    assert runs[0]["nframes"] == 3
    assert runs[0]["grid"] == [12, 24]
    assert os.path.normpath(resolve_frame_run(hub.run_roots, runs[0]["id"])) \
        == os.path.normpath(run_dir)
    with pytest.raises(ValueError):
        resolve_frame_run(hub.run_roots, "../../etc")


def test_hub_starts_from_init_frame_and_reset_returns_to_it(tmp_path):
    cfg = _small_cfg(tmp_path)
    run_dir, recorded = _record_run(cfg)
    path = frame_path(run_dir, 2)

    hub = Hub(cfg, init_frame=path)
    assert hub.model.initialization_source == "frame"
    assert hub.model.t == recorded.t
    np.testing.assert_allclose(to_cpu(hub.model.T_layers),
                               to_cpu(recorded.T_layers), rtol=1e-6)
    assert hub.init_frame_info["frame"] == 2
    assert hub.init_frame_info["time"] == recorded.t.isoformat()

    hub.model.step(2)
    hub._reset_live_state(start_recording=True)
    assert hub.model.t == recorded.t
    assert hub.model.step_count == 0


def test_load_frame_command_switches_frame_and_keeps_state_on_error(tmp_path):
    cfg = _small_cfg(tmp_path)
    run_dir, recorded = _record_run(cfg)
    hub = Hub(cfg)
    run_id = list_frame_runs(hub.run_roots)[0]["id"]
    ws = _FakeWs()

    asyncio.run(hub.handle_cmd(
        {"cmd": "load_frame", "run": run_id, "frame": 2}, ws))
    assert ws.texts[-1]["ok"] is True
    assert ws.texts[-1]["init_frame"]["frame"] == 2
    assert hub.model.t == recorded.t
    assert hub.playing is True  # the previous play state is restored
    loaded_model = hub.model

    asyncio.run(hub.handle_cmd(
        {"cmd": "load_frame", "run": run_id, "frame": 99}, ws))
    assert ws.texts[-1]["ok"] is False
    assert hub.model is loaded_model
    assert hub.init_frame_info["frame"] == 2

    asyncio.run(hub.handle_cmd({"cmd": "load_frame", "run": None}, ws))
    assert ws.texts[-1]["ok"] is True
    assert ws.texts[-1]["init_frame"] is None
    assert hub.model.initialization_source != "frame"
    assert hub.init_frame is None


def test_recording_from_a_frame_notes_its_source(tmp_path):
    cfg = _small_cfg(tmp_path)
    run_dir, _ = _record_run(cfg)
    cfg["server"]["record_enabled"] = True
    path = frame_path(run_dir, 1)

    hub = Hub(cfg, init_frame=path)
    assert hub.recorder.run_dir != run_dir
    with open(os.path.join(hub.recorder.run_dir, "manifest.json"),
              encoding="utf-8") as fp:
        manifest = json.load(fp)
    assert os.path.normpath(manifest["init_frame"]) == os.path.normpath(path)
