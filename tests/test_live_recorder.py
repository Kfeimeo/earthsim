import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from server import LiveRecorder


class _Model:
    nlat, nlon = 2, 4
    dt = 60.0
    backend = "cpu"
    levels_m = np.array([100.0], np.float32)
    subsolar = (4.0, 20.0)

    def __init__(self):
        self.step_count = 0
        self.t = dt.datetime(2026, 5, 1)

    def fields_cpu(self, include_layers=False):
        fields = {"temp": np.zeros((2, 4), np.float32)}
        if include_layers:
            fields["u_layers"] = np.zeros((1, 2, 4), np.float32)
        return fields


def test_live_recorder_writes_initial_and_interval_frames(tmp_path):
    cfg = SimpleNamespace(server=SimpleNamespace(
        record_dir=str(tmp_path), record_every_steps=2, record_enabled=True))
    model = _Model()
    recorder = LiveRecorder(cfg)

    recorder.start(model)
    model.step_count = 1
    assert not recorder.save(model)
    model.step_count = 2
    model.t += dt.timedelta(seconds=120)
    assert recorder.save(model)

    with open(f"{recorder.run_dir}/manifest.json", encoding="utf-8") as fp:
        manifest = json.load(fp)
    assert manifest["times"] == ["2026-05-01T00:00:00", "2026-05-01T00:02:00"]
    assert (Path(recorder.run_dir) / "frames" / "frame_000001.npz").exists()
