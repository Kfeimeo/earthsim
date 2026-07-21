from pathlib import Path

import numpy as np

from sim import topo


def _write_npz(path: Path, shape, value):
    lats = np.linspace(-90, 90, shape[0], dtype=np.float64)
    lons = np.linspace(0, 360, shape[1], endpoint=False, dtype=np.float64)
    arr = np.full(shape, value, dtype=np.float32)
    np.savez_compressed(path, topo=arr, lats=lats, lons=lons)


def test_resolve_topo_path_selects_finest_candidate(tmp_path):
    coarse = tmp_path / "etopo20.npz"
    fine = tmp_path / "etopo2022_60s.npz"
    _write_npz(coarse, (3, 6), -100.0)
    _write_npz(fine, (7, 14), 500.0)

    assert topo.resolve_topo_path([str(coarse), str(fine)]) == str(fine)

    elev, land = topo.load_topo([str(coarse), str(fine)], 4, 8)
    assert elev.shape == (4, 8)
    assert float(elev.mean()) == 500.0
    assert float(land.mean()) == 1.0
