import json
import os

import numpy as np
import pytest

from sim.config import load_config
from sim.model import EarthModel
from sim.precompute import FramePlayer
from sim.regrid import Regridder, grid_centres, regrid_fields, remap_levels


def weighted_mean(field, nlat):
    lat, _ = grid_centres(nlat, field.shape[-1])
    w = np.cos(np.radians(lat))[:, None]
    return float((field * w).sum() / (w.sum() * field.shape[-1]))


def smooth_field(nlat, nlon):
    lat, lon = grid_centres(nlat, nlon)
    return (10.0 * np.sin(np.radians(lat))[:, None] ** 2
            + 3.0 * np.cos(np.radians(2 * lon))[None, :]).astype(np.float32)


# ------------------------------------------------------------ operators
def test_identity_and_constant_preservation():
    field = smooth_field(12, 24)
    assert Regridder((12, 24), (12, 24))(field) is field
    for shape in ((6, 12), (24, 48), (6, 48), (24, 12)):
        out = Regridder((12, 24), shape)(np.full((12, 24), 7.0, np.float32))
        np.testing.assert_allclose(out, 7.0, rtol=1e-6)
        assert out.shape == shape


def test_coarsening_is_conservative_and_refining_is_smooth():
    fine = smooth_field(48, 96)
    coarse = Regridder((48, 96), (12, 24))(fine)
    assert coarse.shape == (12, 24)
    assert weighted_mean(coarse, 12) == pytest.approx(weighted_mean(fine, 48), rel=1e-5)
    # Aggregated values stay within the range of the source.
    assert coarse.min() >= fine.min() - 1e-5 and coarse.max() <= fine.max() + 1e-5

    refined = Regridder((12, 24), (48, 96))(coarse)
    assert refined.shape == (48, 96)
    # Interpolation reproduces the smooth field closely and introduces no
    # block edges: neighbouring differences are small everywhere.
    assert np.abs(refined - fine).max() < 0.6
    assert np.abs(np.diff(refined, axis=1)).max() < 0.5
    # Longitude wraps: the seam column is as smooth as the interior.
    seam = np.abs(refined[:, 0] - refined[:, -1]).max()
    assert seam < 0.5


def test_mixed_direction_and_leading_axes():
    source = np.stack([smooth_field(24, 12) + k for k in range(3)])
    out = Regridder((24, 12), (6, 48))(source)
    assert out.shape == (3, 6, 48)
    assert weighted_mean(out[1], 6) == pytest.approx(weighted_mean(source[1], 24), rel=2e-2)


def test_regrid_fields_skips_non_grid_entries():
    frame = {"temp": smooth_field(12, 24), "u_layers": np.zeros((3, 12, 24), np.float32),
             "subsolar": np.array([1.0, 2.0], np.float32)}
    out = regrid_fields(frame, (6, 12))
    assert out["temp"].shape == (6, 12)
    assert out["u_layers"].shape == (3, 6, 12)
    np.testing.assert_array_equal(out["subsolar"], frame["subsolar"])


def test_remap_levels_interpolates_in_log_pressure():
    src_p = np.array([95000.0, 70000.0, 50000.0, 30000.0])
    field = np.log(src_p)[:, None, None] * np.ones((4, 2, 3))
    dst_p = np.array([90000.0, 60000.0, 40000.0, 20000.0, 10000.0])
    out = remap_levels(field, src_p, dst_p)
    assert out.shape == (5, 2, 3)
    np.testing.assert_allclose(out[:3, 0, 0], np.log(dst_p[:3]), rtol=1e-6)
    # Beyond the top source layer values are clamped.
    np.testing.assert_allclose(out[3:, 0, 0], np.log(30000.0), rtol=1e-6)
    columns = np.stack([src_p * 0.9, src_p * 1.05], axis=-1)[:, :, None] * np.ones((4, 2, 3))
    out3 = remap_levels(field, columns, dst_p)
    assert out3.shape == (5, 2, 3)


# ------------------------------------------------------------ frames
def small_config(nlat, nlon, levels=16):
    cfg = load_config()
    cfg["backend"] = "cpu"
    cfg["grid"].update(nlat=nlat, nlon=nlon, topo_file="")
    cfg["grid"]["topo_files"] = []
    cfg["data"]["init_mode"] = "ideal"
    cfg["time"]["dt"] = 60.0
    cfg["server"]["record_enabled"] = False
    cfg["physics"]["vertical"]["coordinate"] = "hybrid"
    cfg["physics"]["vertical"]["hybrid"]["levels"] = levels
    cfg["physics"]["dynamics"].update(top_pressure_pa=300.0,
                                      min_surface_pressure_pa=45000.0)
    return cfg


def write_run(tmp_path, model, full_state, nframes=2):
    run_dir = tmp_path / "run"
    frames = run_dir / "frames"
    frames.mkdir(parents=True)
    times = []
    for k in range(nframes):
        model.step(2)
        fields = model.fields_cpu(include_layers=True, full_state=full_state)
        np.savez_compressed(frames / f"frame_{k:06d}.npz",
                            **{key: np.asarray(v, np.float32) for key, v in fields.items()},
                            subsolar=np.array(model.subsolar, np.float32))
        times.append(model.t.isoformat())
    manifest = {
        "grid": {"nlat": model.nlat, "nlon": model.nlon}, "dt": model.dt,
        "save_every_steps": 2, "backend": "cpu",
        "atmosphere_levels_m": model.levels_m.tolist(),
        "atmosphere_levels_hpa": [float(p) / 100.0 for p in model.reference_pressure_layers_pa],
        "vertical_coordinate": "hybrid", "hybrid_a_pa": model.hyai.tolist(),
        "hybrid_b": model.hybi.tolist(), "top_pressure_pa": model.top_pressure_pa,
        "times": times,
    }
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as fp:
        json.dump(manifest, fp)
    return run_dir


def test_frame_player_regrids_to_target_shape(tmp_path):
    model = EarthModel(small_config(12, 24))
    run_dir = write_run(tmp_path, model, full_state=False)

    native = FramePlayer(str(run_dir))
    fields, _ = native.frame(0)
    assert fields["temp"].shape == (12, 24) and native.shape == (12, 24)

    finer = FramePlayer(str(run_dir), target_shape=(24, 48))
    fields, _ = finer.frame(1)
    assert fields["temp"].shape == (24, 48)
    assert fields["u_layers"].shape == (model.nz, 24, 48)
    assert fields["subsolar"].shape == (2,)

    coarser = FramePlayer(str(run_dir), target_shape=(6, 12))
    fields, _ = coarser.frame(0)
    native_fields, _ = native.frame(0)
    assert weighted_mean(fields["temp"], 6) == pytest.approx(
        weighted_mean(native_fields["temp"], 12), rel=1e-4)


def test_hub_plays_recording_on_configured_grid(tmp_path):
    from server import Hub

    model = EarthModel(small_config(12, 24))
    run_dir = write_run(tmp_path, model, full_state=False)
    cfg = small_config(18, 36)
    hub = Hub(cfg, playback_dir=str(run_dir))
    assert len(hub.lats) == 18 and len(hub.lons) == 36
    assert hub.fields["temp"].shape == (18, 36)
    assert hub.land.shape == (18, 36)
    result = hub.analyze(10.0, 20.0)
    assert result["profile"]["count"] == model.nz

    cfg["server"]["playback_grid"] = "native"
    native = Hub(cfg, playback_dir=str(run_dir))
    assert len(native.lats) == 12 and native.fields["temp"].shape == (12, 24)


@pytest.mark.parametrize("target_shape", [(18, 36), (8, 16)])
def test_frame_initialization_regrids_full_state(tmp_path, target_shape):
    source = EarthModel(small_config(12, 24))
    source.ground_water[...] = 42.0 * np.asarray(source.land)
    run_dir = write_run(tmp_path, source, full_state=True)
    frame_path = run_dir / "frames" / "frame_000001.npz"

    cfg = small_config(*target_shape)
    cfg["data"].update(init_mode="frame", frame_file=str(frame_path))
    model = EarthModel(cfg)

    assert model.initialization_source == "frame"
    assert model.T_layers.shape == (source.nz,) + target_shape
    assert model.t == source.t
    model.check_health()
    # Global means of the regridded state follow the recording.
    for attr in ("Ts", "surface_pressure"):
        src_mean = weighted_mean(np.asarray(getattr(source, attr)), 12)
        dst_mean = weighted_mean(np.asarray(getattr(model, attr)), target_shape[0])
        assert dst_mean == pytest.approx(src_mean, rel=5e-3)
    src_wind = weighted_mean(np.asarray(source.u_layers[5]), 12)
    dst_wind = weighted_mean(np.asarray(model.u_layers[5]), target_shape[0])
    assert dst_wind == pytest.approx(src_wind, abs=0.3)
    assert float(np.asarray(model.T_layers[-1]).mean()) == pytest.approx(
        float(np.asarray(source.T_layers[-1]).mean()), abs=1.0)
    model.step(3)
    model.check_health()


def test_frame_initialization_remaps_levels_and_surface_only_frames(tmp_path):
    source = EarthModel(small_config(12, 24, levels=16))
    run_dir = write_run(tmp_path, source, full_state=False)

    cfg = small_config(12, 24, levels=20)
    cfg["data"].update(init_mode="frame", frame_file=str(run_dir))   # directory: last frame
    model = EarthModel(cfg)

    assert model.nz == 20 and model.u_layers.shape == (20, 12, 24)
    assert model.initialization_source == "frame"
    # Lowest-layer temperature follows the recorded 2-D field.
    np.testing.assert_allclose(np.asarray(model.T_layers[0]),
                               np.asarray(source.T_layers[0]), atol=0.5)
    # Winds are remapped in pressure, so the jet level keeps its speed.
    assert float(np.abs(model.u_layers).max()) == pytest.approx(
        float(np.abs(source.u_layers).max()), rel=0.2)
    model.step(2)
    model.check_health()


def test_frame_initialization_requires_a_file():
    cfg = small_config(8, 16)
    cfg["data"].update(init_mode="frame", frame_file="")
    with pytest.raises(ValueError, match="frame_file"):
        EarthModel(cfg)
