"""提前演算: 离线跑模拟并把帧存盘, 供 UI 回放/逐帧分析。

精度可通过 config 中 grid / dt 灵活调节 —— 预演算模式下
可以开更高分辨率, 回放时无实时性能压力。
"""
import datetime as dt
import json
import os
import re
import time

import numpy as np

from .backend import to_cpu
from .model import EarthModel


try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - only used when tqdm is not installed
    tqdm = None


def load_initial_frame(path):
    """Return (fields, time, manifest) of one saved frame file.

    The simulation time and the grid description live in the run's
    ``manifest.json`` next to the ``frames`` directory.
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"初始帧文件不存在: {path}")
    match = re.fullmatch(r"frame_(\d+)\.npz", os.path.basename(path))
    if match is None:
        raise ValueError(f"初始帧文件名应为 frame_NNNNNN.npz: {path}")
    manifest_path = os.path.join(
        os.path.dirname(os.path.dirname(path)), "manifest.json")
    if not os.path.isfile(manifest_path):
        raise ValueError(
            f"找不到初始帧对应的 manifest.json: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fp:
        manifest = json.load(fp)
    index, times = int(match.group(1)), manifest.get("times", [])
    if index >= len(times):
        raise ValueError(
            f"manifest.json 只记录了 {len(times)} 帧, 没有第 {index} 帧的时刻")
    with np.load(path) as archive:
        fields = {key: archive[key] for key in archive.files}
    return fields, dt.datetime.fromisoformat(times[index]), manifest


def check_frame_matches_model(model, manifest, fields):
    """Raise ValueError unless the frame was produced on the model setup."""
    problems = []

    def differs(name, saved, current, rtol=1e-5):
        if saved is None:
            problems.append(f"manifest 缺少 {name}")
            return
        saved = np.atleast_1d(np.asarray(saved, dtype=np.float64))
        current = np.atleast_1d(np.asarray(current, dtype=np.float64))
        if saved.shape != current.shape:
            problems.append(
                f"{name} 长度不一致: 帧 {saved.size}, 当前 {current.size}")
        elif not np.allclose(saved, current, rtol=rtol, atol=1e-6):
            problems.append(f"{name} 数值不一致")

    grid = manifest.get("grid", {})
    saved_grid = (grid.get("nlat"), grid.get("nlon"))
    if saved_grid != (model.nlat, model.nlon):
        problems.append(
            f"网格分辨率不一致: 帧 {saved_grid[0]}x{saved_grid[1]}, "
            f"当前 {model.nlat}x{model.nlon}")
    if not getattr(model, "_primitive_enabled", False):
        problems.append("当前动力核心不是 primitive_equations")
    if manifest.get("dynamics_core") != "primitive_equations":
        problems.append(
            f"帧的动力核心为 {manifest.get('dynamics_core', '未记录')}, "
            "需要 primitive_equations")
    elif getattr(model, "_primitive_enabled", False):
        if manifest.get("vertical_coordinate") != model.vertical_coordinate:
            problems.append(
                f"垂直坐标不一致: 帧 {manifest.get('vertical_coordinate')}, "
                f"当前 {model.vertical_coordinate}")
        differs("hybrid_a_pa (垂直层 A 系数)", manifest.get("hybrid_a_pa"),
                model.hyai)
        differs("hybrid_b (垂直层 B 系数)", manifest.get("hybrid_b"),
                model.hybi)
        differs("top_pressure_pa (模式顶气压)",
                manifest.get("top_pressure_pa"), model.top_pressure_pa)
    differs("atmosphere_levels_m (层高度)",
            manifest.get("atmosphere_levels_m"), model.levels_m, rtol=1e-4)

    expected = (model.nz, model.nlat, model.nlon)
    if "T_layers" not in fields:
        problems.append("帧不含分层场 (T_layers 等), 无法作为初始条件")
    elif fields["T_layers"].shape != expected:
        problems.append(
            f"帧数组形状 {fields['T_layers'].shape} 与当前 {expected} 不一致")
    elif "ground_water" in fields:
        # 储水只存在于陆地格点: 海洋格点上有水说明海陆分布 (地形) 不同
        ocean = np.asarray(to_cpu(model.land)) < 0.5
        if np.any(fields["ground_water"][ocean] > 0):
            problems.append("帧的海陆分布与当前地形不一致")
    if getattr(model, "ocean_layers_enabled", False) \
            and "sst_deep" not in fields:
        problems.append("当前启用了深层海洋, 但帧不含深层海洋场")

    if problems:
        raise ValueError(
            "初始帧与当前配置不一致:\n  - " + "\n  - ".join(problems))
    if abs(float(manifest.get("dt", model.dt)) - model.dt) > 1e-9:
        print(f"[precompute] 注意: 帧的时间步长 dt={manifest.get('dt')}s, "
              f"当前 dt={model.dt}s", flush=True)


def run_precompute(cfg, days=None, progress=True, init_frame=None):
    pc = cfg.precompute
    out_dir = pc.out_dir
    init_frame = str(init_frame or getattr(pc, "init_frame", "") or "")
    if init_frame:
        init_frame = os.path.abspath(init_frame)
        source_dir = os.path.dirname(os.path.dirname(init_frame))
        if os.path.abspath(out_dir) == source_dir:
            raise ValueError(
                "precompute.out_dir 与初始帧所在目录相同, 继续演算会覆盖原有帧; "
                "请改用新的输出目录")
    frames_dir = os.path.join(out_dir, "frames")

    model = EarthModel(cfg)
    if init_frame:
        fields, start_time, source_manifest = load_initial_frame(init_frame)
        check_frame_matches_model(model, source_manifest, fields)
        model.load_frame_state(fields, start_time)
        print(f"[precompute] 初始条件: {init_frame}  "
              f"(模拟时刻 {start_time.isoformat()})", flush=True)
    os.makedirs(frames_dir, exist_ok=True)
    days = float(days if days is not None else pc.days)
    save_every = int(pc.save_every_steps)
    total_steps = int(days * 86400 / model.dt)
    nframes = total_steps // save_every

    manifest = {
        "grid": {"nlat": model.nlat, "nlon": model.nlon},
        "dt": model.dt,
        "save_every_steps": save_every,
        "backend": model.backend,
        "atmosphere_levels_m": model.levels_m.tolist(),
        "times": [],
    }
    if init_frame:
        manifest["init_frame"] = init_frame
    if getattr(model, "_primitive_enabled", False):
        manifest["dynamics_core"] = "primitive_equations"
        manifest["vertical_coordinate"] = model.vertical_coordinate
        manifest["hybrid_a_pa"] = model.hyai.tolist()
        manifest["hybrid_b"] = model.hybi.tolist()
        manifest["sigma_interfaces"] = model.sigma_interfaces.tolist()
        manifest["top_pressure_pa"] = float(model.top_pressure_pa)

    t0 = time.time()
    frame_iter = range(nframes)
    progress_bar = None
    if progress and tqdm is not None:
        progress_bar = tqdm(
            frame_iter,
            total=nframes,
            desc="[precompute] 预演算",
            unit="frame",
            dynamic_ncols=True,
        )
        frame_iter = progress_bar

    for k in frame_iter:
        model.step(save_every)
        f = model.fields_cpu(include_layers=True)
        np.savez_compressed(
            os.path.join(frames_dir, f"frame_{k:06d}.npz"),
            **{key: v.astype(np.float32) for key, v in f.items()},
            subsolar=np.array(model.subsolar, np.float32),
        )
        manifest["times"].append(model.t.isoformat())

        if progress_bar is not None:
            progress_bar.set_postfix_str(
                f"模拟时刻={model.t.isoformat(timespec='seconds')}"
            )
        elif progress and (k % 10 == 0 or k == nframes - 1):
            el = time.time() - t0
            print(
                f"[precompute] 帧 {k + 1}/{nframes}  "
                f"模拟时刻 {model.t}  耗时 {el:.1f}s",
                flush=True,
            )

    if progress_bar is not None:
        progress_bar.close()

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=1)
    print(f"[precompute] 完成: {nframes} 帧 -> {out_dir}")
    return out_dir


class FramePlayer:
    """回放器: 惰性加载 + 小缓存。"""

    def __init__(self, run_dir):
        self.dir = run_dir
        with open(os.path.join(run_dir, "manifest.json"), encoding="utf-8") as fp:
            self.manifest = json.load(fp)
        self.n = len(self.manifest["times"])
        self._cache, self._order = {}, []

    def frame(self, k):
        k = int(max(0, min(self.n - 1, k)))
        if k not in self._cache:
            z = np.load(os.path.join(self.dir, "frames", f"frame_{k:06d}.npz"))
            self._cache[k] = {key: z[key] for key in z.files}
            self._order.append(k)
            if len(self._order) > 32:
                self._cache.pop(self._order.pop(0), None)
        return self._cache[k], self.manifest["times"][k]
