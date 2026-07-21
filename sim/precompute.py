"""提前演算: 离线跑模拟并把帧存盘, 供 UI 回放/逐帧分析。

精度可通过 config 中 grid / dt 灵活调节 —— 预演算模式下
可以开更高分辨率, 回放时无实时性能压力。
"""
import json
import os
import time

import numpy as np

from .model import EarthModel


try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - only used when tqdm is not installed
    tqdm = None


def run_precompute(cfg, days=None, progress=True):
    pc = cfg.precompute
    out_dir = pc.out_dir
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    model = EarthModel(cfg)
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
