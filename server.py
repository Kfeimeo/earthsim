"""EarthSim 可视化服务器。

- 实时模式: 后台线程推进模拟, WebSocket 推送帧 (可暂停/播放/逐帧/调速)
- 回放模式: 播放 precompute 生成的帧序列 (可拖动时间轴)
启动: python run.py serve [-c config.yaml] [--playback output/run1]
"""
import asyncio
import datetime as dt
import json
import os
import struct
import time

import numpy as np

from sim.analysis import analyze_point
from sim.backend import to_cpu
from sim.model import EarthModel
from sim.precompute import FramePlayer
from sim.primitive import standard_atmosphere_height
from sim import topo as _topo

ROOT = os.path.dirname(os.path.abspath(__file__))

# 各图层的固定显示范围 (量化 + 前端色标共用)
LAYER_RANGES = {
    "press": (955.0, 1070.0), "temp": (-45.0, 45.0), "sst": (-4.0, 34.0),
    "hum": (0.0, 24.0), "cloud": (0.0, 1.0), "precip": (0.0, 1.2),
    "ice": (0.0, 1.0), "ground_water": (0.0, 150.0),
}
SCALARS = list(LAYER_RANGES.keys())
# 随所选大气层次变化的标量 (第 0 层 = 海平面气压 / 近地面气温 / 近地面比湿)
LEVEL_SCALARS = ("press", "temp", "hum")
_ISA_SEA_LEVEL_HPA = 1013.25


def standard_atmosphere_pressure_hpa(height_m):
    """U.S. Standard Atmosphere 1976 pressure (hPa) at a geometric height."""
    z = float(height_m)
    if z < 11000.0:
        return _ISA_SEA_LEVEL_HPA * (1.0 - 2.25577e-5 * z) ** 5.25588
    if z < 20000.0:
        return 226.32 * np.exp(-(z - 11000.0) / 6341.6)
    return 54.75 * (1.0 + 4.6155e-6 * (z - 20000.0)) ** -34.163


def level_display_ranges(height_m, reference_pressure_hpa=None):
    """Display ranges for the level-dependent scalars of one model level.

    The surface ranges are shifted with the standard atmosphere so that a
    colormap stays informative at every level: pressure scales with the ISA
    pressure, temperature follows the 6.5 K/km lapse up to the tropopause
    and humidity decays with a 2.5 km scale height. When the layer's
    reference pressure is known it defines the level (via the ISA height);
    otherwise the nominal height is used.
    """
    if reference_pressure_hpa is not None and reference_pressure_hpa > 0:
        p_ref = float(reference_pressure_hpa)
        z = float(standard_atmosphere_height(p_ref * 100.0)[0])
    else:
        z = max(float(height_m), 0.0)
        p_ref = standard_atmosphere_pressure_hpa(z)
    z = max(z, 0.0)
    p_lo, p_hi = LAYER_RANGES["press"]
    scale = p_ref / _ISA_SEA_LEVEL_HPA
    t_lo, t_hi = LAYER_RANGES["temp"]
    shift = -0.0065 * min(z, 11000.0)
    q_hi = LAYER_RANGES["hum"][1] * np.exp(-z / 2500.0)
    q_hi = float(max(float(f"{q_hi:.2g}"), 0.05))
    return {
        "press": (round(p_lo * scale, 1), round(p_hi * scale, 1)),
        "temp": (round(t_lo + shift), round(t_hi + shift)),
        "hum": (0.0, q_hi),
    }


def global_field_stats(lats, temp, press, u, v):
    """Area-weighted global extrema and means of the displayed level."""
    w = np.cos(np.radians(np.asarray(lats, dtype=np.float64)))[:, None]
    w = np.broadcast_to(w, temp.shape)
    wsum = float(w.sum())
    speed = np.hypot(np.asarray(u, np.float64), np.asarray(v, np.float64))

    def stats(a):
        a = np.asarray(a, dtype=np.float64)
        return {"min": round(float(a.min()), 2), "max": round(float(a.max()), 2),
                "mean": round(float((a * w).sum() / wsum), 2)}

    return {"temp": stats(temp), "press": stats(press), "wind": stats(speed)}


def ground_water_stats(lats, land, ground_water):
    """Area-weighted land statistics of the ground-water store (mm)."""
    water = np.asarray(ground_water, dtype=np.float64)
    w = np.cos(np.radians(np.asarray(lats, dtype=np.float64)))[:, None]
    w = np.broadcast_to(w, water.shape) * (np.asarray(land) > 0.5)
    wsum = float(w.sum())
    if wsum <= 0.0:
        return None
    on_land = water[w > 0]
    return {"min": round(float(on_land.min()), 2),
            "max": round(float(on_land.max()), 2),
            "mean": round(float((water * w).sum() / wsum), 2)}


class LiveRecorder:
    """Persist live frames in the same format consumed by ``FramePlayer``."""

    def __init__(self, cfg):
        server = cfg.server
        root = str(getattr(server, "record_dir", "output/recordings"))
        self.root = root if os.path.isabs(root) else os.path.join(ROOT, root)
        self.every_steps = max(1, int(getattr(server, "record_every_steps", 15)))
        self.enabled = bool(getattr(server, "record_enabled", True))
        self.run_dir = None
        self.frame_count = 0

    def _new_run_dir(self):
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(self.root, stamp)
        path, suffix = base, 1
        while os.path.exists(path):
            suffix += 1
            path = f"{base}_{suffix}"
        os.makedirs(os.path.join(path, "frames"), exist_ok=True)
        return path

    def _write_manifest(self, model):
        manifest = {
            "grid": {"nlat": model.nlat, "nlon": model.nlon},
            "dt": model.dt,
            "save_every_steps": self.every_steps,
            "backend": model.backend,
            "atmosphere_levels_m": model.levels_m.tolist(),
            "times": self.times,
        }
        if getattr(model, "_primitive_enabled", False):
            manifest["dynamics_core"] = "primitive_equations"
            manifest["vertical_coordinate"] = model.vertical_coordinate
            manifest["hybrid_a_pa"] = model.hyai.tolist()
            manifest["hybrid_b"] = model.hybi.tolist()
            manifest["sigma_interfaces"] = model.sigma_interfaces.tolist()
            manifest["top_pressure_pa"] = float(model.top_pressure_pa)
        with open(os.path.join(self.run_dir, "manifest.json"), "w",
                  encoding="utf-8") as fp:
            json.dump(manifest, fp, ensure_ascii=False, indent=1)

    def start(self, model):
        started = time.perf_counter()
        self.run_dir = self._new_run_dir()
        print(f"[startup] recording directory created: "
              f"{time.perf_counter() - started:.3f}s", flush=True)
        self.frame_count = 0
        self.times = []
        self.save(model, force=True)

    def save(self, model, force=False):
        if not self.enabled:
            return False
        if self.run_dir is None:
            self.start(model)
            return True
        if not force and model.step_count % self.every_steps:
            return False
        stage_started = time.perf_counter()
        fields = model.fields_cpu(include_layers=True)
        fields = {name: np.asarray(value, dtype=np.float32)
                  for name, value in fields.items()}
        fields["subsolar"] = np.asarray(model.subsolar, dtype=np.float32)
        if force:
            print(f"[startup] initial recording fields prepared: "
                  f"{time.perf_counter() - stage_started:.3f}s", flush=True)
        path = os.path.join(self.run_dir, "frames",
                            f"frame_{self.frame_count:06d}.npz")
        stage_started = time.perf_counter()
        np.savez_compressed(path, **fields)
        if force:
            size_mib = os.path.getsize(path) / (1024 * 1024)
            print(f"[startup] initial recording compressed and written "
                  f"({size_mib:.1f} MiB): "
                  f"{time.perf_counter() - stage_started:.3f}s", flush=True)
        self.times.append(model.t.isoformat())
        self.frame_count += 1
        stage_started = time.perf_counter()
        self._write_manifest(model)
        if force:
            print(f"[startup] recording manifest written: "
                  f"{time.perf_counter() - stage_started:.3f}s", flush=True)
        return True

    def set_enabled(self, enabled, model):
        enabled = bool(enabled)
        if enabled and not self.enabled:
            self.enabled = True
            self.start(model)
        else:
            self.enabled = enabled


class Hub:
    """模拟/回放运行器 + WebSocket 广播。"""

    def __init__(self, cfg, playback_dir=None):
        self.cfg = cfg
        self.playing = True
        self.speed = 1.0
        self.clients = set()
        self.mode = "playback" if playback_dir else "live"
        stride = self._clip_vector_stride(cfg.server.vector_stride)
        self.vector_strides = {"wind": stride, "ocean": stride}
        self.wind_layer_index = 0
        self.ocean_layer_index = 0
        self.level_fields = None
        # 储水图层的色标上限取配置的持水容量
        self.layer_ranges = dict(LAYER_RANGES)
        capacity = float(getattr(cfg.physics, "ground_water_capacity_mm", 0) or 0)
        if capacity > 0:
            self.layer_ranges["ground_water"] = (0.0, capacity)
        if playback_dir:
            self.recorder = None
            self.player = FramePlayer(playback_dir)
            self.idx = 0
            self.model = None
            g = self.player.manifest["grid"]
            self.lats, self.lons = _topo.sim_grid(g["nlat"], g["nlon"])
            topo_spec = cfg.grid.get("topo_files", cfg.grid.get("topo_file", ""))
            elev, land = _topo.load_topo(topo_spec,
                                         g["nlat"], g["nlon"])
            self.land = land
            self.fields, self.time_iso = self.player.frame(0)
        else:
            self.recorder = LiveRecorder(cfg)
            self._reset_live_state(start_recording=True)

    def _reset_live_state(self, start_recording=False):
        reset_started = time.perf_counter()
        stage_started = time.perf_counter()
        self.model = EarthModel(self.cfg)
        print(f"[startup] EarthModel created: "
              f"{time.perf_counter() - stage_started:.3f}s", flush=True)
        if self.cfg.time.spinup_days:
            n = int(self.cfg.time.spinup_days * 86400 / self.model.dt)
            print(f"[server] spin-up {self.cfg.time.spinup_days} 天 ({n} 步)...")
            stage_started = time.perf_counter()
            self.model.step(n)
            print(f"[startup] model spin-up ({n} steps): "
                  f"{time.perf_counter() - stage_started:.3f}s", flush=True)
        self.lats, self.lons = self.model.lats, self.model.lons
        stage_started = time.perf_counter()
        self.land = to_cpu(self.model.land)
        print(f"[startup] land mask copied to CPU: "
              f"{time.perf_counter() - stage_started:.3f}s", flush=True)
        stage_started = time.perf_counter()
        self.fields = self.model.fields_cpu()
        print(f"[startup] initial display fields copied to CPU: "
              f"{time.perf_counter() - stage_started:.3f}s", flush=True)
        self.time_iso = self.model.t.isoformat()
        if self.recorder and self.recorder.enabled:
            if start_recording or self.recorder.run_dir is None:
                stage_started = time.perf_counter()
                self.recorder.start(self.model)
                print(f"[startup] initial recording completed: "
                      f"{time.perf_counter() - stage_started:.3f}s", flush=True)
        print(f"[startup] live state initialized: "
              f"{time.perf_counter() - reset_started:.3f}s", flush=True)

    # -------- 推进 --------
    def _advance(self, nsteps=None):
        if self.mode == "live":
            self.model.step(nsteps or int(self.cfg.server.steps_per_frame))
            self.fields = self.model.fields_cpu()
            self.time_iso = self.model.t.isoformat()
            self.recorder.save(self.model)
        else:
            self.idx = min(self.idx + 1, self.player.n - 1)
            self.fields, self.time_iso = self.player.frame(self.idx)
            if self.idx >= self.player.n - 1:
                self.playing = False

    def seek(self, k):
        if self.mode == "playback":
            self.idx = int(k)
            self.fields, self.time_iso = self.player.frame(self.idx)

    @staticmethod
    def _clip_vector_stride(v):
        return int(np.clip(int(v), 1, 60))

    def _atmosphere_levels(self):
        if self.model:
            return self.model.levels_m.tolist()
        return self.player.manifest.get("atmosphere_levels_m", [])

    def _clip_atmosphere_layer(self, k):
        levels = self._atmosphere_levels()
        n = max(len(levels), 1)
        return int(np.clip(int(k), 0, n - 1))

    def _has_layer_winds(self, f):
        return self.model is not None or ("u_layers" in f and "v_layers" in f)

    def _has_ocean_layers(self, f):
        if self.model is not None:
            return bool(getattr(self.model, "ocean_layers_enabled", False))
        return "uo_deep" in f and "vo_deep" in f

    def _wind_components(self, f):
        k = self._clip_atmosphere_layer(self.wind_layer_index)
        self.wind_layer_index = k
        if self.model:
            return to_cpu(self.model.u_layers[k]), to_cpu(self.model.v_layers[k])
        if "u_layers" in f and "v_layers" in f:
            return f["u_layers"][k], f["v_layers"][k]
        return f["u"], f["v"]

    def _has_layer_scalars(self, f):
        if self.model is not None:
            return True
        return "T_layers" in f and "q_layers" in f

    def _level_scalars(self, f):
        """Pressure/temperature/humidity of the selected level + their ranges.

        Level 0 keeps the surface products (reduced sea-level pressure, near
        surface air temperature and humidity). Higher levels expose the model
        layer itself: layer pressure in hPa, layer temperature in degC and
        layer specific humidity in g/kg.
        """
        k = self._clip_atmosphere_layer(self.wind_layer_index)
        out = {name: f[name] for name in LEVEL_SCALARS}
        ranges = {name: LAYER_RANGES[name] for name in LEVEL_SCALARS}
        if k == 0 or not self._has_layer_scalars(f):
            return out, ranges
        if self.model:
            m = self.model
            out["temp"] = to_cpu(m.T_layers[k]) - 273.15
            out["hum"] = to_cpu(m.q_layers[k]) * 1000.0
            if getattr(m, "_primitive_enabled", False):
                out["press"] = to_cpu(m.pressure_layers_pa[k]) / 100.0
        else:
            out["temp"] = f["T_layers"][k] - 273.15
            out["hum"] = f["q_layers"][k] * 1000.0
            if "pressure_layers_pa" in f:
                out["press"] = f["pressure_layers_pa"][k] / 100.0
        levels = self._atmosphere_levels()
        level_ranges = level_display_ranges(
            levels[k] if k < len(levels) else 0.0,
            self._level_reference_pressure_hpa(k))
        if out["press"] is f["press"]:
            level_ranges["press"] = LAYER_RANGES["press"]
        ranges.update(level_ranges)
        return out, ranges

    def _level_reference_pressure_hpa(self, k):
        """Reference-column pressure of layer k (hPa), or None if unknown."""
        if self.model is not None:
            ref = getattr(self.model, "reference_pressure_layers_pa", None)
            if ref is not None and k < len(ref):
                return float(ref[k]) / 100.0
            return None
        man = self.player.manifest
        a, b = man.get("hybrid_a_pa"), man.get("hybrid_b")
        if not a or not b or k + 1 >= len(a):
            return None
        ps = float(self.cfg.physics.dynamics.reference_surface_pressure_pa)
        p_i = np.asarray(a, dtype=np.float64) + np.asarray(b, dtype=np.float64) * ps
        return float(0.5 * (p_i[k] + p_i[k + 1])) / 100.0

    def _clip_ocean_layer(self, k):
        n = 2 if self._has_ocean_layers(self.fields) else 1
        return int(np.clip(int(k), 0, n - 1))

    def _ocean_components(self, f):
        k = self._clip_ocean_layer(self.ocean_layer_index)
        self.ocean_layer_index = k
        if k == 1:
            if self.model:
                return to_cpu(self.model.uo_deep), to_cpu(self.model.vo_deep)
            return f.get("uo_deep", f["uo"]), f.get("vo_deep", f["vo"])
        return f["uo"], f["vo"]

    # -------- 帧编码 --------
    def packet(self):
        f = self.fields
        if not self._has_layer_winds(f):
            self.wind_layer_index = 0
        if not self._has_ocean_layers(f):
            self.ocean_layer_index = 0
        meta = {"type": "frame", "time": self.time_iso,
                "mode": self.mode, "playing": self.playing,
                "speed": self.speed, "layers": [], "vecs": [],
                "shape": [len(self.lats), len(self.lons)],
                "vector_strides": dict(self.vector_strides),
                "wind_layer_index": self._clip_atmosphere_layer(self.wind_layer_index),
                "wind_layer_available": self._has_layer_winds(f),
                "ocean_layer_index": self._clip_ocean_layer(self.ocean_layer_index),
                "ocean_layer_available": self._has_ocean_layers(f)}
        meta["atmosphere_levels_m"] = self._atmosphere_levels()
        meta["atmosphere_layer_index"] = meta["wind_layer_index"]
        meta["level_scalars_available"] = self._has_layer_scalars(f)
        level_fields, level_ranges = self._level_scalars(f)
        wind_u, wind_v = self._wind_components(f)
        self.level_fields = dict(level_fields, u=wind_u, v=wind_v,
                                 index=meta["wind_layer_index"])
        meta["stats"] = global_field_stats(
            self.lats, level_fields["temp"], level_fields["press"],
            wind_u, wind_v)
        meta["stats"]["layer"] = meta["wind_layer_index"]
        if "ground_water" in f:
            gw = ground_water_stats(self.lats, self.land, f["ground_water"])
            if gw:
                meta["stats"]["ground_water"] = gw
        if self.mode == "playback":
            meta["frame"], meta["nframes"] = self.idx, self.player.n
            ss = f.get("subsolar")
            meta["subsolar"] = [float(ss[0]), float(ss[1])] if ss is not None else [0, 0]
        else:
            meta["step"] = self.model.step_count
            meta["subsolar"] = list(getattr(self.model, "subsolar", (0, 0)))
            meta["initialization"] = getattr(
                self.model, "initialization_source", "unknown")
            meta["recording"] = self.recorder.enabled
            meta["recorded_frames"] = self.recorder.frame_count
        payload = bytearray()
        for name in SCALARS:
            lo, hi = level_ranges.get(name, self.layer_ranges[name])
            field = level_fields.get(name, f.get(name))
            if field is None:             # 旧回放帧没有该字段
                continue
            a = np.clip((field - lo) / (hi - lo), 0, 1)
            b = (a * 255).astype(np.uint8).tobytes()
            meta["layers"].append({"name": name, "off": len(payload),
                                   "len": len(b), "min": lo, "max": hi})
            payload += b
        for name, (cu, cv) in {"wind": ("u", "v"), "ocean": ("uo", "vo")}.items():
            stride = self.vector_strides.get(
                name, self._clip_vector_stride(self.cfg.server.vector_stride))
            if name == "wind":
                u, v = wind_u, wind_v
            else:
                u, v = self._ocean_components(f)
            vec = np.stack([u[::stride, ::stride],
                            v[::stride, ::stride]], -1).astype(np.float32)
            b = vec.tobytes()
            meta["vecs"].append({"name": name, "off": len(payload),
                                 "len": len(b), "shape": list(vec.shape),
                                 "stride": stride})
            payload += b
        mj = json.dumps(meta).encode()
        return struct.pack("<I", len(mj)) + mj + bytes(payload)

    # -------- 主循环 --------
    async def loop(self):
        srv = self.cfg.server
        while True:
            dt_frame = 1.0 / (float(srv.max_fps) * max(self.speed, 0.01))
            if self.playing and self.clients:
                try:
                    await asyncio.to_thread(self._advance)
                except FloatingPointError as e:
                    print("[server] 数值异常:", e)
                    self.playing = False
                await self.broadcast()
            await asyncio.sleep(max(dt_frame, 0.02) if self.playing else 0.1)

    async def broadcast(self, ws=None):
        pkt = self.packet()
        targets = [ws] if ws else list(self.clients)
        for c in targets:
            try:
                await c.send_bytes(pkt)
            except Exception:
                self.clients.discard(c)

    # -------- 控制 --------
    async def handle_cmd(self, msg):
        cmd = msg.get("cmd")
        if cmd == "play":
            self.playing = True
        elif cmd == "pause":
            self.playing = False
        elif cmd == "step":       # 逐帧
            self.playing = False
            await asyncio.to_thread(self._advance,
                                    1 if self.mode == "live" else None)
            await self.broadcast()
        elif cmd == "back" and self.mode == "playback":
            self.playing = False
            self.seek(self.idx - 1)
            await self.broadcast()
        elif cmd == "seek" and self.mode == "playback":
            self.playing = False
            self.seek(msg.get("value", 0))
            await self.broadcast()
        elif cmd == "speed":
            self.speed = float(np.clip(msg.get("value", 1.0), 0.1, 16))
        elif cmd == "reset" and self.mode == "live":
            was_playing = self.playing
            await asyncio.to_thread(self._reset_live_state, True)
            self.playing = was_playing
            await self.broadcast()
        elif cmd == "record" and self.mode == "live":
            enabled = bool(msg.get("enabled", not self.recorder.enabled))
            await asyncio.to_thread(self.recorder.set_enabled, enabled, self.model)
            await self.broadcast()
        elif cmd == "set_vector_stride":
            target = str(msg.get("target", "all"))
            stride = self._clip_vector_stride(
                msg.get("value", self.cfg.server.vector_stride))
            names = ("wind", "ocean") if target == "all" else (target,)
            for name in names:
                if name in self.vector_strides:
                    self.vector_strides[name] = stride
            await self.broadcast()
        elif cmd in ("set_wind_layer", "set_atmosphere_layer"):
            # 一个高度选择同时作用于风场与随高度变化的标量图层
            self.wind_layer_index = self._clip_atmosphere_layer(
                msg.get("value", self.wind_layer_index))
            await self.broadcast()
        elif cmd == "set_ocean_layer":
            self.ocean_layer_index = self._clip_ocean_layer(
                msg.get("value", self.ocean_layer_index))
            await self.broadcast()
        elif cmd == "edit_temp" and self.mode == "live":
            def _edit():
                self.model.apply_temp_edit(
                    lat_deg=float(msg.get("lat", 0)),
                    lon_deg=float(msg.get("lon", 0)),
                    radius_km=float(np.clip(msg.get("radius", 800), 50, 6000)),
                    delta=float(np.clip(msg.get("delta", 5), -30, 30)),
                    target=str(msg.get("target", "both")))
                self.fields = self.model.fields_cpu()
            await asyncio.to_thread(_edit)
            if not self.playing:          # 暂停时立即回显编辑效果
                await self.broadcast()
        elif cmd == "edit_wind_zero" and self.mode == "live":
            def _edit():
                self.model.apply_wind_zero_edit(
                    lat_deg=float(msg.get("lat", 0)),
                    lon_deg=float(msg.get("lon", 0)),
                    radius_km=float(np.clip(msg.get("radius", 800), 50, 6000)),
                    layer=msg.get("layer", self.wind_layer_index))
                self.fields = self.model.fields_cpu()
            await asyncio.to_thread(_edit)
            if not self.playing:
                await self.broadcast()
        elif cmd == "edit_cyclone" and self.mode == "live":
            def _edit():
                self.model.apply_cyclone_edit(
                    lat_deg=float(msg.get("lat", 0)),
                    lon_deg=float(msg.get("lon", 0)),
                    radius_km=float(np.clip(msg.get("radius", 900), 50, 6000)),
                    strength_ms=float(np.clip(msg.get("strength", 35), 0, 150)),
                    layer=msg.get("layer", self.wind_layer_index))
                self.fields = self.model.fields_cpu()
            await asyncio.to_thread(_edit)
            if not self.playing:
                await self.broadcast()


def create_app(cfg, playback_dir=None):
    create_started = time.perf_counter()
    stage_started = time.perf_counter()
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    print(f"[startup] FastAPI modules imported: "
          f"{time.perf_counter() - stage_started:.3f}s", flush=True)

    stage_started = time.perf_counter()
    hub = Hub(cfg, playback_dir)
    print(f"[startup] simulation hub created ({hub.mode}): "
          f"{time.perf_counter() - stage_started:.3f}s", flush=True)
    app = FastAPI(title="EarthSim")
    app.state.hub = hub

    stage_started = time.perf_counter()
    base_png = os.path.join(ROOT, "web", "generated", "base.png")
    base_width = int(getattr(cfg.server, "basemap_width", 2160))
    needs_basemap = not os.path.exists(base_png)
    if not needs_basemap:
        from PIL import Image
        with Image.open(base_png) as image:
            needs_basemap = image.width != base_width
    print(f"[startup] basemap cache checked "
          f"({'miss' if needs_basemap else 'hit'}): "
          f"{time.perf_counter() - stage_started:.3f}s", flush=True)
    if needs_basemap:
        print("[server] 生成地形底图...")
        topo_spec = cfg.grid.get("topo_files", cfg.grid.get("topo_file", ""))
        stage_started = time.perf_counter()
        _topo.make_base_texture(topo_spec, base_png, width=base_width)
        print(f"[startup] basemap generated: "
              f"{time.perf_counter() - stage_started:.3f}s", flush=True)

    @app.on_event("startup")
    async def _start():
        asyncio.create_task(hub.loop())

    @app.get("/")
    async def index():
        return FileResponse(os.path.join(ROOT, "web", "index.html"))

    @app.get("/api/basemap.png")
    async def basemap():
        return FileResponse(base_png)

    @app.get("/api/land.bin")
    async def land_bin():
        from fastapi.responses import Response
        return Response((hub.land * 255).astype(np.uint8).tobytes(),
                        media_type="application/octet-stream")

    @app.get("/api/meta")
    async def meta():
        levels = (hub.model.levels_m.tolist() if hub.model else
                  hub.player.manifest.get("atmosphere_levels_m", []))
        return {"mode": hub.mode, "shape": [len(hub.lats), len(hub.lons)],
                "layers": SCALARS, "ranges": hub.layer_ranges,
                "atmosphere_levels_m": levels,
                "nframes": hub.player.n if hub.mode == "playback" else None,
                "backend": hub.model.backend if hub.model else "playback",
                "initialization": (getattr(hub.model, "initialization_source", "unknown")
                                   if hub.model else "playback"),
                "dt": float(cfg.time.dt),
                "vector_strides": dict(hub.vector_strides),
                "wind_layer_index": hub._clip_atmosphere_layer(hub.wind_layer_index),
                "atmosphere_layer_index": hub._clip_atmosphere_layer(hub.wind_layer_index),
                "wind_layer_available": hub._has_layer_winds(hub.fields),
                "level_scalars_available": hub._has_layer_scalars(hub.fields),
                "ocean_layer_index": hub._clip_ocean_layer(hub.ocean_layer_index),
                "ocean_layer_available": hub._has_ocean_layers(hub.fields),
                "recording": bool(hub.recorder and hub.recorder.enabled),
                "recorded_frames": (hub.recorder.frame_count if hub.recorder else 0)}

    @app.get("/api/diagnostics")
    async def diagnostics():
        if hub.model is None:
            return {"available": False, "mode": hub.mode}
        out = hub.model.energy_diagnostics()
        out["step"] = int(hub.model.step_count)
        out["time"] = hub.model.t.isoformat()
        return out

    @app.get("/api/analyze")
    async def analyze(lat: float, lon: float, layer: int = -1):
        try:
            out = analyze_point(hub.fields, hub.lats, hub.lons,
                                hub.land, lat, lon)
            # 附带当前所选大气层次 (与画面上的图层/风场一致) 的要素
            lf = hub.level_fields
            if layer >= 0 and lf and lf.get("index") == hub._clip_atmosphere_layer(layer):
                levels = hub._atmosphere_levels()
                k = lf["index"]
                i = int(np.clip(np.abs(hub.lats - out["lat"]).argmin(),
                                0, len(hub.lats) - 1))
                j = int(np.argmin(np.minimum(
                    np.abs(hub.lons - out["lon"]),
                    360 - np.abs(hub.lons - out["lon"]))))
                u, v = float(lf["u"][i, j]), float(lf["v"][i, j])
                out["level"] = {
                    "index": k,
                    "height_m": (round(float(levels[k])) if k < len(levels)
                                 else None),
                    "temp": round(float(lf["temp"][i, j]), 1),
                    "pressure": round(float(lf["press"][i, j]), 1),
                    "humidity": round(float(lf["hum"][i, j]), 2),
                    "wind_speed": round(float(np.hypot(u, v)), 1),
                    "wind_dir": round((np.degrees(np.arctan2(u, v)) + 360) % 360),
                }
            return out
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        hub.clients.add(ws)
        await hub.broadcast(ws)
        try:
            while True:
                msg = json.loads(await ws.receive_text())
                await hub.handle_cmd(msg)
        except (WebSocketDisconnect, Exception):
            hub.clients.discard(ws)

    app.mount("/static", StaticFiles(directory=os.path.join(ROOT, "web")),
              name="static")
    print(f"[startup] FastAPI routes registered; create_app complete: "
          f"{time.perf_counter() - create_started:.3f}s", flush=True)
    return app
