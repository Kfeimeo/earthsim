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
from sim import topo as _topo

ROOT = os.path.dirname(os.path.abspath(__file__))

# 各图层的固定显示范围 (量化 + 前端色标共用)
LAYER_RANGES = {
    "press": (955.0, 1070.0), "temp": (-45.0, 45.0), "sst": (-4.0, 34.0),
    "hum": (0.0, 24.0), "cloud": (0.0, 1.0), "precip": (0.0, 1.2),
    "ice": (0.0, 1.0),
}
SCALARS = list(LAYER_RANGES.keys())


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
            lo, hi = LAYER_RANGES[name]
            a = np.clip((f[name] - lo) / (hi - lo), 0, 1)
            b = (a * 255).astype(np.uint8).tobytes()
            meta["layers"].append({"name": name, "off": len(payload),
                                   "len": len(b), "min": lo, "max": hi})
            payload += b
        for name, (cu, cv) in {"wind": ("u", "v"), "ocean": ("uo", "vo")}.items():
            stride = self.vector_strides.get(
                name, self._clip_vector_stride(self.cfg.server.vector_stride))
            if name == "wind":
                u, v = self._wind_components(f)
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
        elif cmd == "set_wind_layer":
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
                "layers": SCALARS, "ranges": LAYER_RANGES,
                "atmosphere_levels_m": levels,
                "nframes": hub.player.n if hub.mode == "playback" else None,
                "backend": hub.model.backend if hub.model else "playback",
                "initialization": (getattr(hub.model, "initialization_source", "unknown")
                                   if hub.model else "playback"),
                "dt": float(cfg.time.dt),
                "vector_strides": dict(hub.vector_strides),
                "wind_layer_index": hub._clip_atmosphere_layer(hub.wind_layer_index),
                "wind_layer_available": hub._has_layer_winds(hub.fields),
                "ocean_layer_index": hub._clip_ocean_layer(hub.ocean_layer_index),
                "ocean_layer_available": hub._has_ocean_layers(hub.fields),
                "recording": bool(hub.recorder and hub.recorder.enabled),
                "recorded_frames": (hub.recorder.frame_count if hub.recorder else 0)}

    @app.get("/api/analyze")
    async def analyze(lat: float, lon: float):
        try:
            return analyze_point(hub.fields, hub.lats, hub.lons,
                                 hub.land, lat, lon)
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
