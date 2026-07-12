"""EarthSim 可视化服务器。

- 实时模式: 后台线程推进模拟, WebSocket 推送帧 (可暂停/播放/逐帧/调速)
- 回放模式: 播放 precompute 生成的帧序列 (可拖动时间轴)
启动: python run.py serve [-c config.yaml] [--playback output/run1]
"""
import asyncio
import json
import os
import struct

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from earthsim.analysis import analyze_point
from earthsim.backend import to_cpu
from earthsim.model import EarthModel
from earthsim.precompute import FramePlayer
from earthsim import topo as _topo

ROOT = os.path.dirname(os.path.abspath(__file__))

# 各图层的固定显示范围 (量化 + 前端色标共用)
LAYER_RANGES = {
    "press": (955.0, 1070.0), "temp": (-45.0, 45.0), "sst": (-4.0, 34.0),
    "hum": (0.0, 24.0), "cloud": (0.0, 1.0), "precip": (0.0, 1.2),
    "ice": (0.0, 1.0),
}
SCALARS = list(LAYER_RANGES.keys())


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
        if playback_dir:
            self.player = FramePlayer(playback_dir)
            self.idx = 0
            self.model = None
            g = self.player.manifest["grid"]
            self.lats, self.lons = _topo.sim_grid(g["nlat"], g["nlon"])
            elev, land = _topo.load_topo(cfg.grid.topo_file,
                                         g["nlat"], g["nlon"])
            self.land = land
            self.fields, self.time_iso = self.player.frame(0)
        else:
            self.model = EarthModel(cfg)
            if cfg.time.spinup_days:
                n = int(cfg.time.spinup_days * 86400 / self.model.dt)
                print(f"[server] spin-up {cfg.time.spinup_days} 天 ({n} 步)...")
                self.model.step(n)
            self.lats, self.lons = self.model.lats, self.model.lons
            self.land = to_cpu(self.model.land)
            self.fields = self.model.fields_cpu()
            self.time_iso = self.model.t.isoformat()

    # -------- 推进 --------
    def _advance(self, nsteps=None):
        if self.mode == "live":
            self.model.step(nsteps or int(self.cfg.server.steps_per_frame))
            self.fields = self.model.fields_cpu()
            self.time_iso = self.model.t.isoformat()
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

    # -------- 帧编码 --------
    def packet(self):
        f = self.fields
        meta = {"type": "frame", "time": self.time_iso,
                "mode": self.mode, "playing": self.playing,
                "speed": self.speed, "layers": [], "vecs": [],
                "shape": [len(self.lats), len(self.lons)],
                "vector_strides": dict(self.vector_strides)}
        if self.mode == "playback":
            meta["frame"], meta["nframes"] = self.idx, self.player.n
            ss = f.get("subsolar")
            meta["subsolar"] = [float(ss[0]), float(ss[1])] if ss is not None else [0, 0]
        else:
            meta["step"] = self.model.step_count
            meta["subsolar"] = list(getattr(self.model, "subsolar", (0, 0)))
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
            vec = np.stack([f[cu][::stride, ::stride],
                            f[cv][::stride, ::stride]], -1).astype(np.float32)
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
        elif cmd == "set_vector_stride":
            target = str(msg.get("target", "all"))
            stride = self._clip_vector_stride(
                msg.get("value", self.cfg.server.vector_stride))
            names = ("wind", "ocean") if target == "all" else (target,)
            for name in names:
                if name in self.vector_strides:
                    self.vector_strides[name] = stride
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


def create_app(cfg, playback_dir=None):
    hub = Hub(cfg, playback_dir)
    app = FastAPI(title="EarthSim")
    app.state.hub = hub

    base_png = os.path.join(ROOT, "web", "generated", "base.png")
    if not os.path.exists(base_png):
        print("[server] 生成地形底图...")
        _topo.make_base_texture(cfg.grid.topo_file, base_png)

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
        return {"mode": hub.mode, "shape": [len(hub.lats), len(hub.lons)],
                "layers": SCALARS, "ranges": LAYER_RANGES,
                "nframes": hub.player.n if hub.mode == "playback" else None,
                "backend": hub.model.backend if hub.model else "playback",
                "dt": float(cfg.time.dt),
                "vector_strides": dict(hub.vector_strides)}

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
    return app
