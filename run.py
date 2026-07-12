#!/usr/bin/env python3
"""EarthSim 入口。

  python run.py serve                       # 实时模拟 + 可视化
  python run.py serve --playback output/run1  # 回放预演算结果
  python run.py precompute --days 5         # 离线预演算
  python run.py benchmark                   # 测试单步性能 (CPU/CUDA)
"""
import argparse
import time


def main():
    ap = argparse.ArgumentParser(description="EarthSim 地球天气模拟")
    ap.add_argument("command", choices=["serve", "precompute", "benchmark"])
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--playback", default=None, help="回放目录 (serve 用)")
    ap.add_argument("--days", type=float, default=None, help="预演算天数")
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()

    from earthsim.config import load_config
    cfg = load_config(args.config)

    if args.command == "precompute":
        from earthsim.precompute import run_precompute
        run_precompute(cfg, days=args.days)

    elif args.command == "benchmark":
        from earthsim.model import EarthModel
        m = EarthModel(cfg)
        print(f"后端: {m.backend}  网格: {m.nlat}x{m.nlon}  dt={m.dt}s")
        m.step(5)  # 预热
        n = 100
        t0 = time.time()
        m.step(n)
        m.check_health()
        el = time.time() - t0
        print(f"{n} 步耗时 {el:.2f}s  ({n/el:.1f} 步/秒, "
              f"模拟加速比 x{n*m.dt/el:.0f})")

    else:  # serve
        import uvicorn
        from server import create_app
        app = create_app(cfg, playback_dir=args.playback)
        host = cfg.server.host
        port = args.port or int(cfg.server.port)
        print(f"打开浏览器访问  http://localhost:{port}")
        uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
