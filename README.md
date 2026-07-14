# EarthSim — CUDA 加速地球天气模拟

多层湿浅水大气 + 平板海洋 + 冰雪反照率反馈的全球天气模拟系统,使用真实 ETOPO 地形,支持实时交互与离线预演算回放,浏览器内 3D 地球可视化。

## 功能

- **物理**:分层大气动力(水平风、质量连续垂直速度、浮力、垂直平流和层间湍混)、地形阻力/阻挡/沿等高线偏转与迎风坡抬升、辐射、水汽循环、云降水、海冰反馈和风生洋流
- **加速**:自动检测 CUDA(CuPy),含手写融合 kernel(迎风平流+扩散);无 GPU 时回退 NumPy,物理代码完全一致
- **可视化**:three.js 3D 地球,拖动/缩放/惯性;图层:气压、气温、海温、水汽、云量、降水;叠加:云层、冰雪、昼夜晨昏线、风场箭头、洋流箭头
- **分析**:点击地球任意位置,给出该地天气结论(晴/多云/阴/小雨/中雨/大雨/雪/大风)及全部物理量
- **播放控制**:实时模式支持暂停/播放/逐帧/变速;回放模式额外支持后退与时间轴拖动
- **温度编辑**(实时模式):左侧「温度编辑」进入编辑模式后,点击或按住拖动即可给任意区域升温/降温(高斯笔刷,幅度 ±20 °C、半径 200–4000 km 可调,可只作用于地表/海面或大气)。暂停时编辑会立即回显;继续播放可观察扰动如何被环流、蒸发与辐射响应消化——例如给热带海面升温制造出气旋式辐合与强降水

## 安装

```bash
pip install -r requirements.txt
# 可选 GPU 加速(按 CUDA 版本):
pip install cupy-cuda12x
```

地形数据已附带(`data/etopo20.npz`,真实 ETOPO 20 角分)。如需重新下载:

```bash
python scripts/get_topo.py
```

## 运行

### 1. 实时模拟

```bash
python run.py serve
# 打开 http://localhost:8000
```

### 2. 预演算 + 回放(高精度提前算好再看)

```bash
python run.py precompute --days 30        # 演算 30 模拟日,存 output/run1
python run.py serve --playback output/run1
```

回放模式下界面出现时间轴与「上一帧」按钮,可任意拖动/倒退。

### 3. 性能基准

```bash
python run.py benchmark
```

## 配置(config.yaml)

| 段 | 关键项 | 说明 |
|---|---|---|
| backend | `auto / cuda / cpu` | 计算后端 |
| grid | `nlat, nlon` | 分辨率(实时建议 90×180,预演算可 180×360 以上)|
| time | `dt, spinup_days` | 时间步长(秒)、启动前预热天数 |
| physics | `vertical, topography, H0, g_eff, drag_ocean_atmosphere, drag_land_atmosphere, visc …` | 分层、地形作用及其他物理超参数 |
| precompute | `days, frame_interval_s, out_dir` | 预演算时长与帧间隔 |
| server | `host, port, max_fps, vector_stride` | 服务与推流参数 |

前端页面需联网加载 three.js CDN(cdnjs / unpkg 自动回退)。

## 真实资料初始化

模型支持启动时读取一次 NetCDF 快照，不会在积分过程中 nudging。推荐准备：

- ERA5 pressure levels：`t/q/u/v`，坐标包含 `level/latitude/longitude/time`；
- ERA5 single levels：`t2m/skt/sst/u10/v10/msl`；也可以与上面的文件合并；
- 可选 CMEMS/Copernicus Marine：`uo/vo` 以及海表温度和海冰覆盖率。

文件路径在 `config.yaml` 的 `data` 节中配置。文件存在时 `init_mode: auto` 会使用真实状态；要强制检查资料完整性可改为 `real`。模型的五个高度层会从 ERA5 压力层按标准大气压高关系插值，表面海温覆盖海洋网格，海平面气压初始化显示用气压厚度场，`uo/vo` 初始化海流。当前 `data/etopo20.npz` 已是随仓库提供的真实 ETOPO 地形，不需要随天气资料更新。

## 目录

```
earthsim/            物理与引擎(config/backend/topo/physics/model/analysis/precompute/kernels.cu)
server.py            FastAPI + WebSocket 服务
run.py               命令行入口 (serve / precompute / benchmark)
web/                 前端 (index.html / style.css / app.js)
data/etopo20.npz     真实地形
config.yaml          超参数
```

## 已知简化

多层浅水近似(非静力原始方程模式)、平板海洋(无深层环流)、诊断式云与降水。适合教学演示与大尺度环流形态,不用于真实预报。
## Data downloads

The files configured under `data:` are NetCDF data snapshots, not files that
ship with this repository.

- `data/era5_pressure_levels.nc`: download from Copernicus Climate Data Store,
  dataset "ERA5 hourly data on pressure levels". Select variables `t`, `q`,
  `u`, `v`, pressure levels covering roughly 1000-200 hPa, the target date/time,
  global area, and NetCDF output.
- `data/era5_single_levels.nc`: download from Copernicus Climate Data Store,
  dataset "ERA5 hourly data on single levels". Select `t2m`, `skt`, `sst`,
  `u10`, `v10`, `msl`, and optionally cloud/precipitation fields.
- `data/cmems_surface_currents.nc`: optional Copernicus Marine output with
  surface `uo`/`vo` and optionally `thetao`, `sst`, `siconc`.

Topography supports automatic selection. `grid.topo_files` is checked first;
the model uses the highest-resolution existing `.npz` from that list. Download
finer NOAA ETOPO 2022 terrain with:

```bash
python scripts/get_topo.py --resolution 60s
python scripts/get_topo.py --resolution 30s
```

`60s` is usually the practical default. `30s` is much larger. The old lightweight
terrain can still be rebuilt with `python scripts/get_topo.py`.
