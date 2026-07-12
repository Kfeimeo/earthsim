# EarthSim — CUDA 加速地球天气模拟

单层湿浅水大气 + 平板海洋 + 冰雪反照率反馈的全球天气模拟系统,使用真实 ETOPO 地形,支持实时交互与离线预演算回放,浏览器内 3D 地球可视化。

## 功能

- **物理**:大气动力(科氏力/气压梯度/平流/摩擦)、昼夜与季节辐射、感热/潜热通量、水汽循环(蒸发→凝结→降水,潜热加热)、云量诊断、海冰与反照率反馈、风生洋流
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
| physics | `h0, g_eff, beta_t, drag, visc, rh_crit …` | 全部物理超参数 |
| precompute | `days, frame_interval_s, out_dir` | 预演算时长与帧间隔 |
| server | `host, port, max_fps, vector_stride` | 服务与推流参数 |

前端页面需联网加载 three.js CDN(cdnjs / unpkg 自动回退)。

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

单层浅水近似(无垂直分层)、平板海洋(无深层环流)、诊断式云与降水。适合教学演示与大尺度环流形态,不用于真实预报。
