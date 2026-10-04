# EarthSim — CUDA 加速地球天气模拟

多层湿静力原始方程大气 + 平板海洋 + 冰雪反照率反馈的全球天气模拟系统，支持实时交互、离线预演算回放和浏览器内 3D 地球可视化。默认理想初始化采用平坦动力学下边界以建立平衡；ETOPO 仍用于地表显示和海陆分布，真实资料初始化保留地形动力学。

## 功能

- **物理**:分层大气动力(水平风、质量连续垂直速度、浮力、垂直平流和层间湍混)、地形阻力/阻挡/沿等高线偏转与迎风坡抬升、辐射、水汽循环、云降水、海冰反馈和风生洋流
- **输送与加速**:默认使用 MUSCL/TVD 二阶有限体积输送（MC 限制器），保留一阶迎风回退；自动检测 CUDA(CuPy)，CPU/GPU 使用一致的面通量离散
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

文件路径在 `config.yaml` 的 `data` 节中配置。默认 `init_mode: ideal`；`auto` 尝试真实资料，读取失败时回退到平衡理想状态；`real` 读取失败会报错。资料先按 `vertical.levels_m` 的名义高度插值，再初始化 sigma 网格；这不是严格的原生 sigma 分析场重映射。海平面气压结合地形和最低层虚温估算地面气压，表面海温覆盖海洋网格，`uo/vo` 初始化海流。

NetCDF4/HDF5 文件需要当前解释器中可用的 `netCDF4` 或 `h5netcdf` 后端；仅有 scipy 时不能读取这类文件。读取成功也不代表平衡：插值、地形变化和压力转换可能产生残差。当前未实现 ERA5 数字滤波初始化或地形平衡调整，超过启动阈值的数据会在首次正式积分前被拒绝；不要仅为绕过检查而提高阈值。

## 平衡理想初始化与启动检查

原始方程的理想初值按以下顺序构建：指定轴对称 `T(纬度,p)` 和压力相关比湿；设置平坦地面位势及常数 `ps`；使用包含水汽的虚温静力积分得到 `Phi`；用积分器同一压力梯度离散算子求解各层梯度风，令 `v=0`；最后加入默认 **0.001 K** 的非轴对称温度扰动。风场不再独立指定。`ideal_wind_enabled: false` 同时去掉温度纬向差异，生成静止背景。

此模式会将动力学地形高度、坡度和地面位势置零，即使 `topography.enabled: true`，也不会把真实山地强行塞入轴对称平衡。海陆热力差异、摩擦、辐射和水汽过程仍会在积分后驱动调整；初始动力平衡不等于完整物理过程的稳态。

相关设置位于 `physics.initial_conditions`：

```yaml
balanced_perturbation_k: 0.001  # 0 可测试无扰动平衡；绝对值上限 0.01 K
balance_tolerance_ms2: 1.0e-4
```

旧的 `ideal_wave_amp_K`、`ideal_humidity_wave`、`ideal_wind_wave_ms` 和手绘急流参数只用于旧浅水初始化，不控制原始方程平衡初值。梯度风无实根或超过 `umax` 时直接报错，不裁剪风场来伪造平衡。

启动计算离散南北动量残差：

```text
Rv = -Dy(Phi) - Rd*Tv*Dy(log p) - f*u - tan(latitude)/a*u²
```

日志输出每层最大绝对残差、对应纬度、纬向平均残差的最大绝对值，以及初始动力学 `ps` tendency 的最大值和面积平均值（Pa/s，含配置的压力扩散，不含后续蒸发/降水）。完整的逐纬度带符号平均残差位于 `model.initial_balance`，可导出 JSON：

```bash
python scripts/check_balance.py --output output/initial_balance.json
# 较小 CPU 网格检查并试走 100 步：
python scripts/check_balance.py --backend cpu --nlat 72 --nlon 144 --steps 100
```

若受限 Windows 环境的默认 CuPy 缓存目录不可写，可先在 PowerShell 中设置工作区内缓存目录：`$env:CUPY_CACHE_DIR = "$PWD/output/cupy-cache"`。

检查作用于加入扰动后的实际初值。任一层超过阈值或诊断出现非有限值，首次积分会被拒绝；检查脚本也会以失败状态退出并保留诊断文件。这是所列动力残差的检查，不能替代真实资料的完整平衡分析。压力梯度与科氏力在时间推进时联合中点求解，以减少分步旋转对平衡的破坏。

## 极区稳定化与守恒记录

原始方程不再分别平滑 `u/v/T/q` 状态。每步用相同的纬向滤波算子联合处理柱质量增量与质量加权的动量、温度、比湿增量，再以更新后的柱质量恢复变量。纬圈质量及质量加权量在浮点误差内守恒，平衡背景不被直接平滑。经向通量散度使用真实球面面积因子；`cos_clamp` 仅继续保护纬向网格尺度，曲率项使用真实 `tan(latitude)/a`。

`numerics.polar_divergence_damping_m2s` 默认 `2.0e5`，对高纬质量加权纬向平均风的球面散度施加负伴随阻尼，抑制包括交错格点模式在内的轴对称扰动；采用步长相关的显式强度上限。它不对温度、湿度做普通经向平滑。设为 0 可关闭该阻尼。

`model.stabilization_budget` 记录最近一步阻尼和滤波各自造成的全球质量、轴向相对角动量、经向动量、动能及总能量变化；`stabilization_budget_total` 保存累计变化。差值先逐格计算再全球求和，动能使用速度平方差的乘积展开，避免两个巨大全球总量相减吞掉小变化。能量按各层分段等温静力解积分，包含内能 `Cv*T`、潜热 `Lv*q`、动能、重力位能及有限顶层气压对应的位势修正；内能采用模型的常数干空气比热。它用于稳定化审计，不代表包含所有物理过程与边界功的完整能量闭合。变量裁剪及其他物理过程的变化不计入这两项审计。

`[stabilization]` 是定期审计输出，不是错误报警。`numerics.diagnostic_interval_steps` 默认 60，控制它和 `[stability]` 的采样间隔。后者检查裁剪前的各层最大风速及纬度、高纬散度 RMS 和纬向平均散度 RMS、温度/地面气压范围、动力学地面气压 tendency、平流 Courant 数、扩散数以及超出变量上下限的格点比例；压力还检查两个较早阶段的未裁剪候选值。`stabilization_power_wm2` 把单步总能量变化换算为全球平均功率密度，便于判断量级。

`status: bounds_exceeded` 表示采样时存在越界格点，应查看 `outside_bounds_fraction` 中具体哪个变量越界；不能仅看最终已裁剪的风速。`cfl_exceeded` 表示平流 Courant 数超过 1 或扩散数超过 0.5。`no_limit_exceeded` 仅表示本次采样没有触及这些限制，不能保证重力波或长期非线性稳定性。采样发现非有限值会在最终裁剪前报错；间隔内的瞬时越界仍可能未被采到，排查时可把间隔设为 1。

`scripts/check_balance.py` 将每次采样写入输出文件同名的 `.jsonl`，并把末次诊断和累计稳定化变化写入 `.json`，例如 `--steps 601 --output output/stability.json`。旧版本正在运行的进程不会自动加载这些代码修改，需重启后才能得到新诊断。

这仍是规则经纬网格上的短期处理。在 `720×1440` 下不能简单取消 `cos_clamp`；接近极点的真实纬向格距会带来更严格的 CFL 限制。极区 Fourier 截断、纬向通量合并/reduced grid 或准均匀网格尚未实现，长时间高分辨率稳定性需要单独验证。

## 目录

```
sim/                 物理与引擎(config/backend/topo/physics/primitive/balance/model 等)
server.py            FastAPI + WebSocket 服务
run.py               命令行入口 (serve / precompute / benchmark)
web/                 前端 (index.html / style.css / app.js)
data/etopo20.npz     真实地形
config.yaml          超参数
```

## 已知简化

默认是静力原始方程，另保留旧多层浅水模式；海洋为平板/可选双层简化，云和降水采用诊断方案。适合教学演示与大尺度环流实验，不用于真实预报。
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

## Atmospheric dynamics core

The default `physics.dynamics.core: primitive_equations` integrates the moist
hydrostatic primitive equations in a terrain-following sigma coordinate:

- surface pressure is prognostic and represents total atmospheric column mass;
- layer pressure and geopotential are diagnosed from hydrostatic balance;
- temperature includes pressure-work (`kappa T omega / p`);
- the same interface mass flux transports heat, moisture, and momentum;
- sigma mass flux is exactly zero at the ground and fixed-pressure model top;
- terrain enters through surface geopotential, hydrostatic pressure reduction,
  the transformed pressure-gradient force, and physical surface stress.

The lower geometric boundary follows `w = u grad(zs)`, while its coordinate
normal velocity is zero. Empirical mountain lift/blocking is disabled in this
core to avoid counting terrain forcing twice. The former layered moist
shallow-water solver is retained as
`physics.dynamics.core: shallow_water` for comparison and compatibility.

This is a hydrostatic core. It does not resolve nonhydrostatic acoustic or
convective vertical momentum, so kilometre-scale cloud-resolving simulations
would still require a separate nonhydrostatic dynamical core and microphysics.
