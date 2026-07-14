"""物理过程与球面数值算子。

所有函数以 xp (numpy 或 cupy) 数组编写, CPU/GPU 共用一份代码;
在 CUDA 后端下, 标量平流+扩散走 kernels.cu 的手写融合 kernel。

模型: 多层湿浅水大气 + 平板海洋
  u, v   各高度层水平风场 (m/s)
  w      垂直速度 (m/s)，由质量连续、浮力和地形抬升共同驱动
  h      各层等效厚度/位势 (m), 兼作气压场 (暖->低压 热力强迫)
  Ta     各高度层气温 (K)
  q      各高度层比湿 (kg/kg)
  Ts     下垫面温度 (海表 SST / 陆面) (K)
  uo, vo 洋流 (m/s)
诊断量: cloud(云量), precip(降水 mm/h), ice/snow, rh
"""
import numpy as _np

A_EARTH = 6.371e6
OMEGA = 7.292e-5
SIGMA = 5.67e-8
CP = 1004.0
LV = 2.5e6
RHO_A = 1.2
P0 = 1.0e5
MCOL = 1.0e4          # 大气柱质量 kg/m^2
RHO_W, CW = 1000.0, 4186.0
C_LAND = 2.5e6        # 陆面薄层热容 J/m^2/K


class Ops:
    """给定网格的球面差分算子集合(预计算度量项)。"""

    def __init__(self, xp, lats_deg, nlon, cos_clamp=0.2,
                 pf_lat=65.0, pf_passes=6, use_cuda_kernel=False):
        self.xp = xp
        nlat = len(lats_deg)
        self.nlat, self.nlon = nlat, nlon
        lat = xp.asarray(_np.radians(lats_deg), dtype=xp.float32)[:, None]
        self.lat = lat
        dlon = 2 * _np.pi / nlon
        dlat = _np.pi / nlat
        cosl = xp.maximum(xp.cos(lat), cos_clamp)
        self.dx = (A_EARTH * cosl * dlon).astype(xp.float32)   # [nlat,1]
        self.dy = _np.float32(A_EARTH * dlat)
        self.invdx = (1.0 / self.dx).astype(xp.float32)
        self.invdy = _np.float32(1.0 / self.dy)
        self.f = (2 * OMEGA * xp.sin(lat)).astype(xp.float32)
        self.tanl = xp.clip(xp.tan(lat), -3.0, 3.0).astype(xp.float32) / A_EARTH
        # 极区纬向滤波权重
        absd = _np.abs(lats_deg)
        w = _np.clip((absd - pf_lat) / (89.0 - pf_lat), 0, 1) ** 2
        self.pf_w = xp.asarray(w, dtype=xp.float32)[:, None]
        self.pf_passes = pf_passes
        # GPU 融合 kernel
        self.cuda_adv = None
        if use_cuda_kernel:
            from . import cuda_kernels
            if cuda_kernels.load():
                self.cuda_adv = cuda_kernels
                self.invdx_flat = self.invdx[:, 0].copy()
                self.pf_w_flat = self.pf_w[:, 0].copy()

    # ---------- 基础算子 ----------
    def rollx(self, F, k):
        return self.xp.roll(F, k, axis=1)

    def shifty(self, F, k):
        """纬向平移, 边界复制。k=+1: 取南侧值。"""
        xp = self.xp
        if k == 1:
            return xp.concatenate([F[:1], F[:-1]], axis=0)
        return xp.concatenate([F[1:], F[-1:]], axis=0)

    def ddx(self, F):
        return (self.rollx(F, -1) - self.rollx(F, 1)) * (0.5 * self.invdx)

    def ddy(self, F):
        return (self.shifty(F, -1) - self.shifty(F, 1)) * (0.5 * self.invdy)

    def lap(self, F):
        return ((self.rollx(F, 1) + self.rollx(F, -1) - 2 * F) * self.invdx ** 2
                + (self.shifty(F, 1) + self.shifty(F, -1) - 2 * F) * self.invdy ** 2)

    def upwind_adv(self, F, u, v):
        """一阶迎风平流项 -u dF/dx - v dF/dy 的负值已含。返回 tendency。"""
        xp = self.xp
        dxm = (F - self.rollx(F, 1)) * self.invdx
        dxp = (self.rollx(F, -1) - F) * self.invdx
        dym = (F - self.shifty(F, 1)) * self.invdy
        dyp = (self.shifty(F, -1) - F) * self.invdy
        return -(u * xp.where(u > 0, dxm, dxp) + v * xp.where(v > 0, dym, dyp))

    def adv_diff_step(self, F, u, v, K, dt):
        """F += dt*(adv + K lap)。GPU 下走手写 CUDA kernel。"""
        if self.cuda_adv is not None:
            return self.cuda_adv.adv_diff(F, u, v, self.invdx_flat,
                                          float(self.invdy), float(K), float(dt))
        return F + dt * (self.upwind_adv(F, u, v) + K * self.lap(F))

    def polar_filter(self, F):
        """高纬纬向 1-2-1 平滑, 抑制极点数值噪声。"""
        if self.cuda_adv is not None:
            return self.cuda_adv.polar_filter(F, self.pf_w_flat,
                                              self.pf_passes)
        s = F
        for _ in range(self.pf_passes):
            s = (0.25 * self.xp.roll(s, 1, axis=-1) + 0.5 * s
                 + 0.25 * self.xp.roll(s, -1, axis=-1))
        weight_shape = (1,) * (F.ndim - 2) + (self.nlat, 1)
        return F + self.pf_w.reshape(weight_shape) * (s - F)

    def coriolis_rotate(self, u, v, dt):
        """科氏力: 精确旋转(无条件稳定)。北半球向右偏转。"""
        th = self.f * dt
        c, s = self.xp.cos(th), self.xp.sin(th)
        return u * c + v * s, v * c - u * s


# ---------- 热力学工具 ----------
def qsat(xp, T):
    """饱和比湿 (Tetens 公式)。"""
    es = 610.78 * xp.exp(17.27 * (T - 273.15) / xp.maximum(T - 35.85, 1.0))
    return 0.622 * es / P0


def insolation(xp, lat_rad, lons_deg, t_utc, S0, diurnal=True):
    """瞬时大气顶入射 (W/m^2) 与太阳直射点。

    t_utc: datetime。返回 (Q[nlat,nlon], subsolar_lat, subsolar_lon)。
    """
    doy = t_utc.timetuple().tm_yday
    frac = (t_utc.hour + t_utc.minute / 60 + t_utc.second / 3600) / 24.0
    decl = _np.radians(23.44) * _np.sin(2 * _np.pi * (doy - 80) / 365.25)
    sub_lon = (180.0 - 360.0 * frac) % 360.0  # 正午对应太阳直射经度
    lam = xp.asarray(_np.radians(lons_deg), dtype=xp.float32)[None, :]
    if diurnal:
        hour_ang = lam - _np.radians(sub_lon)
        cosz = (xp.sin(lat_rad) * _np.sin(decl)
                + xp.cos(lat_rad) * _np.cos(decl) * xp.cos(hour_ang))
        Q = S0 * xp.maximum(cosz, 0.0)
    else:  # 日平均
        h0 = xp.arccos(xp.clip(-xp.tan(lat_rad) * _np.tan(decl), -1, 1))
        Q = (S0 / _np.pi) * (h0 * xp.sin(lat_rad) * _np.sin(decl)
                             + xp.cos(lat_rad) * _np.cos(decl) * xp.sin(h0))
        Q = Q * xp.ones_like(lam)
    return Q.astype(xp.float32), _np.degrees(decl), sub_lon
