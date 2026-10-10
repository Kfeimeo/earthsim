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
RD = 287.0
RV = 461.5
KAPPA = RD / CP
RHO_A = 1.2
P0 = 1.0e5
MCOL = 1.0e4          # 大气柱质量 kg/m^2
RHO_W, CW = 1000.0, 4186.0
C_LAND = 2.5e6        # 陆面薄层热容 J/m^2/K


class Ops:
    """给定网格的球面差分算子集合(预计算度量项)。"""

    def __init__(self, xp, lats_deg, nlon, cos_clamp=0.2,
                 pf_lat=65.0, pf_passes=6, use_cuda_kernel=False,
                 lons_deg=None, advection_scheme="upwind",
                 advection_limiter="mc"):
        self.xp = xp
        nlat = len(lats_deg)
        self.nlat, self.nlon = nlat, nlon
        lat = xp.asarray(_np.radians(lats_deg), dtype=xp.float32)[:, None]
        self.lat = lat
        if lons_deg is None:
            lons_deg = _np.arange(nlon, dtype=_np.float32) * (360.0 / nlon)
        self.lon_rad = xp.asarray(
            _np.radians(lons_deg), dtype=xp.float32)[None, :]
        dlon = 2 * _np.pi / nlon
        dlat = _np.pi / nlat
        self.coslat = xp.cos(lat).astype(xp.float32)
        cosl = xp.maximum(self.coslat, cos_clamp)
        # Only zonal lengths are regularized. Meridional flux divergence must
        # divide by the actual cell area to conserve spherical global mass.
        self.invcoslat = (1.0 / self.coslat).astype(xp.float32)
        self.dx = (A_EARTH * cosl * dlon).astype(xp.float32)   # [nlat,1]
        self.dy = _np.float32(A_EARTH * dlat)
        self.invdx = (1.0 / self.dx).astype(xp.float32)
        self.invdy = _np.float32(1.0 / self.dy)
        self.f = (2 * OMEGA * xp.sin(lat)).astype(xp.float32)
        self.tanl = xp.tan(lat).astype(xp.float32) / A_EARTH
        self.advection_scheme = str(advection_scheme).lower()
        self.advection_limiter = str(advection_limiter).lower()
        north_face_mask = _np.ones((nlat, 1), dtype=_np.float32)
        north_face_mask[-1] = 0.0
        self.north_face_mask = xp.asarray(north_face_mask)
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
                self.coslat_flat = self.coslat[:, 0].copy()
                self.invcoslat_flat = self.invcoslat[:, 0].copy()
                self.pf_w_flat = self.pf_w[:, 0].copy()

    # ---------- 基础算子 ----------
    def rollx(self, F, k):
        return self.xp.roll(F, k, axis=-1)

    def shifty(self, F, k):
        """纬向平移, 边界复制。k=+1: 取南侧值。"""
        xp = self.xp
        if k == 1:
            return xp.concatenate([F[..., :1, :], F[..., :-1, :]], axis=-2)
        return xp.concatenate([F[..., 1:, :], F[..., -1:, :]], axis=-2)

    def ddx(self, F):
        if self.cuda_adv is not None:
            return self.cuda_adv.gradient(
                F, self.invdx_flat, float(self.invdy))[0]
        return (self.rollx(F, -1) - self.rollx(F, 1)) * (0.5 * self.invdx)

    def ddy(self, F):
        if self.cuda_adv is not None:
            return self.cuda_adv.gradient(
                F, self.invdx_flat, float(self.invdy))[1]
        return (self.shifty(F, -1) - self.shifty(F, 1)) * (0.5 * self.invdy)

    def gradient(self, F):
        if self.cuda_adv is not None:
            return self.cuda_adv.gradient(
                F, self.invdx_flat, float(self.invdy))
        return self.ddx(F), self.ddy(F)

    def divergence(self, u, v):
        if self.cuda_adv is not None:
            return self.cuda_adv.divergence(
                u, v, self.invdx_flat, float(self.invdy),
                self.coslat_flat, self.invcoslat_flat)
        return self.ddx(u) + self.ddy(v * self.coslat) * self.invcoslat

    def lap(self, F):
        if self.cuda_adv is not None and F.dtype == self.xp.float32:
            return self.cuda_adv.laplacian(
                F, self.invdx_flat, float(self.invdy))
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

    def _limited_slope(self, backward, forward):
        """Return a TVD-limited, dimensionless cell slope."""
        xp = self.xp
        same_sign = backward * forward > 0.0
        if self.advection_limiter == "minmod":
            magnitude = xp.minimum(xp.abs(backward), xp.abs(forward))
            return xp.where(same_sign, xp.sign(backward) * magnitude, 0.0)
        if self.advection_limiter == "vanleer":
            denominator = backward + forward
            slope = 2.0 * backward * forward / xp.where(
                xp.abs(denominator) > 1.0e-30, denominator, 1.0)
            return xp.where(same_sign, slope, 0.0)
        if self.advection_limiter == "mc":
            centred = 0.5 * (backward + forward)
            magnitude = xp.minimum(
                xp.abs(centred),
                xp.minimum(2.0 * xp.abs(backward),
                           2.0 * xp.abs(forward)))
            return xp.where(same_sign, xp.sign(centred) * magnitude, 0.0)
        raise ValueError(
            f"unsupported MUSCL limiter: {self.advection_limiter}")

    def finite_volume_divergence(self, flux_u, flux_v):
        """Spherical control-volume divergence from centred face fluxes."""
        xp = self.xp
        east_face = 0.5 * (flux_u + self.rollx(flux_u, -1))
        west_face = self.rollx(east_face, 1)

        weighted_v = flux_v * self.coslat
        north_face = 0.5 * (
            weighted_v + self.shifty(weighted_v, -1))
        north_face = north_face * self.north_face_mask
        south_face = xp.concatenate(
            [xp.zeros_like(north_face[..., :1, :]),
             north_face[..., :-1, :]], axis=-2)
        return ((east_face - west_face) * self.invdx
                + (north_face - south_face) * self.invdy * self.invcoslat)

    def muscl_flux_divergence(self, F, flux_u, flux_v):
        """Divergence of ``flux * F`` using MUSCL/TVD face states."""
        xp = self.xp

        west = self.rollx(F, 1)
        east = self.rollx(F, -1)
        slope_x = self._limited_slope(F - west, east - F)
        slope_east = self.rollx(slope_x, -1)
        flux_east = 0.5 * (flux_u + self.rollx(flux_u, -1))
        state_east = xp.where(
            flux_east >= 0.0,
            F + 0.5 * slope_x,
            east - 0.5 * slope_east)
        transported_east = flux_east * state_east
        transported_west = self.rollx(transported_east, 1)

        south = self.shifty(F, 1)
        north = self.shifty(F, -1)
        slope_y = self._limited_slope(F - south, north - F)
        slope_north = self.shifty(slope_y, -1)
        weighted_v = flux_v * self.coslat
        flux_north = 0.5 * (
            weighted_v + self.shifty(weighted_v, -1))
        flux_north = flux_north * self.north_face_mask
        state_north = xp.where(
            flux_north >= 0.0,
            F + 0.5 * slope_y,
            north - 0.5 * slope_north)
        transported_north = flux_north * state_north
        transported_south = xp.concatenate(
            [xp.zeros_like(transported_north[..., :1, :]),
             transported_north[..., :-1, :]], axis=-2)

        return ((transported_east - transported_west) * self.invdx
                + (transported_north - transported_south)
                * self.invdy * self.invcoslat)

    def muscl_adv(self, F, u, v):
        """MUSCL/TVD approximation of the advective tendency ``-V.grad(F)``."""
        return (-self.muscl_flux_divergence(F, u, v)
                + F * self.finite_volume_divergence(u, v))

    def adv_diff_step(self, F, u, v, K, dt):
        """F += dt*(adv + K lap)。GPU 下走手写 CUDA kernel。"""
        if self.cuda_adv is not None:
            return self.cuda_adv.adv_diff(
                F, u, v, self.invdx_flat, float(self.invdy), float(K),
                float(dt), scheme=self.advection_scheme,
                limiter=self.advection_limiter,
                coslat=self.coslat_flat,
                invcoslat=self.invcoslat_flat)
        if F.ndim > 2:
            return self.xp.stack([
                self.adv_diff_step(Fk, uk, vk, K, dt)
                for Fk, uk, vk in zip(F, u, v)
            ], axis=0)
        if self.advection_scheme == "muscl_tvd":
            advective = self.muscl_adv(F, u, v)
        else:
            advective = self.upwind_adv(F, u, v)
        return F + dt * (advective + K * self.lap(F))

    def polar_filter(self, F):
        """高纬纬向 1-2-1 平滑, 抑制极点数值噪声。"""
        # Moist mass increments can be promoted to float64. Keep that
        # precision via array operations rather than passing doubles to the
        # float32-only raw CUDA kernel (or silently reinterpreting them).
        if self.cuda_adv is not None and F.dtype == self.xp.float32:
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
def qsat(xp, T, pressure=P0):
    """Return saturation specific humidity at temperature and pressure.

    ``pressure`` may be a scalar or an array in Pa.  The denominator uses the
    exact moist-air conversion rather than assuming every model level is at
    1000 hPa.
    """
    es = 610.78 * xp.exp(17.27 * (T - 273.15) / xp.maximum(T - 35.85, 1.0))
    p = xp.maximum(pressure, es + 1.0)
    return 0.622 * es / xp.maximum(p - 0.378 * es, 1.0)


def insolation(xp, lat_rad, lon_rad, t_utc, S0, diurnal=True):
    """瞬时大气顶入射 (W/m^2) 与太阳直射点。

    t_utc: datetime。返回 (Q[nlat,nlon], subsolar_lat, subsolar_lon)。
    """
    doy = t_utc.timetuple().tm_yday
    frac = (t_utc.hour + t_utc.minute / 60 + t_utc.second / 3600) / 24.0
    decl = _np.radians(23.44) * _np.sin(2 * _np.pi * (doy - 80) / 365.25)
    sub_lon = (180.0 - 360.0 * frac) % 360.0  # 正午对应太阳直射经度
    lam = lon_rad
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
