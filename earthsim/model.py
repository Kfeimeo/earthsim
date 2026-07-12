"""EarthModel: 状态管理 + 时间积分。"""
import datetime as _dt
import numpy as _np

from . import topo as _topo
from .backend import init_backend, to_cpu
from .physics import (Ops, qsat, insolation, A_EARTH,
                      SIGMA, CP, LV, RHO_A, MCOL, RHO_W, CW, C_LAND)


class EarthModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.xp, self.backend = init_backend(cfg.backend)
        xp = self.xp
        g = cfg.grid
        self.nlat, self.nlon = int(g.nlat), int(g.nlon)
        self.lats, self.lons = _topo.sim_grid(self.nlat, self.nlon)
        elev, land = _topo.load_topo(g.topo_file, self.nlat, self.nlon)
        self.elev = xp.asarray(elev)
        self.land = xp.asarray(land)          # 1=陆地
        self.ocean = 1.0 - self.land
        n = cfg.numerics
        self.ops = Ops(xp, self.lats, self.nlon,
                       cos_clamp=n.cos_clamp, pf_lat=n.polar_filter_lat,
                       pf_passes=int(n.polar_filter_passes),
                       use_cuda_kernel=(self.backend == "cuda"))
        self.dt = float(cfg.time.dt)
        self.t = _dt.datetime.fromisoformat(str(cfg.time.start))
        self.step_count = 0
        _, d0, l0 = insolation(self.xp, self.ops.lat, self.lons, self.t,
                               cfg.physics.S0, cfg.physics.diurnal_cycle)
        self.subsolar = (float(d0), float(l0))
        self._init_state()

    # ------------------------------------------------------------
    def _init_state(self):
        xp, p = self.xp, self.cfg.physics
        lat = self.ops.lat
        f32 = xp.float32
        Teq = (315.0 - 42.0 * xp.sin(lat) ** 2) * xp.ones((1, self.nlon))
        self.Ta = Teq.astype(f32).copy()
        # 海表: 不低于 -2C; 陆面随高度递减
        self.Ts = (Teq - 0.0065 * xp.maximum(self.elev, 0) * self.land)
        self.Ts = xp.where((self.ocean > 0.5) & (self.Ts < 271.2),
                           271.2, self.Ts).astype(f32)
        self.q = (0.65 * qsat(xp, self.Ta)).astype(f32)
        self.h = (p.H0 - p.beta_T * (self.Ta - self.Ta.mean())).astype(f32)
        z = xp.zeros((self.nlat, self.nlon), f32)
        self.u, self.v = z.copy(), z.copy()
        self.uo, self.vo = z.copy(), z.copy()
        self.cloud, self.precip = z.copy(), z.copy()
        self.ice, self.snow = z.copy(), z.copy()
        self._diag_surface()

    # ------------------------------------------------------------
    def _diag_surface(self):
        """冰/雪诊断(由温度决定)。"""
        xp, p = self.xp, self.cfg.physics
        if p.ice_albedo:
            self.ice = (self.ocean *
                        xp.clip((271.4 - self.Ts) / 2.0, 0, 1)).astype(xp.float32)
            self.snow = (self.land *
                         xp.clip((273.5 - self.Ts) / 6.0, 0, 1)).astype(xp.float32)
        else:
            self.ice = xp.zeros_like(self.Ts)
            self.snow = xp.zeros_like(self.Ts)

    # ------------------------------------------------------------
    def step(self, nsteps=1):
        for _ in range(int(nsteps)):
            self._step_once()

    def _step_once(self):
        xp, p, o, dt = self.xp, self.cfg.physics, self.ops, self.dt
        u, v, h, Ta, q, Ts = self.u, self.v, self.h, self.Ta, self.q, self.Ts
        spd = xp.sqrt(u * u + v * v) + 1.0

        # ============ 辐射 ============
        Q_sw, decl, sub_lon = insolation(xp, o.lat, self.lons, self.t,
                                         p.S0, p.diurnal_cycle)
        self.subsolar = (float(decl), float(sub_lon))
        qs_a = qsat(xp, Ta)
        rh = xp.clip(q / qs_a, 0, 1.3)
        gq = xp.clip(q / 0.02, 0, 1)

        if p.radiation:
            alb_sfc = (self.ocean * (p.alb_ocean * (1 - self.ice) + p.alb_ice * self.ice)
                       + self.land * (p.alb_land * (1 - self.snow) + p.alb_snow * self.snow))
            alb = xp.clip(alb_sfc + p.alb_cloud * self.cloud, 0, 0.85)
            SW_sfc = Q_sw * (1 - alb) * 0.80          # 20% 被大气吸收/散射
            SW_air = Q_sw * (1 - 0.30 * self.cloud) * 0.18
            LW_up = 0.98 * SIGMA * Ts ** 4
            eps_dn = xp.clip(0.60 + 0.25 * gq + 0.15 * self.cloud, 0, 0.98)
            LW_dn = eps_dn * SIGMA * Ta ** 4
            olr_f = xp.clip(0.62 - 0.20 * gq - 0.08 * self.cloud, 0.28, 0.62)
            OLR_air = olr_f * SIGMA * Ta ** 4
        else:
            SW_sfc = SW_air = LW_up = LW_dn = OLR_air = xp.zeros_like(Ta)

        # ============ 地表通量 ============
        SH = 1.2e-3 * RHO_A * CP * spd * (Ts - Ta)
        if p.moisture:
            evap_eff = self.ocean + p.land_evap * self.land
            E = (p.c_evap * RHO_A * spd * evap_eff *
                 xp.maximum(qsat(xp, Ts) - q, 0))          # kg/m^2/s
            LE = LV * E
        else:
            E = LE = xp.zeros_like(Ta)

        # 下垫面热容: 海洋混合层 vs 陆面薄层
        C_sfc = self.ocean * (RHO_W * CW * p.mld) + self.land * C_LAND
        Ts = Ts + dt * (SW_sfc + LW_dn - LW_up - SH - LE) / C_sfc
        # 海冰下海水温度下限
        Ts = xp.where(self.ocean > 0.5, xp.maximum(Ts, 268.0), Ts)

        # ============ 大气热力 + 水汽 ============
        Ta = o.adv_diff_step(Ta, u, v, p.diff_T, dt)
        Ta = Ta + dt * (SH + SW_air + 0.85 * (LW_up - LW_dn) - OLR_air) / (MCOL * CP)
        if p.radiation:  # 向辐射平衡弱弛豫(保底约束, 防漂移)
            Teq = 302.0 - 42.0 * xp.sin(o.lat) ** 2 + 8.0 * xp.sin(o.lat) * _np.sin(_np.radians(self.subsolar[0]) * 2)
            Ta = Ta + dt / p.tau_relax_T * (Teq - Ta)

        # 风场辐散 (动力学与凝结共用)
        div = o.ddx(u) + o.ddy(v * xp.cos(o.lat)) / xp.maximum(xp.cos(o.lat), 0.2)

        if p.moisture:
            q = o.adv_diff_step(q, u, v, p.diff_q, dt)
            q = q + dt * E / MCOL
            qs_a = qsat(xp, Ta)
            # 辐合区(低压)对流增强: 有效凝结阈值降低 -> 降水集中于 ITCZ/气旋
            rh_eff = p.rh_crit * (1 - xp.clip(2.5e4 * xp.maximum(-div, 0), 0, 0.3))
            exc = xp.maximum(q - rh_eff * qs_a, 0)
            # 辐散区(高压)下沉干燥 -> 副热带晴空/沙漠带
            q = q * (1 - dt * 0.7 * xp.clip(div, 0, 1e-4))
            dq = exc * (1 - _np.exp(-dt / p.tau_cond))
            q = q - dq
            Ta = Ta + (LV / CP) * dq                      # 凝结潜热
            self.precip = (dq * MCOL / dt * 3600.0).astype(xp.float32)  # mm/h
            rh = xp.clip(q / qsat(xp, Ta), 0, 1.3)
            cl = xp.clip((rh - 0.58) / 0.32, 0, 1) ** 1.4
            cl = cl * (1 - xp.clip(1.5e5 * div, 0, 0.45))  # 下沉抑制云
            cl = xp.maximum(cl, xp.clip(self.precip / 2.0, 0, 1))
            self.cloud = (0.7 * self.cloud + 0.3 * cl).astype(xp.float32)  # 云记忆
            q = xp.clip(q, 0, 0.05)

        # ============ 动力学 (湿浅水) ============
        h_eq = p.H0 - p.beta_T * (Ta - Ta.mean())
        h = o.adv_diff_step(h, u, v, p.visc * 0.5, dt)
        h = h + dt * (-h * div + (h_eq - h) / p.tau_h)
        h = xp.clip(h, 0.4 * p.H0, 1.8 * p.H0)

        dhdx, dhdy = o.ddx(h), o.ddy(h)
        u = o.adv_diff_step(u, u, v, p.visc, dt)
        v = o.adv_diff_step(v, u, v, p.visc, dt)
        u = u + dt * (-p.g_eff * dhdx - p.drag * u + o.tanl * u * v)
        v = v + dt * (-p.g_eff * dhdy - p.drag * v - o.tanl * u * u)
        u, v = o.coriolis_rotate(u, v, dt)
        u = xp.clip(u, -p.umax, p.umax)
        v = xp.clip(v, -p.umax, p.umax)

        # ============ 海洋 ============
        if p.ocean:
            uo, vo = self.uo, self.vo
            uo = uo + dt * (p.tau_ocean * spd * u - p.drag_ocean * uo) \
                 + dt * p.visc_ocean * o.lap(uo)
            vo = vo + dt * (p.tau_ocean * spd * v - p.drag_ocean * vo) \
                 + dt * p.visc_ocean * o.lap(vo)
            uo, vo = o.coriolis_rotate(uo, vo, dt * 0.15)  # 埃克曼偏转(弱化)
            open_o = self.ocean * (1 - self.ice)
            self.uo = (xp.clip(uo, -2, 2) * open_o).astype(xp.float32)
            self.vo = (xp.clip(vo, -2, 2) * open_o).astype(xp.float32)
            # 洋流输送热量
            Ts = xp.where(self.ocean > 0.5,
                          o.adv_diff_step(Ts, self.uo, self.vo, 2e3, dt), Ts)

        # ============ 极区滤波 + 限幅 ============
        u, v, h = o.polar_filter(u), o.polar_filter(v), o.polar_filter(h)
        Ta, q = o.polar_filter(Ta), o.polar_filter(q)
        self.u, self.v = u.astype(xp.float32), v.astype(xp.float32)
        self.h = h.astype(xp.float32)
        self.Ta = xp.clip(Ta, 160, 345).astype(xp.float32)
        self.q = q.astype(xp.float32)
        self.Ts = xp.clip(Ts, 170, 350).astype(xp.float32)
        self._diag_surface()

        self.t += _dt.timedelta(seconds=self.dt)
        self.step_count += 1

    # ------------------------------------------------------------
    def apply_temp_edit(self, lat_deg, lon_deg, radius_km=800.0,
                        delta=5.0, target="both"):
        """在 (lat, lon) 为中心、radius_km 为尺度的高斯区域内
        增减温度 delta (K)。target: surface / air / both。"""
        xp = self.xp
        la0 = _np.radians(float(lat_deg))
        lo0 = _np.radians(float(lon_deg) % 360.0)
        la = self.ops.lat                                  # [nlat,1] 弧度
        lo = xp.asarray(_np.radians(self.lons),
                        dtype=xp.float32)[None, :]         # [1,nlon]
        cosd = (xp.sin(la) * _np.sin(la0) +
                xp.cos(la) * _np.cos(la0) * xp.cos(lo - lo0))
        d_km = xp.arccos(xp.clip(cosd, -1.0, 1.0)) * (A_EARTH / 1000.0)
        w = xp.exp(-(d_km / max(float(radius_km), 50.0)) ** 2)
        dT = (float(delta) * w).astype(xp.float32)
        if target in ("surface", "both"):
            self.Ts = xp.clip(self.Ts + dT, 170, 350).astype(xp.float32)
            # 变暖立即消融冰雪 / 骤冷时下一步会重新诊断
            if float(delta) > 0:
                melt = xp.clip((self.Ts - 271.4) / 2.0, 0, 1)
                self.ice = xp.minimum(self.ice, 1 - melt * self.ocean)
                melt_l = xp.clip((self.Ts - 273.5) / 6.0, 0, 1)
                self.snow = xp.minimum(self.snow, 1 - melt_l * self.land)
        if target in ("air", "both"):
            self.Ta = xp.clip(self.Ta + dT, 160, 345).astype(xp.float32)
            # 保持相对湿度不突变: 随饱和比湿同步缩放水汽
            self.q = xp.clip(self.q * qsat(xp, self.Ta) /
                             qsat(xp, self.Ta - dT), 0, 0.04).astype(xp.float32)

    # ------------------------------------------------------------
    def pressure_hpa(self):
        """把厚度场映射为习惯的海平面气压 (hPa), 仅用于展示。"""
        return 1013.0 + (self.h - self.cfg.physics.H0) * 0.045

    def fields_cpu(self):
        """导出全部展示字段到 numpy。"""
        return {
            "press": to_cpu(self.pressure_hpa()),
            "temp": to_cpu(self.Ta) - 273.15,
            "sst": to_cpu(self.Ts) - 273.15,
            "hum": to_cpu(self.q) * 1000.0,          # g/kg
            "cloud": to_cpu(self.cloud),
            "precip": to_cpu(self.precip),
            "ice": to_cpu(self.xp.maximum(self.ice, self.snow)),
            "u": to_cpu(self.u), "v": to_cpu(self.v),
            "uo": to_cpu(self.uo), "vo": to_cpu(self.vo),
        }

    def check_health(self):
        import numpy as np
        f = self.fields_cpu()
        for k, a in f.items():
            if not np.isfinite(a).all():
                raise FloatingPointError(f"字段 {k} 出现 NaN/Inf")
        return f
