"""EarthModel: 状态管理 + 时间积分。"""
import datetime as _dt
import numpy as _np
import time as _time

from . import topo as _topo
from .backend import init_backend, to_cpu
from .data_loader import RealDataError, load_real_initialization
from .physics import (Ops, qsat, insolation, A_EARTH,
                      SIGMA, CP, LV, RD, RHO_A, MCOL,
                      RHO_W, CW, C_LAND)
from .primitive import (validate_sigma_interfaces, sigma_to_hybrid,
                        generate_hybrid_coefficients,
                        validate_hybrid_coefficients,
                        standard_atmosphere_height, hybrid_pressures, exner,
                        hydrostatic_state, hybrid_mass_flux,
                        mass_consistent_transport,
                        pressure_vertical_velocity, sponge_profile,
                        column_total_energy, implicit_vertical_diffusion,
                        dry_convective_adjustment)


class EarthModel:
    def __init__(self, cfg):
        init_started = _time.perf_counter()
        self.cfg = cfg
        stage_started = _time.perf_counter()
        self.xp, self.backend = init_backend(cfg.backend)
        print(f"[startup] backend initialized ({self.backend}): "
              f"{_time.perf_counter() - stage_started:.3f}s", flush=True)
        xp = self.xp
        g = cfg.grid
        self.nlat, self.nlon = int(g.nlat), int(g.nlon)
        self.lats, self.lons = _topo.sim_grid(self.nlat, self.nlon)
        topo_spec = g.get("topo_files", g.get("topo_file", ""))
        stage_started = _time.perf_counter()
        elev, land = _topo.load_topo(topo_spec, self.nlat, self.nlon)
        print(f"[startup] model topography ready ({self.nlat}x{self.nlon}): "
              f"{_time.perf_counter() - stage_started:.3f}s", flush=True)
        stage_started = _time.perf_counter()
        self.elev = xp.asarray(elev)
        self.land = xp.asarray(land)          # 1=陆地
        self.ocean = 1.0 - self.land
        print(f"[startup] topography copied to {self.backend}: "
              f"{_time.perf_counter() - stage_started:.3f}s", flush=True)
        n = cfg.numerics
        self.advection_scheme = str(n.advection).lower()
        if self.advection_scheme in {"muscl", "tvd"}:
            self.advection_scheme = "muscl_tvd"
        if self.advection_scheme not in {"upwind", "muscl_tvd"}:
            raise ValueError(
                "numerics.advection must be 'upwind' or 'muscl_tvd'")
        self.advection_limiter = str(n.get("advection_limiter", "mc")).lower()
        if self.advection_limiter not in {"minmod", "vanleer", "mc"}:
            raise ValueError(
                "numerics.advection_limiter must be minmod, vanleer, or mc")
        stage_started = _time.perf_counter()
        self.ops = Ops(xp, self.lats, self.nlon,
                       cos_clamp=n.cos_clamp, pf_lat=n.polar_filter_lat,
                       pf_passes=int(n.polar_filter_passes),
                       use_cuda_kernel=(self.backend == "cuda"),
                       lons_deg=self.lons,
                       advection_scheme=self.advection_scheme,
                       advection_limiter=self.advection_limiter)
        print(f"[startup] numerical operators initialized: "
              f"{_time.perf_counter() - stage_started:.3f}s", flush=True)
        # Area weights for global budgets. The finite-volume operators use
        # the clamped cos(lat) metric near the poles, so this is the measure
        # in which the discrete mass and tracer integrals are conserved.
        cell_weight = 1.0 / self.ops.invcoslat
        self._area_weights = (
            cell_weight / (cell_weight.sum() * self.nlon)).astype(xp.float32)
        self.dt = float(cfg.time.dt)
        self.t = _dt.datetime.fromisoformat(str(cfg.time.start))
        self.step_count = 0
        dynamics_core = str(cfg.physics.dynamics.core).lower()
        if dynamics_core not in {"primitive_equations", "shallow_water"}:
            raise ValueError(
                "physics.dynamics.core must be primitive_equations or shallow_water")
        self._primitive_enabled = dynamics_core == "primitive_equations"
        _, d0, l0 = insolation(self.xp, self.ops.lat, self.ops.lon_rad, self.t,
                               cfg.physics.S0, cfg.physics.diurnal_cycle)
        self.subsolar = (float(d0), float(l0))
        stage_started = _time.perf_counter()
        self._init_vertical_grid()
        print(f"[startup] vertical grid initialized ({self.nz} levels): "
              f"{_time.perf_counter() - stage_started:.3f}s", flush=True)
        stage_started = _time.perf_counter()
        self._init_terrain_dynamics()
        self._init_land_hydrology()
        print(f"[startup] terrain dynamics initialized: "
              f"{_time.perf_counter() - stage_started:.3f}s", flush=True)
        stage_started = _time.perf_counter()
        self._init_state()
        print(f"[startup] model state initialized ({self.initialization_source}): "
              f"{_time.perf_counter() - stage_started:.3f}s", flush=True)
        print(f"[startup] EarthModel initialization complete: "
              f"{_time.perf_counter() - init_started:.3f}s", flush=True)

    # ------------------------------------------------------------
    def _init_vertical_grid(self):
        """Build layer centres, interfaces and the hybrid sigma-pressure grid."""
        vp = self.cfg.physics.vertical
        dyn = self.cfg.physics.dynamics
        xp = self.xp
        levels = [float(z) for z in vp.levels_m]
        if not bool(vp.enabled):
            levels = levels[:1]
        if not levels or any(z < 0 for z in levels):
            raise ValueError("physics.vertical.levels_m must contain non-negative heights")
        if any(b <= a for a, b in zip(levels, levels[1:])):
            raise ValueError("physics.vertical.levels_m must be strictly increasing")

        self.vertical_coordinate = "height"
        if self._primitive_enabled:
            coordinate = str(vp.get("coordinate", "sigma")).lower()
            if coordinate not in {"sigma", "hybrid"}:
                raise ValueError(
                    "physics.vertical.coordinate must be sigma or hybrid")
            reference_ps = float(dyn.reference_surface_pressure_pa)
            top_pressure = float(dyn.top_pressure_pa)
            if not bool(vp.enabled):
                coordinate = "sigma"
                hyai, hybi = sigma_to_hybrid([1.0, 0.0], top_pressure)
            elif coordinate == "sigma":
                sigma = validate_sigma_interfaces(
                    vp.sigma_interfaces, len(levels))
                hyai, hybi = sigma_to_hybrid(sigma, top_pressure)
            else:
                hy = vp.hybrid
                explicit_a = list(hy.get("hyai", None) or [])
                explicit_b = list(hy.get("hybi", None) or [])
                if explicit_a or explicit_b:
                    hyai = _np.asarray(explicit_a, dtype=_np.float64)
                    hybi = _np.asarray(explicit_b, dtype=_np.float64)
                    if hyai.size and abs(hyai[-1] - top_pressure) > 1.0:
                        raise ValueError(
                            "physics.dynamics.top_pressure_pa must equal the "
                            "last explicit hyai value")
                else:
                    hyai, hybi = generate_hybrid_coefficients(
                        int(hy.levels), top_pressure, reference_ps,
                        float(hy.transition_pressure_pa),
                        top_layers=int(hy.top_layers),
                        top_stretch=float(hy.top_stretch),
                        surface_layers=int(hy.surface_layers),
                        surface_ratio=float(hy.surface_ratio),
                        sigma_exponent=float(hy.sigma_exponent))
            hyai, hybi = validate_hybrid_coefficients(
                hyai, hybi, reference_ps, float(dyn.min_surface_pressure_pa))
            top_pressure = float(hyai[-1])
            self.top_pressure_pa = top_pressure
            self.vertical_coordinate = coordinate
            self.hyai, self.hybi = hyai, hybi
            nz = len(hyai) - 1
            p_ref_i = (hyai.astype(_np.float64)
                       + hybi.astype(_np.float64) * reference_ps)
            p_ref_full = 0.5 * (p_ref_i[:-1] + p_ref_i[1:])
            self.reference_pressure_interfaces_pa = p_ref_i.astype(_np.float32)
            self.reference_pressure_layers_pa = p_ref_full.astype(_np.float32)
            if coordinate == "hybrid":
                # Nominal heights (initialization, data interpolation and
                # display labels) follow the standard atmosphere.
                levels = [float(z) for z in standard_atmosphere_height(p_ref_full)]
            # Nominal sigma of the reference column; exact in sigma mode.
            nominal = (p_ref_i - top_pressure) / (reference_ps - top_pressure)
            nominal[0], nominal[-1] = 1.0, 0.0
            self.sigma_interfaces = nominal.astype(_np.float32)
            self.sigma_centres = 0.5 * (self.sigma_interfaces[:-1]
                                        + self.sigma_interfaces[1:])
            self.sigma_thickness = (self.sigma_interfaces[:-1]
                                    - self.sigma_interfaces[1:])
            self._hyai_xp = xp.asarray(hyai, dtype=xp.float32)
            self._hybi_xp = xp.asarray(hybi, dtype=xp.float32)
            self._d_hyai_xp = xp.asarray(
                hyai[:-1] - hyai[1:], dtype=xp.float32)[:, None, None]
            self._d_hybi_xp = xp.asarray(
                hybi[:-1] - hybi[1:], dtype=xp.float32)[:, None, None]
            self._init_upper_damping(p_ref_full, top_pressure, nz)

        self.levels_m = _np.asarray(levels, dtype=_np.float32)
        self.nz = len(levels)
        edges = _np.empty(self.nz + 1, dtype=_np.float32)
        edges[0] = 0.0
        if self.nz == 1:
            edges[1] = max(1000.0, 2.0 * levels[0])
        else:
            edges[1:-1] = 0.5 * (self.levels_m[:-1] + self.levels_m[1:])
            edges[-1] = self.levels_m[-1] + 0.5 * (
                self.levels_m[-1] - self.levels_m[-2])
        self.level_edges_m = edges
        self.layer_dz_m = _np.diff(edges).astype(_np.float32)

        scale_h = max(float(vp.scale_height), float(vp.scale_height_min_m))
        mass = _np.exp(-edges[:-1] / scale_h) - _np.exp(-edges[1:] / scale_h)
        mass /= mass.sum()
        self.layer_mass_fractions = mass.astype(_np.float32)
        self._z3 = xp.asarray(self.levels_m, dtype=xp.float32)[:, None, None]
        self._mass3 = xp.asarray(mass, dtype=xp.float32)[:, None, None]
        if self._primitive_enabled:
            # Fraction of the reference column mass held by each layer.
            self.layer_mass_fractions = self.sigma_thickness.copy()
            self._mass3 = xp.asarray(
                self.sigma_thickness, dtype=xp.float32)[:, None, None]

    def _init_upper_damping(self, reference_pressure_layers, top_pressure, nz):
        """Sponge weights, Rayleigh rates and divergence-damping metrics."""
        xp, o, dyn = self.xp, self.ops, self.cfg.physics.dynamics
        ud = dyn.upper_damping
        weights = _np.zeros(nz, dtype=_np.float32)
        if bool(ud.enabled):
            weights = sponge_profile(
                reference_pressure_layers,
                float(ud.sponge_base_pressure_pa), top_pressure)
        self.sponge_weights = weights
        active = _np.nonzero(weights > 0.0)[0]
        self.sponge_active = bool(ud.enabled) and active.size > 0
        self._sponge_first_level = int(active[0]) if active.size else nz
        tau = max(float(ud.rayleigh_tau_s), self.dt)
        self._sponge_rate_xp = xp.asarray(
            weights / tau, dtype=xp.float32)[:, None, None]
        self._sponge_weights_xp = xp.asarray(
            weights, dtype=xp.float32)[:, None, None]
        self._sponge_diffusion_factor = max(float(ud.diffusion_multiplier) - 1.0, 0.0)
        # Second-order divergence damping scaled by the explicit stability
        # limit of the local grid cell, so one dimensionless coefficient is
        # safe at every latitude.
        coefficient = float(dyn.divergence_damping_coeff)
        self.divergence_damping_active = coefficient > 0.0
        cell = xp.minimum(o.dx, xp.float32(o.dy))
        self._divergence_damping_nu = (
            coefficient * cell * cell / self.dt).astype(xp.float32)
        profile = 1.0 + max(float(ud.divergence_damping_multiplier) - 1.0,
                            0.0) * weights
        self._divergence_damping_profile_xp = xp.asarray(
            profile, dtype=xp.float32)[:, None, None]
    def _init_terrain_dynamics(self):
        """Pre-compute a smoothed terrain gradient used by the wind scheme."""
        xp, o, tp = self.xp, self.ops, self.cfg.physics.topography
        terrain = xp.maximum(self.elev, 0.0) * self.land
        center = float(tp.smooth_center_weight)
        neighbor = float(tp.smooth_neighbor_weight)
        for _ in range(max(0, int(tp.smooth_passes))):
            terrain = (neighbor * o.rollx(terrain, 1) + center * terrain
                       + neighbor * o.rollx(terrain, -1))
            terrain = (neighbor * o.shifty(terrain, 1) + center * terrain
                       + neighbor * o.shifty(terrain, -1))
        if not bool(tp.enabled):
            terrain = xp.zeros_like(terrain)
        self.surface_height = terrain.astype(xp.float32)
        self.surface_geopotential = (
            float(self.cfg.physics.g_eff) * self.surface_height).astype(xp.float32)
        slope_x, slope_y = o.gradient(terrain)
        self.terrain_slope_x = slope_x.astype(xp.float32)
        self.terrain_slope_y = slope_y.astype(xp.float32)
        self.terrain_slope = xp.sqrt(self.terrain_slope_x ** 2
                                     + self.terrain_slope_y ** 2).astype(xp.float32)

    def _init_land_hydrology(self):
        """Pre-compute runoff routing targets and ground-water diffusion.

        Routing: steepest descent over the four face neighbours of the
        smoothed model terrain, the same surface that defines
        ``terrain_slope_x/y``. The sea is treated as lying below every land
        cell so coastal cells drain into the ocean. Land cells without a
        lower neighbour are depressions; their runoff keeps the local
        drainage rule and simply leaves the system.
        """
        xp, o, p = self.xp, self.ops, self.cfg.physics
        nlat, nlon = self.nlat, self.nlon
        n = nlat * nlon
        self.river_routing_enabled = bool(p.river_routing)
        self.river_flow = xp.zeros((nlat, nlon), xp.float32)
        zero = xp.zeros((), xp.float64)
        self._river_budget = {"river_discharge_to_ocean_m3_s": zero,
                              "endorheic_drainage_m3_s": zero.copy()}
        # True cell area in m^2. Routing moves water between whole cells, so
        # conservation only needs one consistent area per cell.
        coslat = to_cpu(o.coslat)[:, 0].astype(_np.float64)
        area = coslat * A_EARTH * (2.0 * _np.pi / nlon) * float(o.dy)
        self._cell_area_m2 = xp.asarray(area, dtype=xp.float64)[:, None]

        land = to_cpu(self.land) > 0.5
        terrain = (self.surface_height if bool(p.topography.enabled)
                   else xp.maximum(self.elev, 0.0) * self.land)
        height = _np.where(land, to_cpu(terrain), -1.0).astype(_np.float64)
        dx = to_cpu(o.dx)[:, 0].astype(_np.float64)[:, None]
        dy = float(o.dy)
        index = _np.arange(n).reshape(nlat, nlon)
        target = _np.full((nlat, nlon), n, dtype=_np.int64)   # n marks a sink
        best = _np.zeros((nlat, nlon))

        def from_south(a):   # same convention as Ops.shifty(a, 1)
            return _np.concatenate([a[:1], a[:-1]])

        def from_north(a):
            return _np.concatenate([a[1:], a[-1:]])

        neighbours = (
            (_np.roll(height, -1, axis=1), _np.roll(index, -1, axis=1), dx),
            (_np.roll(height, 1, axis=1), _np.roll(index, 1, axis=1), dx),
            (from_north(height), from_north(index), dy),
            (from_south(height), from_south(index), dy),
        )
        for nb_height, nb_index, dist in neighbours:
            drop = (height - nb_height) / dist
            better = land & (drop > best)
            best = _np.where(better, drop, best)
            target = _np.where(better, nb_index, target)
        drains = land & (target < n)
        target_is_land = land.ravel()[_np.minimum(target, n - 1)]
        self.river_target = xp.asarray(target.ravel())
        self.river_sink = xp.asarray((land & ~drains).astype(_np.float32))
        self.river_to_ocean = xp.asarray(
            (drains & ~target_is_land).astype(_np.float32))

        # Explicit flux-form diffusion coefficients D*dt/dx^2, capped at half
        # the stability limit so one diffusivity is safe at every latitude.
        diffusivity = float(p.ground_water_diffusivity)
        if diffusivity < 0:
            raise ValueError(
                "physics.ground_water_diffusivity must be non-negative")
        self.ground_water_diffusion_enabled = diffusivity > 0.0
        kx = diffusivity * self.dt * o.invdx ** 2
        ky = diffusivity * self.dt * float(o.invdy) ** 2 * xp.ones_like(kx)
        scale = xp.minimum(1.0, 0.25 / xp.maximum(kx + ky, 1.0e-30))
        self._gw_kx = (kx * scale).astype(xp.float32)
        self._gw_ky = (ky * scale).astype(xp.float32)
        # Relative area of the face shared with the northern neighbour, and
        # a mask removing the non-existent face south of the first row.
        self._gw_face_north = (
            0.5 * (o.coslat + o.shifty(o.coslat, -1))).astype(xp.float32)
        south_face = _np.ones((nlat, 1), dtype=_np.float32)
        south_face[0] = 0.0
        self._gw_south_face_mask = xp.asarray(south_face)

    # ------------------------------------------------------------
    def _init_state(self):
        xp, p = self.xp, self.cfg.physics
        ic, bounds = p.initial_conditions, p.bounds
        lat = self.ops.lat
        f32 = xp.float32
        Teq = (float(ic.equilibrium_temp_base_k)
               - float(ic.equilibrium_temp_pole_delta_k)
               * xp.sin(lat) ** 2) * xp.ones((1, self.nlon))
        lapse = float(p.vertical.lapse_rate)
        self.T_layers = xp.stack(
            [Teq - lapse * z for z in self.levels_m], axis=0).astype(f32)
        # 海表: 不低于 -2C; 陆面随高度递减
        self.Ts = (Teq - float(ic.surface_lapse_rate)
                   * xp.maximum(self.elev, 0) * self.land)
        self.Ts = xp.where(
            (self.ocean > 0.5)
            & (self.Ts < float(bounds.ocean_initial_min_temp_k)),
            float(bounds.ocean_initial_min_temp_k), self.Ts).astype(f32)
        init_surface_rh = float(p.init_surface_rh)
        init_upper_rh = float(p.init_upper_rh)
        rh_profile = (init_upper_rh + (init_surface_rh - init_upper_rh)
                      * _np.exp(-self.levels_m
                                / float(ic.humidity_decay_height_m)))
        self.q_layers = (xp.asarray(rh_profile, dtype=f32)[:, None, None]
                         * qsat(xp, self.T_layers)).astype(f32)
        wave_amp = float(p.ideal_wave_amp_K)
        if wave_amp:
            lon = self.ops.lon_rad
            wave = ((xp.sin(float(ic.temp_wave_lon1) * lon
                             + float(ic.temp_wave_phase1) * xp.sin(lat))
                     + float(ic.temp_wave_weight2)
                     * xp.sin(float(ic.temp_wave_lon2) * lon
                              + float(ic.temp_wave_phase2) * xp.sin(lat)))
                    * xp.cos(lat) ** 2)
            decay = xp.exp(-self._z3 / float(ic.temp_wave_decay_height_m))
            self.T_layers = (self.T_layers + wave_amp * decay * wave).astype(f32)
            q_wave = 1.0 + float(p.ideal_humidity_wave) * wave
            self.q_layers = xp.clip(
                self.q_layers * q_wave, float(bounds.humidity_min),
                float(bounds.humidity_max)).astype(f32)
        layer_mean = self.T_layers.mean(axis=(1, 2), keepdims=True)
        self.h_layers = (p.H0 - p.beta_T *
                         (self.T_layers - layer_mean)).astype(f32)
        shape3 = (self.nz, self.nlat, self.nlon)
        self.u_layers, self.v_layers = self._initial_wind_layers(shape3, f32)
        self.w_layers = xp.zeros(shape3, f32)
        z = xp.zeros((self.nlat, self.nlon), f32)
        self.uo, self.vo = z.copy(), z.copy()
        self.cloud, self.precip = z.copy(), z.copy()
        capacity = float(p.ground_water_capacity_mm)
        initial_water = float(p.initial_ground_water_mm)
        if capacity <= 0:
            raise ValueError("physics.ground_water_capacity_mm must be positive")
        if initial_water < 0:
            raise ValueError("physics.initial_ground_water_mm must be non-negative")
        if float(p.ground_evap_exponent) < 0:
            raise ValueError("physics.ground_evap_exponent must be non-negative")
        if float(p.ground_runoff_tau) <= 0:
            raise ValueError("physics.ground_runoff_tau must be positive")
        if float(p.ground_runoff_exponent) <= 1:
            raise ValueError("physics.ground_runoff_exponent must be greater than 1")
        self.ground_water = (self.land * min(initial_water, capacity)).astype(f32)
        self.runoff = z.copy()
        self.ice, self.snow = z.copy(), z.copy()
        self.initialization_source = "idealized"
        self._initial_mslp_hpa = xp.full_like(
            z, float(ic.real_default_mslp_hpa))
        self._sync_surface_views()
        self._diag_surface()
        self._maybe_apply_real_initialization()
        if self._primitive_enabled:
            self._init_primitive_state()
        self._init_ocean_layers()

    def _maybe_apply_real_initialization(self):
        """Replace the ideal state with one real analysis snapshot if enabled."""
        data = getattr(self.cfg, "data", None)
        mode = str(getattr(data, "init_mode", "ideal")).lower()
        if mode not in {"ideal", "auto", "real"}:
            raise ValueError("data.init_mode must be ideal, auto, or real")
        if mode == "ideal":
            return
        try:
            real = load_real_initialization(
                self.cfg, self.lats, self.lons, self.levels_m,
                target_pressure_pa=(
                    self.reference_pressure_layers_pa
                    if self._primitive_enabled else None))
        except (FileNotFoundError, RealDataError, ImportError) as exc:
            if mode == "auto":
                print(f"[model] real initialization unavailable; using idealized state: {exc}")
                return
            raise

        xp, p = self.xp, self.cfg.physics
        ic, bounds, pressure = p.initial_conditions, p.bounds, p.pressure

        def finite(value, fallback):
            if value is None:
                return fallback
            a = _np.asarray(value, dtype=_np.float32)
            fallback_a = _np.asarray(fallback, dtype=_np.float32)
            fallback_value = (float(_np.nanmean(fallback_a))
                              if fallback_a.size else float(fallback_a))
            return _np.nan_to_num(a, nan=_np.nanmean(a) if _np.isfinite(a).any()
                                  else fallback_value, posinf=fallback_value,
                                  neginf=fallback_value)

        temp = finite(real["temp"], float(ic.real_default_temp_k))
        q = finite(real["q"], float(ic.real_default_humidity))
        u = finite(real["u"], 0.0)
        v = finite(real["v"], 0.0)
        if temp.shape != (self.nz, self.nlat, self.nlon):
            raise RealDataError(f"real temperature shape {temp.shape} does not match model")
        self.T_layers = xp.asarray(
            _np.clip(temp, float(bounds.air_temp_min_k),
                     float(bounds.air_temp_max_k)), dtype=xp.float32)
        self.q_layers = xp.asarray(
            _np.clip(q, float(bounds.humidity_min),
                     float(bounds.humidity_max)), dtype=xp.float32)
        self.u_layers = xp.asarray(_np.clip(u, -p.umax, p.umax), dtype=xp.float32)
        self.v_layers = xp.asarray(_np.clip(v, -p.umax, p.umax), dtype=xp.float32)
        self.w_layers = xp.zeros_like(self.T_layers)

        mslp = finite(real.get("mslp_hpa"), float(ic.real_default_mslp_hpa))
        self._initial_mslp_hpa = xp.asarray(mslp, dtype=xp.float32)
        h_base = (float(p.H0)
                  + (mslp - float(pressure.mslp_reference_hpa))
                  * float(pressure.thickness_per_hpa))
        tmean = self.T_layers.mean(axis=(1, 2), keepdims=True)
        h = h_base[None, :, :] - float(p.beta_T) * (
            temp - _np.asarray(to_cpu(tmean)))
        self.h_layers = xp.asarray(_np.clip(
            h, float(pressure.thickness_min_factor) * float(p.H0),
            float(pressure.thickness_max_factor) * float(p.H0)),
            dtype=xp.float32)

        surface = finite(real.get("surface"), temp[0])
        ocean_surface = real.get("ocean_surface")
        if ocean_surface is not None:
            ocean_surface = finite(ocean_surface, surface)
            surface = _np.where(_np.asarray(to_cpu(self.ocean)) > 0.5,
                                ocean_surface, surface)
        self.Ts = xp.asarray(
            _np.clip(surface, float(bounds.surface_temp_min_k),
                     float(bounds.surface_temp_max_k)), dtype=xp.float32)

        self.uo = xp.asarray(_np.clip(
            finite(real.get("ou"), 0.0), -float(bounds.ocean_current_max_ms),
            float(bounds.ocean_current_max_ms)), dtype=xp.float32) * self.ocean
        self.vo = xp.asarray(_np.clip(
            finite(real.get("ov"), 0.0), -float(bounds.ocean_current_max_ms),
            float(bounds.ocean_current_max_ms)), dtype=xp.float32) * self.ocean
        self.cloud = xp.asarray(_np.clip(
            finite(real.get("cloud"), 0.0), float(bounds.cloud_min),
            float(bounds.cloud_max)), dtype=xp.float32)
        self.precip = xp.asarray(_np.clip(
            finite(real.get("precip"), 0.0), 0,
            float(bounds.precip_max_mmh)), dtype=xp.float32)
        sea_ice = real.get("sea_ice")
        if sea_ice is not None:
            self.ice = xp.asarray(_np.clip(
                finite(sea_ice, 0.0), float(bounds.cloud_min),
                float(bounds.cloud_max)), dtype=xp.float32) * self.ocean
        self.initialization_source = "real"
        self._sync_surface_views()
        self._diag_surface()

    def _init_primitive_state(self):
        """Initialize surface pressure and hydrostatic hybrid-level geometry."""
        xp, p = self.xp, self.cfg.physics
        dyn, bounds = p.dynamics, p.bounds
        psl = self._initial_mslp_hpa * 100.0
        tv0 = self.T_layers[0] * (1.0 + 0.608 * self.q_layers[0])
        self.surface_pressure = (
            psl * xp.exp(-self.surface_geopotential
                         / xp.maximum(RD * tv0, 1.0)))
        self.surface_pressure = xp.clip(
            self.surface_pressure,
            float(dyn.min_surface_pressure_pa),
            float(dyn.max_surface_pressure_pa)).astype(xp.float32)
        _, pressure_layers = hybrid_pressures(
            xp, self._hyai_xp, self._hybi_xp, self.surface_pressure)

        if self.initialization_source == "idealized":
            ic = p.initial_conditions
            # Build the ideal atmosphere on pressure surfaces. Using the same
            # T at the same sigma over both mountains and ocean creates a large
            # artificial PGF because those sigma points have different p/z.
            reference_height = -float(p.vertical.scale_height) * xp.log(
                pressure_layers / float(dyn.reference_surface_pressure_pa))
            reference_height = xp.maximum(reference_height, 0.0)
            base_temperature = (
                float(ic.equilibrium_temp_base_k)
                - float(ic.equilibrium_temp_pole_delta_k)
                * xp.sin(self.ops.lat) ** 2)
            self.T_layers = self._equilibrium_temperature_profile(
                base_temperature, reference_height)
            wave_amp = float(p.ideal_wave_amp_K)
            if wave_amp:
                lon, lat = self.ops.lon_rad, self.ops.lat
                wave = ((xp.sin(float(ic.temp_wave_lon1) * lon
                                + float(ic.temp_wave_phase1) * xp.sin(lat))
                         + float(ic.temp_wave_weight2)
                         * xp.sin(float(ic.temp_wave_lon2) * lon
                                  + float(ic.temp_wave_phase2) * xp.sin(lat)))
                        * xp.cos(lat) ** 2)
                self.T_layers += (
                    wave_amp * xp.exp(
                        -reference_height
                        / float(ic.temp_wave_decay_height_m)) * wave)
            self.T_layers = xp.clip(
                self.T_layers, float(bounds.air_temp_min_k),
                float(bounds.air_temp_max_k)).astype(xp.float32)

            # Recompute terrain surface pressure with the thermodynamically
            # consistent lowest-level temperature.
            tv0 = self.T_layers[0] * (1.0 + 0.608 * self.q_layers[0])
            self.surface_pressure = xp.clip(
                psl * xp.exp(-self.surface_geopotential
                             / xp.maximum(RD * tv0, 1.0)),
                float(dyn.min_surface_pressure_pa),
                float(dyn.max_surface_pressure_pa)).astype(xp.float32)
            _, pressure_layers = hybrid_pressures(
                xp, self._hyai_xp, self._hybi_xp, self.surface_pressure)

        # Ideal humidity was originally built with a fixed 1000-hPa qsat.
        # Reconstruct it at the actual layer pressure while preserving the
        # intended relative-humidity profile and wave perturbation.
        if self.initialization_source == "idealized" and p.moisture:
            ic = p.initial_conditions
            rh_profile = (float(p.init_upper_rh)
                          + (float(p.init_surface_rh)
                             - float(p.init_upper_rh))
                          * _np.exp(-self.levels_m
                                    / float(ic.humidity_decay_height_m)))
            rh = xp.asarray(rh_profile, dtype=xp.float32)[:, None, None]
            if float(p.ideal_humidity_wave):
                lon, lat = self.ops.lon_rad, self.ops.lat
                wave = ((xp.sin(float(ic.temp_wave_lon1) * lon
                                + float(ic.temp_wave_phase1) * xp.sin(lat))
                         + float(ic.temp_wave_weight2)
                         * xp.sin(float(ic.temp_wave_lon2) * lon
                                  + float(ic.temp_wave_phase2) * xp.sin(lat)))
                        * xp.cos(lat) ** 2)
                rh = rh * (1.0 + float(p.ideal_humidity_wave) * wave)
            self.q_layers = xp.clip(
                rh * qsat(xp, self.T_layers, pressure_layers),
                float(bounds.humidity_min),
                float(bounds.humidity_max)).astype(xp.float32)

        self._refresh_primitive_diagnostics()
        self.omega_layers = xp.zeros_like(self.T_layers)
        self.sigma_dot_interfaces = xp.zeros(
            (self.nz + 1, self.nlat, self.nlon), dtype=xp.float32)
        zero = xp.zeros((), dtype=xp.float32)
        self._energy_fixer_debt_k = xp.zeros((), dtype=xp.float64)
        self._limiter_counts = {
            name: zero for name in ("wind", "temperature", "humidity",
                                    "surface_pressure")}
        g = float(p.g_eff)
        self._budget = {
            "total_energy_j_m2": self._global_mean(column_total_energy(
                xp, self.u_layers, self.v_layers, self.T_layers,
                self.q_layers, self.layer_pressure_thickness_pa / g,
                self.surface_geopotential,
                (self.surface_pressure - self.top_pressure_pa) / g,
                LV if p.moisture else 0.0)),
            "dynamics_energy_error_w_m2": zero,
            "energy_fixer_dT_k": zero,
            "energy_fixer_pending_dT_k": zero,
            "physics_energy_change_w_m2": zero,
            "physics_forcing_w_m2": zero,
        }
        self._sync_surface_views()

    def _equilibrium_temperature_profile(self, base_temperature, height_agl):
        """Tropospheric lapse rate under a latitude-independent stratosphere.

        The troposphere cools at ``lapse_rate`` until it meets the
        stratospheric profile ``minimum_equilibrium_temp_k +
        stratosphere_lapse_rate * (z - tropopause_height_m)``, so the
        tropopause is higher and colder in the tropics than at the poles and
        the stratosphere carries no meridional temperature gradient.  The
        former profile kept the surface gradient all the way to the model
        top, which forced an unphysical thermal-wind jet there once the top
        was raised into the stratosphere.
        """
        xp, vp = self.xp, self.cfg.physics.vertical
        tropopause = float(vp.tropopause_height_m)
        tropospheric = (base_temperature[None, :, :]
                        - float(vp.lapse_rate) * height_agl)
        stratospheric = (float(vp.minimum_equilibrium_temp_k)
                         + float(vp.stratosphere_lapse_rate)
                         * xp.maximum(height_agl - tropopause, 0.0))
        return xp.maximum(tropospheric, stratospheric)

    def _sync_surface_views(self):
        """Keep the original two-dimensional API mapped to the lowest layer."""
        self.Ta = self.T_layers[0]
        self.q = self.q_layers[0]
        self.h = self.h_layers[0]
        self.u = self.u_layers[0]
        self.v = self.v_layers[0]
        self.w = self.w_layers[0]

    def _init_ocean_layers(self):
        """Initialize the optional active deep-ocean layer from the surface state."""
        xp, p = self.xp, self.cfg.physics
        layers = p.ocean_layers
        self.ocean_layers_enabled = bool(layers.enabled)
        self.ocean_upper_depth_m = float(p.mld)
        self.ocean_lower_depth_m = float(layers.lower_depth_m)
        if self.ocean_layers_enabled:
            if self.ocean_upper_depth_m <= 0:
                raise ValueError("physics.mld must be positive when ocean_layers is enabled")
            if self.ocean_lower_depth_m <= 0:
                raise ValueError("physics.ocean_layers.lower_depth_m must be positive")
            for name in ("interlayer_drag", "interlayer_heat_exchange",
                         "deep_drag", "deep_visc"):
                if float(getattr(layers, name)) < 0:
                    raise ValueError(f"physics.ocean_layers.{name} must be non-negative")

        zero = xp.zeros_like(self.Ts)
        initial_offset = float(layers.deep_initial_offset_k)
        self.To_deep = xp.where(
            self.ocean > 0.5,
            xp.maximum(self.Ts - initial_offset, float(layers.deep_temp_min_k)),
            self.Ts,
        ).astype(xp.float32)
        self.uo_deep = zero.copy()
        self.vo_deep = zero.copy()

    def _advance_ocean(self, Ts, air_u, air_v):
        """Advance wind-driven surface flow and the optional coupled deep layer."""
        xp, p, o, dt = self.xp, self.cfg.physics, self.ops, self.dt
        bounds, ot = p.bounds, p.ocean_transport
        if not p.ocean:
            return Ts

        uo, vo = self.uo, self.vo
        spd = (xp.sqrt(air_u * air_u + air_v * air_v)
               + float(ot.wind_speed_floor_ms))
        uo = (uo + dt * (p.tau_ocean * spd * air_u - p.drag_ocean * uo)
              + dt * p.visc_ocean * o.lap(uo))
        vo = (vo + dt * (p.tau_ocean * spd * air_v - p.drag_ocean * vo)
              + dt * p.visc_ocean * o.lap(vo))
        uo, vo = o.coriolis_rotate(
            uo, vo, dt * float(ot.ekman_coriolis_factor))

        open_ocean = self.ocean * (1 - self.ice)
        if not self.ocean_layers_enabled:
            max_current = float(bounds.ocean_current_max_ms)
            self.uo = (xp.clip(uo, -max_current, max_current)
                       * open_ocean).astype(xp.float32)
            self.vo = (xp.clip(vo, -max_current, max_current)
                       * open_ocean).astype(xp.float32)
            return xp.where(
                self.ocean > 0.5,
                o.adv_diff_step(Ts, self.uo, self.vo,
                                float(ot.surface_heat_diffusivity), dt),
                Ts,
            )

        layers = p.ocean_layers
        uod, vod = self.uo_deep, self.vo_deep
        uod = (uod - dt * float(layers.deep_drag) * uod
               + dt * float(layers.deep_visc) * o.lap(uod))
        vod = (vod - dt * float(layers.deep_drag) * vod
               + dt * float(layers.deep_visc) * o.lap(vod))
        uod, vod = o.coriolis_rotate(
            uod, vod, dt * float(ot.ekman_coriolis_factor))

        depth_ratio = self.ocean_upper_depth_m / self.ocean_lower_depth_m
        momentum_exchange = dt * float(layers.interlayer_drag)
        du, dv = momentum_exchange * (uod - uo), momentum_exchange * (vod - vo)
        uo += du
        vo += dv
        uod -= depth_ratio * du
        vod -= depth_ratio * dv
        max_current = float(bounds.ocean_current_max_ms)
        self.uo = (xp.clip(uo, -max_current, max_current)
                   * open_ocean).astype(xp.float32)
        self.vo = (xp.clip(vo, -max_current, max_current)
                   * open_ocean).astype(xp.float32)
        self.uo_deep = (xp.clip(uod, -max_current, max_current)
                        * open_ocean).astype(xp.float32)
        self.vo_deep = (xp.clip(vod, -max_current, max_current)
                        * open_ocean).astype(xp.float32)

        surface = o.adv_diff_step(Ts, self.uo, self.vo,
                                  float(ot.surface_heat_diffusivity), dt)
        deep = o.adv_diff_step(
            self.To_deep, self.uo_deep, self.vo_deep,
            float(layers.deep_visc), dt,
        )
        heat_exchange = dt * float(layers.interlayer_heat_exchange)
        dtemp = heat_exchange * (deep - surface)
        surface += dtemp
        deep -= depth_ratio * dtemp
        self.To_deep = xp.where(
            self.ocean > 0.5,
            xp.maximum(deep, float(layers.deep_temp_min_k)),
            self.To_deep,
        ).astype(xp.float32)
        return xp.where(self.ocean > 0.5, surface, Ts)

    def _initial_wind_layers(self, shape3, dtype):
        """Return an idealized balanced-ish zonal wind with wave seeds."""
        xp, p = self.xp, self.cfg.physics
        ws = p.ideal_wind_shape
        if not bool(p.ideal_wind_enabled):
            return xp.zeros(shape3, dtype), xp.zeros(shape3, dtype)

        lat = self.ops.lat
        lat_deg = xp.abs(lat * (180.0 / _np.pi))
        z = self._z3
        trade = float(p.ideal_trade_wind_ms)
        surface_westerly = float(p.ideal_surface_westerly_ms)
        jet = float(p.ideal_jet_ms)
        polar = float(p.ideal_polar_easterly_ms)
        jet_lat = float(p.ideal_jet_lat_deg)
        jet_width = max(float(p.ideal_jet_width_deg), 1.0)
        jet_height = float(p.ideal_jet_height_m)
        jet_depth = max(float(p.ideal_jet_depth_m), 1.0)

        trades = (-trade
                  * xp.exp(-(lat_deg / float(ws.trade_lat_width_deg)) ** 2)
                  * xp.exp(-z / float(ws.trade_decay_height_m)))
        low_westerlies = (surface_westerly
                          * xp.exp(-((lat_deg
                                       - float(ws.surface_westerly_lat_deg))
                                      / float(ws.surface_westerly_width_deg))
                                     ** 2)
                          * xp.exp(-z
                                   / float(ws.surface_westerly_decay_height_m)))
        upper_jets = (jet * xp.exp(-((lat_deg - jet_lat) / jet_width) ** 2)
                      * xp.exp(-((z - jet_height) / jet_depth) ** 2))
        polar_easterlies = (-polar
                            * xp.exp(-((lat_deg
                                         - float(ws.polar_easterly_lat_deg))
                                        / float(ws.polar_easterly_width_deg))
                                       ** 2)
                            * xp.exp(-z
                                     / float(ws.polar_easterly_decay_height_m)))
        u = (trades + low_westerlies + upper_jets + polar_easterlies
             + xp.zeros(shape3, dtype))

        wave_amp = float(p.ideal_wind_wave_ms)
        if wave_amp:
            lon = xp.asarray(_np.radians(self.lons), dtype=dtype)[None, :]
            wave_z = xp.exp(-z / float(ws.wave_decay_height_m))
            phase = (float(ws.wave_wavenumber) * lon
                     + float(ws.wave_phase) * xp.sin(lat))
            u = (u + float(ws.wave_u_fraction) * wave_amp * wave_z
                 * xp.sin(phase) * xp.cos(lat) ** 2)
            v = (wave_amp * wave_z
                 * xp.cos(phase) * xp.cos(lat) ** 2)
        else:
            v = xp.zeros_like(u)
        return u.astype(dtype), v.astype(dtype)

    def _vertical_transport(self, field, w, diffusivity, dt):
        """Flux-form vertical exchange without spurious constant-field sources.

        Orographic lift creates substantial vertical velocity convergence over
        land.  Advecting the full state as ``w * field`` turns that convergence
        into an artificial heat/cold source.  Transport only the column anomaly
        so a vertically constant field remains constant while the column
        integral is still conserved.
        """
        if self.nz == 1:
            return field
        xp = self.xp
        dz3 = xp.asarray(self.layer_dz_m, dtype=xp.float32)[:, None, None]
        column_mean = (field * dz3).sum(axis=0, keepdims=True) / dz3.sum()
        anomaly = field - column_mean
        interface_flux = [xp.zeros_like(field[0])]
        diff_flux = [xp.zeros_like(field[0])]
        for k in range(1, self.nz):
            wi = 0.5 * (w[k - 1] + w[k])
            upwind = xp.where(wi >= 0, anomaly[k - 1], anomaly[k])
            interface_flux.append(wi * upwind)
            dz = max(float(self.levels_m[k] - self.levels_m[k - 1]), 1.0)
            diff_flux.append(-float(diffusivity) * (field[k] - field[k - 1]) / dz)
        interface_flux.append(xp.zeros_like(field[0]))
        diff_flux.append(xp.zeros_like(field[0]))

        out = []
        for k in range(self.nz):
            adv = -(interface_flux[k + 1] - interface_flux[k]) / max(float(self.layer_dz_m[k]), 1.0)
            mix = -(diff_flux[k + 1] - diff_flux[k]) / max(float(self.layer_dz_m[k]), 1.0)
            out.append(field[k] + dt * (adv + mix))
        return xp.stack(out, axis=0)

    def _update_vertical_velocity(self, div, u, v, T, dt):
        """Diagnose mass-continuous motion, then add damped thermal buoyancy."""
        xp, p, vp, tp = (self.xp, self.cfg.physics,
                         self.cfg.physics.vertical, self.cfg.physics.topography)
        terrain_w = xp.zeros_like(div[0])
        if tp.enabled:
            terrain_w = (float(tp.lift_efficiency)
                         * (u[0] * self.terrain_slope_x
                            + v[0] * self.terrain_slope_y) * self.land)
            terrain_w = xp.clip(terrain_w, -float(tp.lift_max), float(tp.lift_max))

        interfaces = [terrain_w]
        for k in range(self.nz):
            interfaces.append(interfaces[-1] - div[k] * float(self.layer_dz_m[k]))
        # Enforce a rigid model top without changing the terrain lower boundary.
        top_error = interfaces[-1]
        model_top = max(float(self.level_edges_m[-1]), 1.0)
        interfaces = [wi - top_error * (float(z) / model_top)
                      for wi, z in zip(interfaces, self.level_edges_m)]
        target = xp.stack([0.5 * (interfaces[k] + interfaces[k + 1])
                           for k in range(self.nz)], axis=0)

        # Compare each layer with the local configured environmental profile.
        # This avoids mistaking the equator-to-pole temperature contrast for
        # convective buoyancy while still reacting to column instability.
        reference = (T[0:1] - float(vp.lapse_rate)
                     * (self._z3 - float(self.levels_m[0])))
        anomaly = T - reference
        buoyancy = (p.g_eff * float(vp.buoyancy_factor) * anomaly
                    / xp.maximum(T, float(vp.buoyancy_temp_min_k)))
        # Heating should redistribute air within a column, not accelerate its centre of mass.
        buoyancy -= (buoyancy * self._mass3).sum(axis=0, keepdims=True)
        relax = max(float(vp.continuity_relax), dt)
        w = self.w_layers + dt * ((target - self.w_layers) / relax
                                  + buoyancy - float(vp.w_damping) * self.w_layers)
        return xp.clip(w, -float(vp.w_max), float(vp.w_max)).astype(xp.float32)

    def _surface_atmospheric_drag(self):
        """Return the configured atmospheric drag for each surface type."""
        p = self.cfg.physics
        ocean_drag = float(p.drag_ocean_atmosphere)
        land_drag = float(p.drag_land_atmosphere)
        return self.ocean * ocean_drag + self.land * land_drag

    def _surface_evaporation(self, surface_temp, air_humidity, wind_speed,
                             air_pressure=None, air_temperature=None):
        """Return total and land evaporation in kg m-2 s-1.

        One millimetre of ground water is one kg m-2, so the land flux can be
        capped directly by the water available during this time step.
        """
        xp, p = self.xp, self.cfg.physics
        capacity = float(p.ground_water_capacity_mm)
        wetness = xp.clip(self.ground_water / capacity, 0, 1)
        wetness **= float(p.ground_evap_exponent)
        if air_pressure is None:
            air_pressure = float(
                p.dynamics.reference_surface_pressure_pa)
        if air_temperature is None:
            density = RHO_A
        else:
            density = (air_pressure / xp.maximum(
                RD * air_temperature * (1.0 + 0.608 * air_humidity), 1.0))
        potential = (float(p.c_evap) * density * wind_speed
                     * xp.maximum(qsat(xp, surface_temp, air_pressure)
                                  - air_humidity, 0))
        ocean_evap = potential * self.ocean
        land_evap = (potential * float(p.land_evap) * wetness * self.land)
        land_evap = xp.minimum(land_evap, self.ground_water / self.dt)
        return ocean_evap + land_evap, land_evap

    def _update_ground_water(self, land_evap, precipitation_mm):
        """Apply rain, evaporation, lateral exchange and drainage on land.

        Runoff (overflow plus river loss) is routed downslope: it recharges
        the ground water of the next cell down the terrain, or enters the
        sea at the coast. ``river_flow`` holds the resulting discharge in
        m^3/s, on land the flow leaving the cell and on ocean cells the
        river inflow at the mouth.
        """
        xp, p, dt = self.xp, self.cfg.physics, self.dt
        capacity = float(p.ground_water_capacity_mm)
        water = xp.maximum(
            self.ground_water
            + self.land * precipitation_mm
            - dt * land_evap,
            0,
        )
        if self.ground_water_diffusion_enabled:
            water = self._diffuse_ground_water(water)

        # Water above the reservoir capacity drains immediately. Below that
        # limit, exponent > 1 makes the relative loss rate grow with storage.
        overflow = xp.maximum(water - capacity, 0) * self.land
        water = xp.minimum(water, capacity) * self.land
        tau = float(p.ground_runoff_tau)
        exponent = float(p.ground_runoff_exponent)
        river_loss = (dt * capacity / tau
                      * xp.clip(water / capacity, 0, 1) ** exponent)
        river_loss = xp.minimum(river_loss, water) * self.land
        water = water - river_loss
        runoff_mm = overflow + river_loss
        if self.river_routing_enabled:
            inflow_mm = self._route_runoff(runoff_mm)
            water = water + inflow_mm * self.land
            flow_mm = xp.where(self.land > 0.5, runoff_mm, inflow_mm)
        else:
            flow_mm = runoff_mm
        self.ground_water = water.astype(xp.float32)
        self.runoff = (runoff_mm / dt * 3600.0).astype(xp.float32)
        self.river_flow = (flow_mm * self._cell_area_m2
                           / (1000.0 * dt)).astype(xp.float32)

    def _diffuse_ground_water(self, water):
        """Conservative lateral exchange ``D lap(water)`` between land cells.

        Flux form: water crossing a land-land face leaves one cell and
        enters the other, so nothing crosses the coastline or is created.
        Away from coasts this equals ``D * ops.lap`` with the cos(lat) area
        metric; the coefficients were capped at the stability limit.
        """
        o, land = self.ops, self.land
        fx = (self._gw_kx * (o.rollx(water, -1) - water)
              * land * o.rollx(land, -1))
        water = water + fx - o.rollx(fx, 1)
        fy = (self._gw_ky * (o.shifty(water, -1) - water)
              * land * o.shifty(land, -1) * self._gw_face_north)
        south = o.shifty(fy, 1) * self._gw_south_face_mask
        return water + (fy - south) * o.invcoslat

    def _route_runoff(self, runoff_mm):
        """Move this step's runoff (mm) to the downslope target of each cell.

        Returns the water arriving in every cell in mm: on land it is added
        to the ground-water store, on ocean cells it is the river discharge
        into the sea. Runoff from depressions has no target and is counted
        as endorheic drainage.
        """
        xp, n = self.xp, self.nlat * self.nlon
        mass = (runoff_mm.astype(xp.float64) * self._cell_area_m2).ravel()
        received = xp.bincount(self.river_target, weights=mass,
                               minlength=n + 1)
        received_cells = received[:n].reshape(self.nlat, self.nlon)
        per_second = 1.0 / (1000.0 * self.dt)   # kg per step -> m^3/s
        self._river_budget["river_discharge_to_ocean_m3_s"] = (
            (received_cells * self.ocean).sum() * per_second)
        self._river_budget["endorheic_drainage_m3_s"] = received[n] * per_second
        return (received_cells / self._cell_area_m2).astype(xp.float32)

    def _update_cloud_cover(self, diagnostic_cloud, rh, div):
        """Relax cloud cover toward diagnostics and clear unsupported cloud."""
        xp, p, dt = self.xp, self.cfg.physics, self.dt
        mt = p.moisture_transport
        precip_cloud_scale = float(mt.precip_cloud_diagnostic_scale_mmh)
        if getattr(diagnostic_cloud, "ndim", 2) == 3:
            diagnostic = xp.maximum(xp.max(diagnostic_cloud, axis=0),
                                    xp.clip(self.precip / precip_cloud_scale,
                                            0, 1))
            rh_ref = xp.max(rh, axis=0)
            div_ref = xp.max(div, axis=0)
        else:
            diagnostic = xp.maximum(diagnostic_cloud,
                                    xp.clip(self.precip / precip_cloud_scale,
                                            0, 1))
            rh_ref = rh
            div_ref = div

        tau_form = max(float(p.tau_cloud_form), dt)
        form = 1.0 - _np.exp(-dt / tau_form)
        cloud = self.cloud + form * (diagnostic - self.cloud)

        tau_dissip = max(float(p.tau_cloud_dissip), dt)
        clear_rh = float(p.cloud_clear_rh)
        dry = xp.clip((clear_rh - rh_ref) / max(clear_rh, 1e-3), 0, 1)
        subsidence = xp.clip(
            float(mt.cloud_subsidence_scale) * xp.maximum(div_ref, 0),
            0, float(mt.cloud_subsidence_clear_max))
        precip_scale = max(float(p.cloud_precip_scale), 1e-3)
        rainout = xp.clip(self.precip / precip_scale, 0, 2)
        clear_multiplier = (
            1.0
            + float(p.cloud_dry_clear) * dry
            + float(p.cloud_subsidence_clear) * subsidence
            + float(p.cloud_precip_clear) * rainout
        )
        cloud *= xp.exp(-dt * clear_multiplier / tau_dissip)
        return xp.clip(cloud, float(p.bounds.cloud_min),
                       float(p.bounds.cloud_max)).astype(xp.float32)

    def _diagnostic_cloud(self, rh, div):
        """Diagnose cloud fraction from humidity, then suppress subsidence."""
        xp, p = self.xp, self.cfg.physics
        rh_start = float(p.cloud_rh_start)
        rh_full = max(float(p.cloud_rh_full), rh_start + 1e-3)
        exponent = float(p.cloud_rh_exponent)
        cloud = xp.clip((rh - rh_start) / (rh_full - rh_start), 0, 1) ** exponent
        mt = p.moisture_transport
        return cloud * (1 - xp.clip(float(mt.cloud_subsidence_scale) * div,
                                    0, float(mt.cloud_subsidence_cover_max)))

    def _effective_condensation_rh(self, div):
        """Lower the condensation threshold only in sufficiently convergent flow."""
        xp, p = self.xp, self.cfg.physics
        conv_scale = max(float(p.convective_div_scale), 1e-8)
        conv = xp.clip(xp.maximum(-div, 0) / conv_scale, 0, 1)
        drop = float(p.convective_rh_drop_max)
        mt = p.moisture_transport
        return xp.clip(float(p.rh_crit) - drop * conv,
                       float(mt.relative_humidity_min),
                       float(mt.effective_rh_max))

    def _terrain_acceleration(self, u, v, k):
        """Return roughness drag, mountain blocking and contour deflection."""
        xp, p, tp, o = self.xp, self.cfg.physics, self.cfg.physics.topography, self.ops
        surface_drag = self._surface_atmospheric_drag()
        if not tp.enabled:
            return -surface_drag * u, -surface_drag * v
        attenuation = _np.exp(-float(self.levels_m[k]) /
                              max(float(tp.influence_height), 1.0))
        elev_factor = xp.clip(xp.maximum(self.elev, 0.0) /
                              max(float(tp.elevation_scale), 1.0), 0, 1)
        slope_factor = xp.clip(self.terrain_slope /
                               max(float(tp.slope_scale), 1e-5), 0, 1)
        influence = self.land * elev_factor * attenuation
        block = influence * slope_factor
        rough_drag = surface_drag * (1.0 + float(tp.drag_multiplier) * influence)

        norm = xp.maximum(self.terrain_slope, float(tp.slope_min))
        nx, ny = self.terrain_slope_x / norm, self.terrain_slope_y / norm
        cross = u * nx + v * ny
        hemi = xp.where(o.lat >= 0, 1.0, -1.0)
        tx, ty = -ny * hemi, nx * hemi
        along = u * tx + v * ty
        turn_sign = xp.where(xp.abs(along) > float(tp.turn_speed_threshold),
                             xp.sign(along), 1.0)
        block_rate = float(tp.blocking_rate) * block
        turn_rate = float(tp.deflection_rate) * block
        au = (-rough_drag * u - block_rate * cross * nx
              + turn_rate * xp.abs(cross) * turn_sign * tx)
        av = (-rough_drag * v - block_rate * cross * ny
              + turn_rate * xp.abs(cross) * turn_sign * ty)
        return au, av

    # ------------------------------------------------------------
    def _diag_surface(self):
        """冰/雪诊断(由温度决定)。"""
        xp, p = self.xp, self.cfg.physics
        if p.ice_albedo:
            bounds = p.bounds
            self.ice = (self.ocean * xp.clip(
                (float(bounds.sea_ice_threshold_k) - self.Ts)
                / float(bounds.sea_ice_transition_k),
                float(bounds.cloud_min), float(bounds.cloud_max))
                        ).astype(xp.float32)
            self.snow = (self.land *
                         xp.clip((float(bounds.snow_threshold_k) - self.Ts)
                                 / float(bounds.snow_transition_k),
                                 float(bounds.cloud_min),
                                 float(bounds.cloud_max))).astype(xp.float32)
        else:
            self.ice = xp.zeros_like(self.Ts)
            self.snow = xp.zeros_like(self.Ts)

    # ------------------------------------------------------------
    def step(self, nsteps=1):
        for _ in range(int(nsteps)):
            self._step_once()

    def _step_once(self):
        if self._primitive_enabled:
            return self._step_primitive_equations()
        return self._step_multilayer()

    def _refresh_primitive_diagnostics(self):
        """Refresh pressure, hydrostatic geopotential and geometric height."""
        xp, p = self.xp, self.cfg.physics
        state = hydrostatic_state(
            xp, self.T_layers, self.q_layers, self.surface_pressure,
            self._hyai_xp, self._hybi_xp, self.surface_geopotential)
        (self.pressure_interfaces_pa, self.pressure_layers_pa,
         self.geopotential_layers, self.geopotential_interfaces,
         self.virtual_temperature_layers) = state
        self.layer_pressure_thickness_pa = (
            self._d_hyai_xp + self._d_hybi_xp
            * self.surface_pressure[None, :, :]).astype(xp.float32)
        self.exner_layers = exner(xp, self.pressure_layers_pa)
        self.h_layers = (
            self.geopotential_layers / float(p.g_eff)).astype(xp.float32)

    def _primitive_pressure_gradient(self, geopotential=None,
                                     pressure=None, virtual_temperature=None):
        """Pressure-gradient force on constant-sigma surfaces.

        The second term transforms the hydrostatic geopotential gradient from
        a sigma surface to a constant-pressure surface.  Together with the
        surface geopotential in the hydrostatic integral this is the terrain
        pressure-gradient cancellation required for resting stratified air.
        """
        xp, p, o = self.xp, self.cfg.physics, self.ops
        geopotential = (self.geopotential_layers if geopotential is None
                        else geopotential)
        pressure = (self.pressure_layers_pa if pressure is None
                    else pressure)
        virtual_temperature = (
            self.virtual_temperature_layers if virtual_temperature is None
            else virtual_temperature)
        phi_x, phi_y = o.gradient(geopotential)
        logp_x, logp_y = o.gradient(xp.log(pressure))
        # Evaluate alpha*grad(p) as Rd*Tv*grad(log p) with the same discrete
        # gradient used for Phi. For an isothermal hydrostatic atmosphere
        # Phi = Phi_s + Rd*T*log(ps/p), so terrain terms cancel to roundoff
        # instead of leaving the classic sigma-coordinate mountain error.
        coefficient = RD * virtual_temperature
        return (phi_x + coefficient * logp_x,
                phi_y + coefficient * logp_y)

    def _primitive_boundary_drag_rate(self, geopotential_layers):
        """Surface-stress damping rate distributed through the lowest layers."""
        xp, p, tp = (self.xp, self.cfg.physics,
                     self.cfg.physics.topography)
        z_agl = xp.maximum(
            geopotential_layers / float(p.g_eff)
            - self.surface_height[None, :, :], 0.0)
        depth = max(float(tp.influence_height), 1.0)
        vertical_weight = xp.exp(-z_agl / depth)
        slope_roughness = xp.clip(
            self.terrain_slope / max(float(tp.slope_scale), 1.0e-6), 0, 1)
        roughness = (1.0 + self.land * float(tp.drag_multiplier)
                     * slope_roughness)
        return (self._surface_atmospheric_drag()[None, :, :]
                * vertical_weight * roughness[None, :, :])

    def _global_mean(self, field):
        """Area-weighted global mean of a [nlat, nlon] field (float64)."""
        return (field.astype(self.xp.float64)
                * self._area_weights.astype(self.xp.float64)).sum()

    def _sponge_diffusion(self, field, base_diffusivity):
        """Extra horizontal diffusion in the upper sponge layers."""
        k0 = self._sponge_first_level
        extra = (self._sponge_diffusion_factor * float(base_diffusivity)
                 * self._sponge_weights_xp[k0:])
        if k0 >= field.shape[0] or not float(base_diffusivity):
            return field
        top = field[k0:]
        field[k0:] = top + self.dt * extra * self.ops.lap(top)
        return field

    def _rayleigh_sponge(self, u, v):
        """Implicit Rayleigh friction in the sponge, on eddies by default."""
        k0 = self._sponge_first_level
        if k0 >= u.shape[0]:
            return u, v
        factor = 1.0 / (1.0 + self.dt * self._sponge_rate_xp[k0:])
        if bool(self.cfg.physics.dynamics.upper_damping.damp_zonal_mean):
            u[k0:] = u[k0:] * factor
            v[k0:] = v[k0:] * factor
            return u, v
        mean_u = u[k0:].mean(axis=-1, keepdims=True)
        mean_v = v[k0:].mean(axis=-1, keepdims=True)
        u[k0:] = mean_u + (u[k0:] - mean_u) * factor
        v[k0:] = mean_v + (v[k0:] - mean_v) * factor
        return u, v

    def _divergence_damping(self, u, v):
        """Second-order divergence damping ``nu grad(div V)``."""
        o = self.ops
        div = o.divergence(u, v)
        grad_x, grad_y = o.gradient(div)
        nu = self._divergence_damping_nu * self._divergence_damping_profile_xp
        return nu * grad_x, nu * grad_y

    def _vertical_mixing(self, T, q, u, v, pressure_interfaces,
                         layer_mass, geopotential, surface_pressure):
        """Implicit turbulent mixing of dry static energy, moisture and wind.

        The exchange coefficient follows a prescribed boundary-layer profile
        of eddy diffusivity.  Dry static energy ``cp T + Phi`` is mixed with
        the geopotential held fixed, so column enthalpy is conserved exactly
        and a well-mixed layer ends up dry adiabatic.
        """
        xp, cpx, dt = self.xp, self.cfg.physics.column_physics, self.dt
        free = float(cpx.free_diffusivity_m2s)
        pbl = float(cpx.pbl_diffusivity_m2s)
        depth = max(float(cpx.diffusion_depth_pa), 1.0)
        p_interior = pressure_interfaces[1:-1]
        diffusivity = free + (pbl - free) * xp.exp(
            -(surface_pressure[None, :, :] - p_interior) / depth)
        # Local stability: a gradient Richardson number from the layer
        # differences suppresses mixing through stably stratified, weakly
        # sheared air and leaves neutral or unstable layers fully mixed.
        g = float(self.cfg.physics.g_eff)
        theta = T / exner(xp, 0.5 * (pressure_interfaces[:-1]
                                     + pressure_interfaces[1:]))
        dz = xp.maximum((geopotential[1:] - geopotential[:-1]) / g, 1.0)
        shear_squared = xp.maximum(
            ((u[1:] - u[:-1]) ** 2 + (v[1:] - v[:-1]) ** 2) / (dz * dz),
            float(cpx.min_shear_squared_s2))
        richardson = (g / (0.5 * (theta[1:] + theta[:-1]))
                      * (theta[1:] - theta[:-1]) / dz / shear_squared)
        diffusivity = diffusivity / (
            1.0 + float(cpx.richardson_damping) * xp.maximum(richardson, 0.0))
        tv = T * (1.0 + 0.608 * q)
        tv_interface = 0.5 * (tv[:-1] + tv[1:])
        density = p_interior / xp.maximum(RD * tv_interface, 1.0)
        mass_distance = 0.5 * (layer_mass[:-1] + layer_mass[1:])
        exchange = density * density * diffusivity / mass_distance

        static_energy = CP * T + geopotential
        static_energy = implicit_vertical_diffusion(
            xp, static_energy, layer_mass, exchange, dt)
        T_mixed = (static_energy - geopotential) / CP
        q_mixed = (implicit_vertical_diffusion(xp, q, layer_mass, exchange, dt)
                   if self.cfg.physics.moisture else q)
        kinetic_before = 0.5 * (u * u + v * v)
        u_mixed = implicit_vertical_diffusion(xp, u, layer_mass, exchange, dt)
        v_mixed = implicit_vertical_diffusion(xp, v, layer_mass, exchange, dt)
        if bool(self.cfg.physics.dynamics.frictional_heating):
            T_mixed += (kinetic_before
                        - 0.5 * (u_mixed * u_mixed + v_mixed * v_mixed)) / CP
        return T_mixed, q_mixed, u_mixed, v_mixed

    def _step_primitive_equations(self):
        """Advance the moist hydrostatic primitive equations in hybrid space.

        Order of operations:

        1. surface energy balance and turbulent surface fluxes from the old
           state;
        2. adiabatic dynamics: forward hybrid continuity, flux-form transport
           of potential temperature, moisture and momentum, backward
           pressure-gradient force, Coriolis rotation, implicit surface
           stress and upper sponge with frictional heating, then a global
           total-energy fixer;
        3. column physics: boundary-layer deposition of surface fluxes,
           radiative heating, Newtonian relaxation, implicit vertical mixing
           of dry static energy, dry convective adjustment and large-scale
           condensation;
        4. bound checks that count, but normally never trigger, clipping.
        """
        xp, p, o, dt = self.xp, self.cfg.physics, self.ops, self.dt
        dyn, bounds = p.dynamics, p.bounds
        rt, sf, mt = p.radiation_transfer, p.surface_flux, p.moisture_transport
        cpx = p.column_physics
        g = float(p.g_eff)
        p_top = self.top_pressure_pa
        latent_heat = LV if p.moisture else 0.0

        u0, v0 = self.u_layers, self.v_layers
        T0, q0, ps0, Ts = (
            self.T_layers, self.q_layers, self.surface_pressure, self.Ts)
        low_u, low_v, low_T, low_q = u0[0], v0[0], T0[0], q0[0]
        low_p = self.pressure_layers_pa[0]
        wind = (xp.sqrt(low_u * low_u + low_v * low_v)
                + float(sf.wind_speed_floor_ms))

        # ---------------- surface radiation and turbulent exchange ----------
        Q_sw, decl, sub_lon = insolation(
            xp, o.lat, o.lon_rad, self.t, p.S0, p.diurnal_cycle)
        self.subsolar = (float(decl), float(sub_lon))
        greenhouse_q = xp.clip(
            low_q / float(sf.humidity_greenhouse_scale), 0, 1)
        if p.radiation:
            albedo_surface = (
                self.ocean * (p.alb_ocean * (1 - self.ice)
                              + p.alb_ice * self.ice)
                + self.land * (p.alb_land * (1 - self.snow)
                               + p.alb_snow * self.snow))
            albedo = xp.clip(
                albedo_surface + p.alb_cloud * self.cloud,
                0, float(rt.albedo_max))
            SW_surface = (
                Q_sw * (1 - albedo) * float(rt.surface_sw_fraction))
            SW_air = (
                Q_sw * (1 - float(rt.cloud_sw_absorption) * self.cloud)
                * float(rt.air_sw_fraction))
            LW_up = float(rt.surface_emissivity) * SIGMA * Ts ** 4
            emissivity_down = xp.clip(
                float(rt.down_emissivity_base)
                + float(rt.down_emissivity_humidity) * greenhouse_q
                + float(rt.down_emissivity_cloud) * self.cloud,
                0, float(rt.down_emissivity_max))
            LW_down = emissivity_down * SIGMA * low_T ** 4
            olr_factor = xp.clip(
                float(rt.olr_factor_base)
                - float(rt.olr_factor_humidity) * greenhouse_q
                - float(rt.olr_factor_cloud) * self.cloud,
                float(rt.olr_factor_min), float(rt.olr_factor_max))
            OLR = olr_factor * SIGMA * low_T ** 4
        else:
            SW_surface = SW_air = LW_up = LW_down = OLR = xp.zeros_like(low_T)

        density0 = low_p / xp.maximum(
            RD * low_T * (1.0 + 0.608 * low_q), 1.0)
        sensible = (float(sf.sensible_heat_coeff) * density0 * CP * wind
                    * (Ts - low_T))
        if p.moisture:
            evaporation, land_evaporation = self._surface_evaporation(
                Ts, low_q, wind, low_p, low_T)
            latent = LV * evaporation
        else:
            evaporation = land_evaporation = latent = xp.zeros_like(low_T)

        surface_capacity = (
            self.ocean * (RHO_W * CW * p.mld) + self.land * C_LAND)
        Ts = Ts + dt * (
            SW_surface + LW_down - LW_up - sensible - latent
        ) / surface_capacity
        Ts = xp.where(
            self.ocean > 0.5,
            xp.maximum(Ts, float(bounds.ocean_freezing_min_temp_k)), Ts)

        # ---------------- adiabatic dynamics --------------------------------
        # Potential temperature is the transported thermodynamic variable:
        # adiabatic compression and expansion are then contained in the
        # change of the Exner function along the trajectory instead of an
        # explicit kappa*T*omega/p source term.
        theta0 = T0 / self.exner_layers
        energy0 = column_total_energy(
            xp, u0, v0, T0, q0, self.layer_pressure_thickness_pa / g,
            self.surface_geopotential, (ps0 - p_top) / g, latent_heat)

        # Hybrid continuity: prognose ps and diagnose the shared interface
        # mass flux. Both physical boundaries are impermeable.
        ps_tendency, div_mass, interface_flux, dp0 = hybrid_mass_flux(
            xp, o, u0, v0, ps0, self._hyai_xp, self._hybi_xp)
        ps_diff = float(dyn.surface_pressure_diffusivity)
        if ps_diff:
            ps_tendency = ps_tendency + ps_diff * o.lap(ps0)
        ps1 = ps0 + dt * ps_tendency
        dp1 = self._d_hyai_xp + self._d_hybi_xp * ps1[None, :, :]

        def transport(field, diffusivity):
            return mass_consistent_transport(
                xp, o, field, u0, v0, dp0, dp1, div_mass, interface_flux,
                diffusivity, dt)

        u = transport(u0, p.visc)
        v = transport(v0, p.visc)
        theta = transport(theta0, 0.0)
        q = transport(q0, p.diff_q) if p.moisture else q0.copy()
        if self.sponge_active:
            u = self._sponge_diffusion(u, p.visc)
            v = self._sponge_diffusion(v, p.visc)
            if p.moisture:
                q = self._sponge_diffusion(q, p.diff_q)
        q = o.polar_filter(q)

        pressure_interfaces1, pressure_layers1 = hybrid_pressures(
            xp, self._hyai_xp, self._hybi_xp, ps1)
        exner1 = exner(xp, pressure_layers1)
        T = theta * exner1
        # Smoothing operators act on T, not theta: along a terrain-following
        # surface theta varies with pressure even when T is uniform, so
        # diffusing or filtering theta would heat and cool resting air over
        # mountains and drive spurious winds.
        if float(p.diff_T):
            T += dt * float(p.diff_T) * o.lap(T)
        if self.sponge_active:
            T = self._sponge_diffusion(T, p.diff_T)
        T = o.polar_filter(T)

        # Hydrostatic pressure-gradient force, metric terms, divergence
        # damping, Coriolis force and boundary-layer stress. Use the newly
        # predicted mass/thermodynamic state for the PGF: this
        # forward-backward ordering keeps the external hydrostatic gravity
        # mode bounded, whereas evaluating both ps and wind tendencies at the
        # old state is a forward-Euler oscillator and unconditionally unstable.
        pressure_state = hydrostatic_state(
            xp, T, q, ps1, self._hyai_xp, self._hybi_xp,
            self.surface_geopotential)
        geopotential1 = pressure_state[2]
        pgf_x, pgf_y = self._primitive_pressure_gradient(
            geopotential1, pressure_layers1, pressure_state[4])
        u += dt * (-pgf_x + o.tanl * u0 * v0)
        v += dt * (-pgf_y - o.tanl * u0 * u0)
        if self.divergence_damping_active:
            damp_x, damp_y = self._divergence_damping(u0, v0)
            u += dt * damp_x
            v += dt * damp_y
        u, v = o.coriolis_rotate(u, v, dt)

        # Linear damping terms are integrated implicitly, so no time-step
        # restriction follows from strong drag or a short sponge timescale.
        kinetic_before = 0.5 * (u * u + v * v)
        damping = 1.0 / (1.0 + dt * self._primitive_boundary_drag_rate(
            geopotential1))
        u = u * damping
        v = v * damping
        if self.sponge_active:
            u, v = self._rayleigh_sponge(u, v)
        if bool(dyn.frictional_heating):
            T += (kinetic_before - 0.5 * (u * u + v * v)) / CP
        u = o.polar_filter(u)
        v = o.polar_filter(v)

        layer_mass = xp.maximum(dp1, xp.float32(1.0)) / g
        column_mass = (ps1 - p_top) / g
        energy1 = column_total_energy(
            xp, u, v, T, q, layer_mass, self.surface_geopotential,
            column_mass, latent_heat)
        dynamics_error = self._global_mean(energy1 - energy0)
        fixer_dT = xp.zeros((), dtype=xp.float64)
        if bool(dyn.energy_fixer):
            # The discrete adiabatic step is not exactly energy conserving.
            # Return the global residual as a uniform temperature increment,
            # as CAM's energy fixer does, so the climate does not drift. The
            # increment is accumulated in float64 and applied once it is
            # resolvable by the float32 state (a few 1e-5 K at 300 K).
            self._energy_fixer_debt_k = (
                self._energy_fixer_debt_k
                - dynamics_error / (CP * self._global_mean(column_mass)))
            payable = (xp.abs(self._energy_fixer_debt_k)
                       >= float(dyn.energy_fixer_min_increment_k))
            fixer_dT = xp.where(payable, self._energy_fixer_debt_k, 0.0)
            fixer_dT = fixer_dT.astype(xp.float32).astype(xp.float64)
            self._energy_fixer_debt_k = self._energy_fixer_debt_k - fixer_dT
            T = T + fixer_dT.astype(xp.float32)
            energy1 = energy1 + CP * fixer_dT * column_mass

        # ---------------- column physics ------------------------------------
        omega = pressure_vertical_velocity(
            xp, o, u0, v0, ps0, ps_tendency, interface_flux, self._hybi_xp)

        # Surface fluxes warm and moisten a boundary layer rather than one
        # thin sigma layer; column radiative terms are spread with mass
        # weights through the troposphere.
        pbl_weights = layer_mass * xp.exp(
            -(ps1[None, :, :] - pressure_layers1)
            / max(float(cpx.pbl_depth_pa), 1.0))
        pbl_weights = pbl_weights / pbl_weights.sum(axis=0, keepdims=True)
        radiative_mask = pressure_layers1 > float(cpx.radiative_top_pa)
        radiative_weights = layer_mass * radiative_mask
        radiative_sum = radiative_weights.sum(axis=0, keepdims=True)
        radiative_weights = xp.where(
            radiative_sum > 0.0,
            radiative_weights / xp.maximum(radiative_sum, xp.float32(1e-6)),
            pbl_weights)
        surface_heating = (
            sensible + float(rt.air_longwave_coupling) * (LW_up - LW_down))
        column_heating = SW_air - OLR
        T += dt * (surface_heating[None, :, :] * pbl_weights
                   + column_heating[None, :, :] * radiative_weights) / (
                       layer_mass * CP)
        if p.moisture:
            q += dt * evaporation[None, :, :] * pbl_weights / layer_mass

        relaxation_heating = xp.zeros((), dtype=xp.float64)
        if p.radiation:
            seasonal = (
                float(rt.seasonal_temp_base_k)
                - float(rt.seasonal_temp_pole_delta_k) * xp.sin(o.lat) ** 2
                + float(rt.seasonal_temp_amp_k) * xp.sin(o.lat)
                * _np.sin(_np.radians(self.subsolar[0]) * 2))
            z_agl = xp.maximum(
                geopotential1 / g - self.surface_height[None, :, :], 0.0)
            equilibrium = self._equilibrium_temperature_profile(
                seasonal, z_agl)
            relaxation = (equilibrium - T) / float(p.tau_relax_T)
            T += dt * relaxation
            relaxation_heating = (relaxation * layer_mass * CP).sum(axis=0)

        if bool(cpx.vertical_diffusion) and self.nz > 1:
            T, q, u, v = self._vertical_mixing(
                T, q, u, v, pressure_interfaces1, layer_mass,
                geopotential1, ps1)
        if bool(cpx.dry_adjustment) and self.nz > 1:
            T, q_adjusted = dry_convective_adjustment(
                xp, T, q if p.moisture else None, layer_mass, exner1,
                passes=int(cpx.dry_adjustment_passes))
            if q_adjusted is not None:
                q = q_adjusted

        # Pressure-dependent saturation adjustment and column water budget.
        div = o.divergence(u, v)
        rain_mass = xp.zeros_like(ps1)
        if p.moisture:
            saturation = qsat(xp, T, pressure_layers1)
            rh_effective = self._effective_condensation_rh(div)
            condensed = xp.maximum(
                q - rh_effective * saturation, 0) * (
                    1 - _np.exp(-dt / float(p.tau_cond)))
            q *= 1 - dt * float(mt.subsidence_drying_coeff) * xp.clip(
                div, 0, float(mt.subsidence_divergence_max))
            q -= condensed
            T += (LV / CP) * condensed
            rain_mass = (condensed * layer_mass).sum(axis=0)
            self.precip = (rain_mass / dt * 3600.0).astype(xp.float32)
            self._update_ground_water(land_evaporation, rain_mass)
            relative_humidity = xp.clip(
                q / qsat(xp, T, pressure_layers1),
                float(mt.relative_humidity_min),
                float(bounds.relative_humidity_max))
            cloud_diagnostic = self._diagnostic_cloud(
                relative_humidity, div)
            self.cloud = self._update_cloud_cover(
                cloud_diagnostic, relative_humidity, div)

        # Water crossing the lower boundary changes total atmospheric mass;
        # precipitation leaves it. This term is small but closes the budget.
        ps1 = ps1 + g * (dt * evaporation - rain_mass)
        energy2 = column_total_energy(
            xp, u, v, T, q, layer_mass, self.surface_geopotential,
            (ps1 - p_top) / g, latent_heat)
        physics_change = self._global_mean(energy2 - energy1)
        forcing = self._global_mean(
            surface_heating + column_heating + LV * evaporation
            + relaxation_heating)

        # ---------------- bounds: a safety net that should stay idle --------
        umax = float(p.umax)
        t_min, t_max = float(bounds.air_temp_min_k), float(bounds.air_temp_max_k)
        q_min, q_max = float(bounds.humidity_min), float(bounds.humidity_max)
        ps_min = float(dyn.min_surface_pressure_pa)
        ps_max = float(dyn.max_surface_pressure_pa)
        self._limiter_counts = {
            "wind": ((xp.abs(u) > umax) | (xp.abs(v) > umax)).sum(),
            "temperature": ((T < t_min) | (T > t_max)).sum(),
            "humidity": ((q < q_min) | (q > q_max)).sum(),
            "surface_pressure": ((ps1 < ps_min) | (ps1 > ps_max)).sum(),
        }
        self.u_layers = xp.clip(u, -umax, umax).astype(xp.float32)
        self.v_layers = xp.clip(v, -umax, umax).astype(xp.float32)
        self.T_layers = xp.clip(T, t_min, t_max).astype(xp.float32)
        self.q_layers = xp.clip(q, q_min, q_max).astype(xp.float32)
        self.surface_pressure = xp.clip(ps1, ps_min, ps_max).astype(xp.float32)
        self.Ts = xp.clip(
            self._advance_ocean(Ts, self.u_layers[0], self.v_layers[0]),
            float(bounds.surface_temp_min_k),
            float(bounds.surface_temp_max_k)).astype(xp.float32)

        self._budget = {
            "total_energy_j_m2": self._global_mean(energy2),
            "kinetic_energy_j_m2": self._global_mean(
                (layer_mass * 0.5 * (u * u + v * v)).sum(axis=0)),
            "latent_energy_j_m2": self._global_mean(
                (layer_mass * latent_heat * q).sum(axis=0)),
            "dynamics_energy_error_w_m2": dynamics_error / dt,
            "energy_fixer_dT_k": fixer_dT,
            "energy_fixer_pending_dT_k": self._energy_fixer_debt_k,
            "physics_energy_change_w_m2": physics_change / dt,
            "physics_forcing_w_m2": forcing,
        }

        self._refresh_primitive_diagnostics()
        self.omega_layers = omega.astype(xp.float32)
        mu = xp.maximum(
            self.surface_pressure - p_top, xp.float32(1.0))
        self.sigma_dot_interfaces = (
            interface_flux / mu[None, :, :]).astype(xp.float32)
        vertical_velocity = (
            -RD * self.virtual_temperature_layers * omega
            / xp.maximum(g * self.pressure_layers_pa, 1.0))
        # At a terrain-following impermeable lower boundary, geometric w is
        # u·grad(zs), not zero. sigma_dot is exactly zero there.
        terrain_w = (
            self.u_layers[0] * self.terrain_slope_x
            + self.v_layers[0] * self.terrain_slope_y)
        vertical_velocity[0] = terrain_w
        self.w_layers = xp.clip(
            vertical_velocity, -float(p.vertical.w_max),
            float(p.vertical.w_max)).astype(xp.float32)
        self._sync_surface_views()
        self._diag_surface()
        self.t += _dt.timedelta(seconds=dt)
        self.step_count += 1

    def _step_once_legacy(self):
        xp, p, o, dt = self.xp, self.cfg.physics, self.ops, self.dt
        bounds = p.bounds
        pressure = p.pressure
        rt = p.radiation_transfer
        sf = p.surface_flux
        mt = p.moisture_transport
        u, v, h, Ta, q, Ts = self.u, self.v, self.h, self.Ta, self.q, self.Ts
        spd = xp.sqrt(u * u + v * v) + float(sf.wind_speed_floor_ms)

        # ============ 辐射 ============
        Q_sw, decl, sub_lon = insolation(xp, o.lat, o.lon_rad, self.t,
                                         p.S0, p.diurnal_cycle)
        self.subsolar = (float(decl), float(sub_lon))
        qs_a = qsat(xp, Ta)
        rh = xp.clip(q / qs_a, float(mt.relative_humidity_min),
                     float(bounds.relative_humidity_max))
        gq = xp.clip(q / float(sf.humidity_greenhouse_scale), 0, 1)

        if p.radiation:
            alb_sfc = (self.ocean * (p.alb_ocean * (1 - self.ice) + p.alb_ice * self.ice)
                       + self.land * (p.alb_land * (1 - self.snow) + p.alb_snow * self.snow))
            alb = xp.clip(alb_sfc + p.alb_cloud * self.cloud, 0,
                          float(rt.albedo_max))
            SW_sfc = Q_sw * (1 - alb) * float(rt.surface_sw_fraction)
            SW_air = (Q_sw * (1 - float(rt.cloud_sw_absorption) * self.cloud)
                      * float(rt.air_sw_fraction))
            LW_up = float(rt.surface_emissivity) * SIGMA * Ts ** 4
            eps_dn = xp.clip(float(rt.down_emissivity_base)
                             + float(rt.down_emissivity_humidity) * gq
                             + float(rt.down_emissivity_cloud) * self.cloud,
                             0, float(rt.down_emissivity_max))
            LW_dn = eps_dn * SIGMA * Ta ** 4
            olr_f = xp.clip(float(rt.olr_factor_base)
                            - float(rt.olr_factor_humidity) * gq
                            - float(rt.olr_factor_cloud) * self.cloud,
                            float(rt.olr_factor_min),
                            float(rt.olr_factor_max))
            OLR_air = olr_f * SIGMA * Ta ** 4
        else:
            SW_sfc = SW_air = LW_up = LW_dn = OLR_air = xp.zeros_like(Ta)

        # ============ 地表通量 ============
        SH = float(sf.sensible_heat_coeff) * RHO_A * CP * spd * (Ts - Ta)
        if p.moisture:
            E, land_E = self._surface_evaporation(Ts, q, spd)
            LE = LV * E
        else:
            E = land_E = LE = xp.zeros_like(Ta)

        # 下垫面热容: 海洋混合层 vs 陆面薄层
        C_sfc = self.ocean * (RHO_W * CW * p.mld) + self.land * C_LAND
        Ts = Ts + dt * (SW_sfc + LW_dn - LW_up - SH - LE) / C_sfc
        # 海冰下海水温度下限
        Ts = xp.where(self.ocean > 0.5,
                      xp.maximum(Ts, float(bounds.ocean_freezing_min_temp_k)),
                      Ts)

        # ============ 大气热力 + 水汽 ============
        Ta = o.adv_diff_step(Ta, u, v, p.diff_T, dt)
        Ta = Ta + dt * (SH + SW_air + float(rt.air_longwave_coupling)
                        * (LW_up - LW_dn) - OLR_air) / (MCOL * CP)
        if p.radiation:  # 向辐射平衡弱弛豫(保底约束, 防漂移)
            Teq = (float(rt.seasonal_temp_base_k)
                   - float(rt.seasonal_temp_pole_delta_k) * xp.sin(o.lat) ** 2
                   + float(rt.seasonal_temp_amp_k) * xp.sin(o.lat)
                   * _np.sin(_np.radians(self.subsolar[0]) * 2))
            Ta = Ta + dt / p.tau_relax_T * (Teq - Ta)

        # 风场辐散 (动力学与凝结共用)
        div = o.divergence(u, v)

        if p.moisture:
            q = o.adv_diff_step(q, u, v, p.diff_q, dt)
            q = q + dt * E / MCOL
            qs_a = qsat(xp, Ta)
            # 辐合区(低压)对流增强: 有效凝结阈值降低 -> 降水集中于 ITCZ/气旋
            rh_eff = self._effective_condensation_rh(div)
            exc = xp.maximum(q - rh_eff * qs_a, 0)
            # 辐散区(高压)下沉干燥 -> 副热带晴空/沙漠带
            q = q * (1 - dt * float(mt.subsidence_drying_coeff)
                     * xp.clip(div, 0, float(mt.subsidence_divergence_max)))
            dq = exc * (1 - _np.exp(-dt / p.tau_cond))
            q = q - dq
            Ta = Ta + (LV / CP) * dq                      # 凝结潜热
            self.precip = (dq * MCOL / dt * 3600.0).astype(xp.float32)  # mm/h
            self._update_ground_water(land_E, dq * MCOL)
            rh = xp.clip(q / qsat(xp, Ta), float(mt.relative_humidity_min),
                         float(bounds.relative_humidity_max))
            cl = self._diagnostic_cloud(rh, div)
            self.cloud = self._update_cloud_cover(cl, rh, div)
            q = xp.clip(q, float(bounds.humidity_min),
                        float(bounds.humidity_max))

        # ============ 动力学 (湿浅水) ============
        h_eq = p.H0 - p.beta_T * (Ta - Ta.mean())
        h = o.adv_diff_step(h, u, v,
                            p.visc * float(pressure.thickness_diffusion_factor),
                            dt)
        h = h + dt * (-h * div + (h_eq - h) / p.tau_h)
        h = xp.clip(h, float(pressure.thickness_min_factor) * p.H0,
                    float(pressure.thickness_max_factor) * p.H0)

        dhdx, dhdy = o.gradient(h)
        u = o.adv_diff_step(u, u, v, p.visc, dt)
        v = o.adv_diff_step(v, u, v, p.visc, dt)
        surface_drag = self._surface_atmospheric_drag()
        u = u + dt * (-p.g_eff * dhdx - surface_drag * u + o.tanl * u * v)
        v = v + dt * (-p.g_eff * dhdy - surface_drag * v - o.tanl * u * u)
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
            uo, vo = o.coriolis_rotate(
                uo, vo, dt * float(p.ocean_transport.ekman_coriolis_factor))
            open_o = self.ocean * (1 - self.ice)
            max_current = float(bounds.ocean_current_max_ms)
            self.uo = (xp.clip(uo, -max_current, max_current)
                       * open_o).astype(xp.float32)
            self.vo = (xp.clip(vo, -max_current, max_current)
                       * open_o).astype(xp.float32)
            # 洋流输送热量
            Ts = xp.where(self.ocean > 0.5,
                          o.adv_diff_step(
                              Ts, self.uo, self.vo,
                              float(p.ocean_transport.surface_heat_diffusivity),
                              dt), Ts)

        # ============ 极区滤波 + 限幅 ============
        u, v, h = o.polar_filter(u), o.polar_filter(v), o.polar_filter(h)
        Ta, q = o.polar_filter(Ta), o.polar_filter(q)
        self.u, self.v = u.astype(xp.float32), v.astype(xp.float32)
        self.h = h.astype(xp.float32)
        self.Ta = xp.clip(Ta, float(bounds.air_temp_min_k),
                          float(bounds.air_temp_max_k)).astype(xp.float32)
        self.q = q.astype(xp.float32)
        self.Ts = xp.clip(Ts, float(bounds.surface_temp_min_k),
                          float(bounds.surface_temp_max_k)).astype(xp.float32)
        self._diag_surface()

        self.t += _dt.timedelta(seconds=self.dt)
        self.step_count += 1

    # ------------------------------------------------------------
    def _step_multilayer(self):
        xp, p, o, dt = self.xp, self.cfg.physics, self.ops, self.dt
        bounds = p.bounds
        pressure = p.pressure
        rt = p.radiation_transfer
        sf = p.surface_flux
        mt = p.moisture_transport
        u, v = self.u_layers, self.v_layers
        h, Ta, q, Ts = self.h_layers, self.T_layers, self.q_layers, self.Ts
        u0, v0, Ta0, q0 = u[0], v[0], Ta[0], q[0]
        spd = xp.sqrt(u0 * u0 + v0 * v0) + float(sf.wind_speed_floor_ms)

        Q_sw, decl, sub_lon = insolation(xp, o.lat, o.lon_rad, self.t,
                                         p.S0, p.diurnal_cycle)
        self.subsolar = (float(decl), float(sub_lon))
        gq = xp.clip(q0 / float(sf.humidity_greenhouse_scale), 0, 1)
        if p.radiation:
            alb_sfc = (self.ocean * (p.alb_ocean * (1 - self.ice)
                       + p.alb_ice * self.ice)
                       + self.land * (p.alb_land * (1 - self.snow)
                       + p.alb_snow * self.snow))
            alb = xp.clip(alb_sfc + p.alb_cloud * self.cloud, 0,
                          float(rt.albedo_max))
            SW_sfc = Q_sw * (1 - alb) * float(rt.surface_sw_fraction)
            SW_air = (Q_sw * (1 - float(rt.cloud_sw_absorption) * self.cloud)
                      * float(rt.air_sw_fraction))
            LW_up = float(rt.surface_emissivity) * SIGMA * Ts ** 4
            eps_dn = xp.clip(float(rt.down_emissivity_base)
                             + float(rt.down_emissivity_humidity) * gq
                             + float(rt.down_emissivity_cloud) * self.cloud,
                             0, float(rt.down_emissivity_max))
            LW_dn = eps_dn * SIGMA * Ta0 ** 4
            olr_f = xp.clip(float(rt.olr_factor_base)
                            - float(rt.olr_factor_humidity) * gq
                            - float(rt.olr_factor_cloud) * self.cloud,
                            float(rt.olr_factor_min),
                            float(rt.olr_factor_max))
            OLR_air = olr_f * SIGMA * Ta0 ** 4
        else:
            SW_sfc = SW_air = LW_up = LW_dn = OLR_air = xp.zeros_like(Ta0)

        SH = float(sf.sensible_heat_coeff) * RHO_A * CP * spd * (Ts - Ta0)
        if p.moisture:
            E, land_E = self._surface_evaporation(Ts, q0, spd)
            LE = LV * E
        else:
            E = land_E = LE = xp.zeros_like(Ta0)
        C_sfc = self.ocean * (RHO_W * CW * p.mld) + self.land * C_LAND
        Ts = Ts + dt * (SW_sfc + LW_dn - LW_up - SH - LE) / C_sfc
        Ts = xp.where(self.ocean > 0.5,
                      xp.maximum(Ts, float(bounds.ocean_freezing_min_temp_k)),
                      Ts)

        # Horizontal thermodynamics are integrated independently on each level.
        seasonal = (float(rt.seasonal_temp_base_k)
                    - float(rt.seasonal_temp_pole_delta_k)
                    * xp.sin(o.lat) ** 2
                    + float(rt.seasonal_temp_amp_k) * xp.sin(o.lat)
                    * _np.sin(_np.radians(self.subsolar[0]) * 2))
        lowest_mass = max(float(self.layer_mass_fractions[0]),
                          float(p.vertical.lowest_layer_mass_min))
        Ta_adv = o.adv_diff_step(Ta, u, v, p.diff_T, dt)
        q_adv = (o.adv_diff_step(q, u, v, p.diff_q, dt)
                 if p.moisture else q)
        div = o.divergence(u, v)
        Ta_out, q_out = [], []
        for k in range(self.nz):
            Tk = Ta_adv[k]
            if k == 0:
                Tk += dt * (SH + SW_air + float(rt.air_longwave_coupling)
                            * (LW_up - LW_dn)
                            - OLR_air) / (MCOL * lowest_mass * CP)
            if p.radiation:
                Teq = seasonal - float(p.vertical.lapse_rate) * self.levels_m[k]
                Tk += dt / p.tau_relax_T * (Teq - Tk)
            Ta_out.append(Tk)
            if p.moisture:
                qk = q_adv[k]
                if k == 0:
                    qk += dt * E / (MCOL * lowest_mass)
            else:
                qk = q[k]
            q_out.append(qk)
        Ta = xp.stack(Ta_out, axis=0)
        q = xp.stack(q_out, axis=0)

        if p.moisture:
            qs_a = qsat(xp, Ta)
            rh_eff = self._effective_condensation_rh(div)
            dq = xp.maximum(q - rh_eff * qs_a, 0) * (
                1 - _np.exp(-dt / p.tau_cond))
            q *= 1 - dt * float(mt.subsidence_drying_coeff) * xp.clip(
                div, 0, float(mt.subsidence_divergence_max))
            q -= dq
            Ta += (LV / CP) * dq
            column_dq = (dq * self._mass3).sum(axis=0)
            self.precip = (column_dq * MCOL / dt * 3600.0).astype(xp.float32)
            self._update_ground_water(land_E, column_dq * MCOL)
            rh = xp.clip(q / qsat(xp, Ta), float(mt.relative_humidity_min),
                         float(bounds.relative_humidity_max))
            cl = self._diagnostic_cloud(rh, div)
            self.cloud = self._update_cloud_cover(cl, rh, div)

        # Layer pressure-gradient flow plus terrain drag/blocking/deflection.
        h_adv = o.adv_diff_step(
            h, u, v,
            p.visc * float(pressure.thickness_diffusion_factor), dt)
        u_adv = o.adv_diff_step(u, u, v, p.visc, dt)
        v_adv = o.adv_diff_step(v, u, v, p.visc, dt)
        h_out = []
        for k in range(self.nz):
            h_eq = p.H0 - p.beta_T * (Ta[k] - Ta[k].mean())
            hk = h_adv[k]
            hk += dt * (-hk * div[k] + (h_eq - hk) / p.tau_h)
            hk = xp.clip(hk, float(pressure.thickness_min_factor) * p.H0,
                         float(pressure.thickness_max_factor) * p.H0)
            h_out.append(hk)
        h = xp.stack(h_out, axis=0)
        dhdx, dhdy = o.gradient(h)

        u_out, v_out = [], []
        for k in range(self.nz):
            uk = u_adv[k]
            vk = v_adv[k]
            terrain_u, terrain_v = self._terrain_acceleration(uk, vk, k)
            uk += dt * (-p.g_eff * dhdx[k]
                        + terrain_u + o.tanl * uk * vk)
            vk += dt * (-p.g_eff * dhdy[k]
                        + terrain_v - o.tanl * uk * uk)
            uk, vk = o.coriolis_rotate(uk, vk, dt)
            u_out.append(xp.clip(uk, -p.umax, p.umax))
            v_out.append(xp.clip(vk, -p.umax, p.umax))
        u = xp.stack(u_out, axis=0)
        v = xp.stack(v_out, axis=0)

        self.w_layers = self._update_vertical_velocity(div, u, v, Ta, dt)
        vp = p.vertical
        Ta = self._vertical_transport(Ta, self.w_layers, vp.diff_heat, dt)
        q = self._vertical_transport(q, self.w_layers, vp.diff_moisture, dt)
        u = self._vertical_transport(u, self.w_layers, vp.diff_momentum, dt)
        v = self._vertical_transport(v, self.w_layers, vp.diff_momentum, dt)
        h = self._vertical_transport(h, self.w_layers, vp.diff_momentum, dt)

        Ts = self._advance_ocean(Ts, u[0], v[0])

        u = o.polar_filter(u)
        v = o.polar_filter(v)
        h = o.polar_filter(h)
        Ta = o.polar_filter(Ta)
        q = o.polar_filter(q)
        self.u_layers = xp.clip(u, -p.umax, p.umax).astype(xp.float32)
        self.v_layers = xp.clip(v, -p.umax, p.umax).astype(xp.float32)
        self.h_layers = xp.clip(
            h, float(pressure.thickness_min_factor) * p.H0,
            float(pressure.thickness_max_factor) * p.H0).astype(xp.float32)
        self.T_layers = xp.clip(Ta, float(bounds.air_temp_min_k),
                                float(bounds.air_temp_max_k)).astype(xp.float32)
        self.q_layers = xp.clip(q, float(bounds.humidity_min),
                                float(bounds.humidity_max)).astype(xp.float32)
        self.Ts = xp.clip(Ts, float(bounds.surface_temp_min_k),
                          float(bounds.surface_temp_max_k)).astype(xp.float32)
        self._sync_surface_views()
        self._diag_surface()
        self.t += _dt.timedelta(seconds=self.dt)
        self.step_count += 1

    def apply_temp_edit(self, lat_deg, lon_deg, radius_km=800.0,
                        delta=5.0, target="both"):
        """在 (lat, lon) 为中心、radius_km 为尺度的高斯区域内
        增减温度 delta (K)。target: surface / air / both。"""
        xp = self.xp
        bounds, edit = self.cfg.physics.bounds, self.cfg.physics.edit
        la0 = _np.radians(float(lat_deg))
        lo0 = _np.radians(float(lon_deg) % 360.0)
        la = self.ops.lat                                  # [nlat,1] 弧度
        lo = xp.asarray(_np.radians(self.lons),
                        dtype=xp.float32)[None, :]         # [1,nlon]
        cosd = (xp.sin(la) * _np.sin(la0) +
                xp.cos(la) * _np.cos(la0) * xp.cos(lo - lo0))
        d_km = xp.arccos(xp.clip(cosd, -1.0, 1.0)) * (A_EARTH / 1000.0)
        w = xp.exp(-(d_km / max(float(radius_km),
                                float(edit.min_radius_km))) ** 2)
        dT = (float(delta) * w).astype(xp.float32)
        if target in ("surface", "both"):
            self.Ts = xp.clip(
                self.Ts + dT, float(bounds.surface_temp_min_k),
                float(bounds.surface_temp_max_k)).astype(xp.float32)
            # 变暖立即消融冰雪 / 骤冷时下一步会重新诊断
            if float(delta) > 0:
                melt = xp.clip((self.Ts - float(bounds.sea_ice_threshold_k))
                               / float(bounds.sea_ice_transition_k),
                               float(bounds.cloud_min), float(bounds.cloud_max))
                self.ice = xp.minimum(self.ice, 1 - melt * self.ocean)
                melt_l = xp.clip((self.Ts - float(bounds.snow_threshold_k))
                                 / float(bounds.snow_transition_k),
                                 float(bounds.cloud_min),
                                 float(bounds.cloud_max))
                self.snow = xp.minimum(self.snow, 1 - melt_l * self.land)
        if target in ("air", "both"):
            old_T = self.T_layers[0].copy()
            self.T_layers[0] = xp.clip(
                old_T + dT, float(bounds.air_temp_min_k),
                float(bounds.air_temp_max_k)).astype(xp.float32)
            # 保持相对湿度不突变: 随饱和比湿同步缩放水汽
            self.q_layers[0] = xp.clip(
                self.q_layers[0] * qsat(xp, self.T_layers[0]) /
                qsat(xp, old_T), float(bounds.humidity_min),
                float(edit.air_humidity_max)).astype(xp.float32)
            self._sync_surface_views()
            if self._primitive_enabled:
                self._refresh_primitive_diagnostics()

    def _regional_weight(self, lat_deg, lon_deg, radius_km):
        xp = self.xp
        min_radius = float(getattr(self.cfg.physics.edit, "min_radius_km", 50.0))
        la0 = _np.radians(float(lat_deg))
        lo0 = _np.radians(float(lon_deg) % 360.0)
        lo = xp.asarray(_np.radians(self.lons), dtype=xp.float32)[None, :]
        cosd = (xp.sin(self.ops.lat) * _np.sin(la0)
                + xp.cos(self.ops.lat) * _np.cos(la0) * xp.cos(lo - lo0))
        d_km = xp.arccos(xp.clip(cosd, -1.0, 1.0)) * (A_EARTH / 1000.0)
        radius = max(float(radius_km), min_radius)
        return d_km, xp.exp(-(d_km / radius) ** 2).astype(xp.float32)

    def _selected_wind_layers(self, layer=None):
        if layer is None or str(layer).lower() == "all":
            return range(self.nz)
        return [int(_np.clip(int(layer), 0, self.nz - 1))]

    def apply_wind_zero_edit(self, lat_deg, lon_deg, radius_km=800.0,
                             layer=None):
        """Damp horizontal wind to zero inside a Gaussian edit region."""
        xp = self.xp
        _, w = self._regional_weight(lat_deg, lon_deg, radius_km)
        keep = (1.0 - w).astype(xp.float32)
        for k in self._selected_wind_layers(layer):
            self.u_layers[k] = (self.u_layers[k] * keep).astype(xp.float32)
            self.v_layers[k] = (self.v_layers[k] * keep).astype(xp.float32)
        self._sync_surface_views()

    def apply_cyclone_edit(self, lat_deg, lon_deg, radius_km=900.0,
                           strength_ms=35.0, layer=None):
        """Add a compact cyclonic vortex to the selected wind layer(s)."""
        xp, p = self.xp, self.cfg.physics
        d_km, _ = self._regional_weight(lat_deg, lon_deg, radius_km)
        radius = max(float(radius_km), float(getattr(p.edit, "min_radius_km", 50.0)))
        r = xp.maximum(d_km / radius, 1.0e-4)
        speed = float(strength_ms) * r * xp.exp(0.5 * (1.0 - r * r))

        la0 = _np.radians(float(lat_deg))
        lo0 = _np.radians(float(lon_deg) % 360.0)
        la = self.ops.lat
        lo = xp.asarray(_np.radians(self.lons), dtype=xp.float32)[None, :]
        dlon = lo - lo0
        sin_c = xp.maximum(xp.sin(d_km / (A_EARTH / 1000.0)), 1.0e-4)
        radial_east = _np.cos(la0) * xp.sin(dlon) / sin_c
        radial_north = (_np.cos(la0) * xp.sin(la)
                        - _np.sin(la0) * xp.cos(la) * xp.cos(dlon)) / sin_c
        spin = 1.0 if float(lat_deg) >= 0.0 else -1.0
        du = (-spin * radial_north * speed).astype(xp.float32)
        dv = (spin * radial_east * speed).astype(xp.float32)

        max_wind = float(getattr(p.edit, "cyclone_max_wind_ms", p.umax))
        max_wind = min(max_wind, float(p.umax))
        for k in self._selected_wind_layers(layer):
            self.u_layers[k] = xp.clip(
                self.u_layers[k] + du, -max_wind, max_wind).astype(xp.float32)
            self.v_layers[k] = xp.clip(
                self.v_layers[k] + dv, -max_wind, max_wind).astype(xp.float32)
        self._sync_surface_views()

    # ------------------------------------------------------------
    def pressure_hpa(self):
        """Return hydrostatically reduced mean sea-level pressure in hPa."""
        if self._primitive_enabled:
            # Hypsometric reduction using the lowest-layer virtual
            # temperature. Unlike the old h mapping, ps itself is prognostic.
            exponent = self.xp.clip(
                self.surface_geopotential / self.xp.maximum(
                    RD * self.virtual_temperature_layers[0], 1.0),
                -2.0, 2.0)
            return self.surface_pressure * self.xp.exp(exponent) / 100.0
        pressure = self.cfg.physics.pressure
        return (float(pressure.mslp_reference_hpa)
                + (self.h - self.cfg.physics.H0)
                * float(pressure.hpa_per_thickness_m))

    def fields_cpu(self, include_layers=False):
        """导出全部展示字段到 numpy。"""
        fields = {
            "press": to_cpu(self.pressure_hpa()),
            "temp": to_cpu(self.Ta) - 273.15,
            "sst": to_cpu(self.Ts) - 273.15,
            "hum": to_cpu(self.q) * 1000.0,          # g/kg
            "cloud": to_cpu(self.cloud),
            "precip": to_cpu(self.precip),
            "ground_water": to_cpu(self.ground_water),
            "runoff": to_cpu(self.runoff),
            "river_flow": to_cpu(self.river_flow),
            "ice": to_cpu(self.xp.maximum(self.ice, self.snow)),
            "u": to_cpu(self.u), "v": to_cpu(self.v),
            "w": to_cpu(self.w),
            "uo": to_cpu(self.uo), "vo": to_cpu(self.vo),
        }
        if include_layers:
            fields["u_layers"] = to_cpu(self.u_layers)
            fields["v_layers"] = to_cpu(self.v_layers)
            if self._primitive_enabled:
                fields["pressure_layers_pa"] = to_cpu(
                    self.pressure_layers_pa)
                fields["surface_pressure_pa"] = to_cpu(
                    self.surface_pressure)
            if self.ocean_layers_enabled:
                fields["sst_deep"] = to_cpu(self.To_deep) - 273.15
                fields["uo_deep"] = to_cpu(self.uo_deep)
                fields["vo_deep"] = to_cpu(self.vo_deep)
        return fields

    def atmosphere_column_cpu(self, lat_index=None, lon_index=None):
        """Export all vertical levels, or one grid-column, for diagnostics."""
        selector = ((slice(None), slice(None), slice(None)) if lat_index is None
                    else (slice(None), int(lat_index), int(lon_index)))
        result = {
            "levels_m": self.levels_m.copy(),
            "temp_k": to_cpu(self.T_layers[selector]),
            "humidity": to_cpu(self.q_layers[selector]),
            "u": to_cpu(self.u_layers[selector]),
            "v": to_cpu(self.v_layers[selector]),
            "w": to_cpu(self.w_layers[selector]),
            "height": to_cpu(self.h_layers[selector]),
        }
        if self._primitive_enabled:
            result["pressure_pa"] = to_cpu(
                self.pressure_layers_pa[selector])
            result["omega_pa_s"] = to_cpu(self.omega_layers[selector])
            result["theta_k"] = to_cpu(
                (self.T_layers / self.exner_layers)[selector])
            result["sigma"] = self.sigma_centres.copy()
            result["reference_pressure_pa"] = (
                self.reference_pressure_layers_pa.copy())
        return result

    def potential_temperature(self):
        """Dry potential temperature on model layers (primitive core)."""
        if not self._primitive_enabled:
            raise RuntimeError(
                "potential temperature requires the primitive-equation core")
        return self.T_layers / self.exner_layers

    def energy_diagnostics(self):
        """Global energy, mass, water and stability diagnostics.

        Energies are area-weighted column integrals in J m-2; the error and
        change terms are the last time step's global tendencies in W m-2.
        ``limiter_counts`` reports how many cells the safety clips touched
        in the last step; a healthy run keeps every count at zero.
        """
        if not self._primitive_enabled:
            return {"available": False}
        xp, o, g = self.xp, self.ops, float(self.cfg.physics.g_eff)
        layer_mass = self.layer_pressure_thickness_pa / g
        out = {key: float(value) for key, value in self._budget.items()}
        out["enthalpy_j_m2"] = float(self._global_mean(
            (layer_mass * CP * self.T_layers).sum(axis=0)))
        out["mean_surface_pressure_pa"] = float(
            self._global_mean(self.surface_pressure))
        out["total_water_kg_m2"] = float(self._global_mean(
            (layer_mass * self.q_layers).sum(axis=0)))
        out["advective_cfl"] = float(
            (xp.abs(self.u_layers) * o.invdx
             + xp.abs(self.v_layers) * o.invdy).max()) * self.dt
        gravity_wave_speed = float(xp.sqrt(
            RD * self._global_mean(self.virtual_temperature_layers[0])))
        out["gravity_wave_cfl"] = (
            gravity_wave_speed * self.dt * float(o.invdx.max()))
        out["limiter_counts"] = {
            key: int(value) for key, value in self._limiter_counts.items()}
        for key, value in self._river_budget.items():
            out[key] = float(value)
        out["available"] = True
        return out

    def check_health(self):
        import numpy as np
        f = self.fields_cpu()
        for k, a in f.items():
            if not np.isfinite(a).all():
                raise FloatingPointError(f"字段 {k} 出现 NaN/Inf")
        return f
