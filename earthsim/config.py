"""加载 YAML 配置, 提供带默认值的点式访问。"""
import copy
import yaml

DEFAULTS = {
    "backend": "auto",
    "grid": {
        "nlat": 90, "nlon": 180,
        "topo_file": "auto",
        "topo_files": [
            "data/etopo2022_30s.npz",
            "data/etopo2022_60s.npz",
            "data/etopo20.npz",
        ],
    },
    "time": {"dt": 240.0, "start": "2026-01-01T00:00:00", "spinup_days": 0},
    "data": {
        # auto uses real data when the configured file exists, otherwise the
        # original idealized initialization remains available for demos/tests.
        "init_mode": "auto",       # ideal | auto | real
        "atmosphere_file": "data/era5_pressure_levels.nc",
        "surface_file": "data/era5_single_levels.nc",
        "ocean_file": "data/cmems_surface_currents.nc",
        "init_time": "",           # empty: use time.start
    },
    "physics": {
        "moisture": True, "radiation": True, "ocean": True,
        "ice_albedo": True, "diurnal_cycle": True,
        "H0": 800.0, "g_eff": 9.8, "beta_T": 22.0, "tau_h": 43200.0,
        "drag": 6.0e-6,
        "drag_ocean_atmosphere": 4.0e-6,
        "drag_land_atmosphere": 1.2e-5,
        "visc": 8.0e4, "diff_T": 4.0e4, "diff_q": 2.0e4,
        "umax": 80.0,
        "S0": 1361.0, "alb_ocean": 0.07, "alb_land": 0.22,
        "alb_ice": 0.62, "alb_snow": 0.70, "alb_cloud": 0.42,
        "tau_relax_T": 864000.0,
        "rh_crit": 0.90, "tau_cond": 5400.0, "c_evap": 9.0e-4,
        "convective_rh_drop_max": 0.12, "convective_div_scale": 3.0e-5,
        "cloud_rh_start": 0.70, "cloud_rh_full": 0.95,
        "cloud_rh_exponent": 1.6,
        "tau_cloud_form": 900.0, "tau_cloud_dissip": 5400.0,
        "cloud_clear_rh": 0.65, "cloud_dry_clear": 3.0,
        "cloud_subsidence_clear": 2.0, "cloud_precip_clear": 0.6,
        "cloud_precip_scale": 1.0,
        "init_surface_rh": 0.58, "init_upper_rh": 0.28,
        "ideal_wave_amp_K": 1.2, "ideal_humidity_wave": 0.10,
        "ideal_wind_enabled": True, "ideal_trade_wind_ms": 5.0,
        "ideal_surface_westerly_ms": 4.0, "ideal_jet_ms": 22.0,
        "ideal_polar_easterly_ms": 2.5, "ideal_jet_lat_deg": 42.0,
        "ideal_jet_width_deg": 12.0, "ideal_jet_height_m": 9000.0,
        "ideal_jet_depth_m": 4500.0, "ideal_wind_wave_ms": 1.2,
        "land_evap": 0.25,
        "ground_water_capacity_mm": 150.0,
        "initial_ground_water_mm": 60.0,
        "ground_evap_exponent": 1.0,
        "ground_runoff_tau": 864000.0,
        "ground_runoff_exponent": 2.0,
        "mld": 40.0, "tau_ocean": 8.0e-7, "drag_ocean": 1.5e-6,
        "visc_ocean": 4.0e4,
        "ocean_layers": {
            "enabled": False,
            "lower_depth_m": 200.0,
            "deep_initial_offset_k": 1.5,
            "interlayer_drag": 2.0e-7,
            "interlayer_heat_exchange": 1.0e-7,
            "deep_drag": 3.0e-7,
            "deep_visc": 2.0e4,
            "deep_temp_min_k": 268.0,
        },
        "vertical": {
            "enabled": True,
            "levels_m": [100.0, 1000.0, 3000.0, 6000.0, 10000.0],
            "lapse_rate": 0.0065,
            "scale_height": 8000.0,
            "continuity_relax": 900.0,
            "buoyancy_factor": 5.0e-4,
            "w_damping": 5.0e-4,
            "w_max": 8.0,
            "diff_momentum": 12.0,
            "diff_heat": 8.0,
            "diff_moisture": 5.0,
        },
        "topography": {
            "enabled": True,
            "smooth_passes": 2,
            "influence_height": 2500.0,
            "elevation_scale": 1500.0,
            "slope_scale": 0.02,
            "drag_multiplier": 8.0,
            "blocking_rate": 8.0e-4,
            "deflection_rate": 3.0e-4,
            "lift_efficiency": 0.7,
            "lift_max": 5.0,
        },
    },
    "numerics": {"advection": "upwind", "cos_clamp": 0.2,
                 "polar_filter_lat": 65.0, "polar_filter_passes": 6},
    "precompute": {"out_dir": "output/run1", "days": 3,
                   "save_every_steps": 15},
    "server": {"host": "0.0.0.0", "port": 8000, "steps_per_frame": 3,
               "max_fps": 10, "vector_stride": 5, "basemap_width": 2160,
               "record_enabled": True, "record_dir": "output/recordings",
               "record_every_steps": 60},
}


class Cfg(dict):
    """dict + 属性访问。"""
    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) else v


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        elif v is None and k in out:
            continue
        else:
            # PyYAML 会把 "4.0e4" 解析为字符串, 按默认值类型强制转换
            d = out.get(k)
            if isinstance(v, str) and isinstance(d, (int, float)) \
                    and not isinstance(d, bool):
                try:
                    v = type(d)(float(v))
                except ValueError:
                    pass
            out[k] = v
    return out


def load_config(path=None):
    user = {}
    if path:
        with open(path, encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
    return Cfg(_merge(DEFAULTS, user))
