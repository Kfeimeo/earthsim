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
        "initial_conditions": {
            "equilibrium_temp_base_k": 308.0,
            "equilibrium_temp_pole_delta_k": 42.0,
            "surface_lapse_rate": 0.0065,
            "humidity_decay_height_m": 4500.0,
            "temp_wave_lon1": 3.0,
            "temp_wave_phase1": 0.7,
            "temp_wave_weight2": 0.5,
            "temp_wave_lon2": 5.0,
            "temp_wave_phase2": -1.3,
            "temp_wave_decay_height_m": 7000.0,
            "real_default_temp_k": 280.0,
            "real_default_humidity": 0.01,
            "real_default_mslp_hpa": 1013.0,
        },
        "bounds": {
            "air_temp_min_k": 160.0,
            "air_temp_max_k": 345.0,
            "surface_temp_min_k": 170.0,
            "surface_temp_max_k": 350.0,
            "ocean_initial_min_temp_k": 271.2,
            "ocean_freezing_min_temp_k": 268.0,
            "sea_ice_threshold_k": 271.4,
            "sea_ice_transition_k": 2.0,
            "snow_threshold_k": 273.5,
            "snow_transition_k": 6.0,
            "humidity_min": 0.0,
            "humidity_max": 0.05,
            "cloud_min": 0.0,
            "cloud_max": 1.0,
            "precip_max_mmh": 1.2,
            "ocean_current_max_ms": 2.0,
            "relative_humidity_max": 1.3,
        },
        "radiation_transfer": {
            "albedo_max": 0.85,
            "surface_sw_fraction": 0.80,
            "cloud_sw_absorption": 0.30,
            "air_sw_fraction": 0.18,
            "surface_emissivity": 0.98,
            "down_emissivity_base": 0.60,
            "down_emissivity_humidity": 0.25,
            "down_emissivity_cloud": 0.15,
            "down_emissivity_max": 0.98,
            "olr_factor_base": 0.62,
            "olr_factor_humidity": 0.20,
            "olr_factor_cloud": 0.08,
            "olr_factor_min": 0.28,
            "olr_factor_max": 0.62,
            "air_longwave_coupling": 0.85,
            "seasonal_temp_base_k": 302.0,
            "seasonal_temp_pole_delta_k": 42.0,
            "seasonal_temp_amp_k": 8.0,
        },
        "surface_flux": {
            "wind_speed_floor_ms": 1.0,
            "sensible_heat_coeff": 1.2e-3,
            "humidity_greenhouse_scale": 0.02,
        },
        "moisture_transport": {
            "subsidence_drying_coeff": 0.7,
            "subsidence_divergence_max": 1.0e-4,
            "precip_cloud_diagnostic_scale_mmh": 2.0,
            "cloud_subsidence_scale": 1.5e5,
            "cloud_subsidence_clear_max": 2.0,
            "cloud_subsidence_cover_max": 0.45,
            "relative_humidity_min": 0.0,
            "effective_rh_max": 1.2,
        },
        "pressure": {
            "mslp_reference_hpa": 1013.0,
            "thickness_per_hpa": 1.0 / 0.045,
            "hpa_per_thickness_m": 0.045,
            "thickness_min_factor": 0.4,
            "thickness_max_factor": 1.8,
            "thickness_diffusion_factor": 0.5,
        },
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
        "ideal_wind_shape": {
            "trade_lat_width_deg": 23.0,
            "trade_decay_height_m": 3500.0,
            "surface_westerly_lat_deg": 48.0,
            "surface_westerly_width_deg": 18.0,
            "surface_westerly_decay_height_m": 4500.0,
            "polar_easterly_lat_deg": 72.0,
            "polar_easterly_width_deg": 11.0,
            "polar_easterly_decay_height_m": 5500.0,
            "wave_decay_height_m": 9000.0,
            "wave_u_fraction": 0.6,
            "wave_wavenumber": 4.0,
            "wave_phase": 0.8,
        },
        "land_evap": 0.25,
        "ground_water_capacity_mm": 150.0,
        "initial_ground_water_mm": 60.0,
        "ground_evap_exponent": 1.0,
        "ground_runoff_tau": 864000.0,
        "ground_runoff_exponent": 2.0,
        "mld": 40.0, "tau_ocean": 8.0e-7, "drag_ocean": 1.5e-6,
        "visc_ocean": 4.0e4,
        "ocean_transport": {
            "wind_speed_floor_ms": 1.0,
            "ekman_coriolis_factor": 0.15,
            "surface_heat_diffusivity": 2.0e3,
        },
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
            "scale_height_min_m": 1000.0,
            "continuity_relax": 900.0,
            "buoyancy_factor": 5.0e-4,
            "w_damping": 5.0e-4,
            "w_max": 8.0,
            "buoyancy_temp_min_k": 180.0,
            "lowest_layer_mass_min": 0.02,
            "diff_momentum": 12.0,
            "diff_heat": 8.0,
            "diff_moisture": 5.0,
        },
        "topography": {
            "enabled": True,
            "smooth_passes": 2,
            "smooth_center_weight": 0.5,
            "smooth_neighbor_weight": 0.25,
            "influence_height": 2500.0,
            "elevation_scale": 1500.0,
            "slope_scale": 0.02,
            "drag_multiplier": 8.0,
            "blocking_rate": 8.0e-4,
            "deflection_rate": 3.0e-4,
            "lift_efficiency": 0.7,
            "lift_max": 5.0,
            "slope_min": 1.0e-6,
            "turn_speed_threshold": 0.1,
        },
        "edit": {
            "min_radius_km": 50.0,
            "air_humidity_max": 0.04,
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
