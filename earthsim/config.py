"""加载 YAML 配置, 提供带默认值的点式访问。"""
import copy
import yaml

DEFAULTS = {
    "backend": "auto",
    "grid": {"nlat": 90, "nlon": 180, "topo_file": "data/etopo20.npz"},
    "time": {"dt": 240.0, "start": "2026-01-01T00:00:00", "spinup_days": 0},
    "physics": {
        "moisture": True, "radiation": True, "ocean": True,
        "ice_albedo": True, "diurnal_cycle": True,
        "H0": 800.0, "g_eff": 9.8, "beta_T": 22.0, "tau_h": 43200.0,
        "drag": 6.0e-6, "visc": 8.0e4, "diff_T": 4.0e4, "diff_q": 4.0e4,
        "umax": 120.0,
        "S0": 1361.0, "alb_ocean": 0.07, "alb_land": 0.22,
        "alb_ice": 0.62, "alb_snow": 0.70, "alb_cloud": 0.42,
        "tau_relax_T": 864000.0,
        "rh_crit": 0.85, "tau_cond": 3600.0, "c_evap": 1.3e-3,
        "land_evap": 0.25,
        "mld": 40.0, "tau_ocean": 8.0e-7, "drag_ocean": 1.5e-6,
        "visc_ocean": 4.0e4,
    },
    "numerics": {"advection": "upwind", "cos_clamp": 0.2,
                 "polar_filter_lat": 65.0, "polar_filter_passes": 6},
    "precompute": {"out_dir": "output/run1", "days": 3,
                   "save_every_steps": 15},
    "server": {"host": "0.0.0.0", "port": 8000, "steps_per_frame": 3,
               "max_fps": 10, "vector_stride": 5},
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
