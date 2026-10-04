"""区域天气分析: 给定经纬度, 输出天气分类与要素。"""
import numpy as np


def classify(temp_c, cloud, precip_mmh, wind_ms):
    """返回 (中文天气, 图标 emoji)。"""
    if precip_mmh > 0.04:
        snow = temp_c < 0.5
        if precip_mmh > 0.30:
            w = ("大雪", "❄️") if snow else ("大雨", "⛈️")
        elif precip_mmh > 0.12:
            w = ("中雪", "🌨️") if snow else ("中雨", "🌧️")
        else:
            w = ("小雪", "🌨️") if snow else ("小雨", "🌦️")
    elif cloud > 0.75:
        w = ("阴", "☁️")
    elif cloud > 0.35:
        w = ("多云", "⛅")
    else:
        w = ("晴", "☀️")
    if wind_ms > 17:
        w = (w[0] + "·大风", w[1] + "💨")
    return w


def locate_cell(lats, lons, lat, lon):
    """Return the grid indices nearest to a latitude/longitude."""
    lon = lon % 360.0
    i = int(np.clip(np.abs(lats - lat).argmin(), 0, len(lats) - 1))
    j = int(np.argmin(np.minimum(np.abs(lons - lon),
                                 360 - np.abs(lons - lon))))
    return i, j


def _wind_direction_deg(u, v):
    """Direction the wind blows toward, degrees clockwise from north."""
    return (np.degrees(np.arctan2(u, v)) + 360.0) % 360.0


def column_profile(column, selected_layer=None):
    """Summarize one atmospheric column for the analysis card.

    ``column`` holds per-level 1-D arrays as returned by
    ``EarthModel.atmosphere_column_cpu(i, j)``; only ``u``/``v`` are
    required, so playback frames that store layer winds and pressures but
    no layer temperature still produce a (shorter) profile.  Levels are
    listed from the model top down to the surface, as in a sounding.
    """
    u = np.asarray(column["u"], dtype=np.float64).reshape(-1)
    v = np.asarray(column["v"], dtype=np.float64).reshape(-1)
    nz = u.size
    if nz == 0:
        return None

    def level_array(name):
        values = column.get(name)
        if values is None:
            return None
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        return values if values.size == nz else None

    pressure = level_array("pressure_pa")
    height = level_array("levels_m")
    temperature = level_array("temp_k")
    humidity = level_array("humidity")
    w = level_array("w")
    omega = level_array("omega_pa_s")
    speed = np.hypot(u, v)
    direction = _wind_direction_deg(u, v)
    relative_humidity = None
    if temperature is not None and humidity is not None:
        from .physics import qsat
        reference = pressure if pressure is not None else 1.0e5
        relative_humidity = humidity / np.maximum(
            qsat(np, temperature, reference), 1e-9)

    levels = []
    for k in range(nz - 1, -1, -1):
        entry = {"index": k, "wind_speed": round(float(speed[k]), 1),
                 "wind_dir": int(round(float(direction[k])))}
        if pressure is not None:
            entry["pressure"] = round(float(pressure[k]) / 100.0, 1)
        if height is not None:
            entry["height"] = int(round(float(height[k])))
        if temperature is not None:
            entry["temp"] = round(float(temperature[k]) - 273.15, 1)
        if humidity is not None:
            entry["humidity"] = round(float(humidity[k]) * 1000.0, 2)
        if relative_humidity is not None:
            entry["rh"] = int(round(100.0 * float(np.clip(
                relative_humidity[k], 0.0, 1.5))))
        if w is not None:
            entry["w"] = round(float(w[k]) * 100.0, 1)        # cm/s
        if omega is not None:
            entry["omega"] = round(float(omega[k]), 3)       # Pa/s
        levels.append(entry)

    out = {"levels": levels, "count": nz}
    if selected_layer is not None:
        k = int(np.clip(int(selected_layer), 0, nz - 1))
        out["selected_layer"] = k
        out["selected"] = next(e for e in levels if e["index"] == k)
    return out


def analyze_point(fields, lats, lons, land, lat, lon, column=None,
                  selected_layer=None):
    """fields: model.fields_cpu() 的结果 (numpy)。

    ``column`` optionally carries the per-level arrays of the same grid cell
    (see :func:`column_profile`), which adds a ``profile`` entry.
    """
    i, j = locate_cell(lats, lons, lat, lon)

    def g(k):
        return float(fields[k][i, j])

    wind = float(np.hypot(g("u"), g("v")))
    wdir = _wind_direction_deg(g("u"), g("v"))  # 吹向
    label, icon = classify(g("temp"), g("cloud"), g("precip"), wind)
    is_land = bool(land[i, j] > 0.5)
    out = {
        "lat": round(float(lats[i]), 2), "lon": round(float(lons[j]), 2),
        "surface": "陆地" if is_land else "海洋",
        "weather": label, "icon": icon,
        "temp": round(g("temp"), 1),
        "pressure": round(g("press"), 1),
        "humidity": round(g("hum"), 2),
        "cloud": round(g("cloud") * 100),
        "precip": round(g("precip"), 2),
        "wind_speed": round(wind, 1),
        "wind_dir": round(wdir),
        "ice": round(g("ice") * 100),
    }
    if is_land:
        if "ground_water" in fields:
            out["ground_water"] = round(g("ground_water"), 1)
    else:
        out["sst"] = round(g("sst"), 1)
        out["current"] = round(float(np.hypot(g("uo"), g("vo"))), 2)
    if column is not None:
        profile = column_profile(column, selected_layer)
        if profile is not None:
            out["profile"] = profile
    return out
