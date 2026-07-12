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


def analyze_point(fields, lats, lons, land, lat, lon):
    """fields: model.fields_cpu() 的结果 (numpy)。"""
    lon = lon % 360.0
    i = int(np.clip(np.abs(lats - lat).argmin(), 0, len(lats) - 1))
    j = int(np.argmin(np.minimum(np.abs(lons - lon),
                                 360 - np.abs(lons - lon))))

    def g(k):
        return float(fields[k][i, j])

    wind = float(np.hypot(g("u"), g("v")))
    wdir = (np.degrees(np.arctan2(g("u"), g("v"))) + 360) % 360  # 吹向
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
    if not is_land:
        out["sst"] = round(g("sst"), 1)
        out["current"] = round(float(np.hypot(g("uo"), g("vo"))), 2)
    return out
