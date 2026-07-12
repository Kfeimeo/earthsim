"""真实地球地形 (ETOPO 20 弧分, 来自 matplotlib/basemap 官方数据)。

- load_topo(): 读取 npz 并重采样到模拟网格, 生成陆海掩膜
- make_base_texture(): 生成用于 UI 的地形底图 PNG (含简单晕渲)
若数据缺失, scripts/get_topo.py 可重新下载; 也提供程序化退化地形。
"""
import os
import numpy as np


def sim_grid(nlat, nlon):
    """格点中心经纬度。行 0 = 南, 经度 0..360。"""
    dlat = 180.0 / nlat
    dlon = 360.0 / nlon
    lats = -90.0 + dlat * (np.arange(nlat) + 0.5)
    lons = dlon * np.arange(nlon)
    return lats, lons


def load_topo(path, nlat, nlon):
    """返回 (elev[nlat,nlon] 米, land_mask[nlat,nlon] 0/1)。"""
    if path and os.path.exists(path):
        z = np.load(path)
        topo, tlats, tlons = z["topo"], z["lats"], z["lons"]
        tlons = np.mod(tlons, 360.0)
        order = np.argsort(tlons)
        # 去掉重复经度列
        tlons, uniq = np.unique(tlons[order], return_index=True)
        topo = topo[:, order][:, uniq]
        if tlats[0] > tlats[-1]:
            tlats, topo = tlats[::-1], topo[::-1]
        lats, lons = sim_grid(nlat, nlon)
        li = np.clip(np.searchsorted(tlats, lats), 0, len(tlats) - 1)
        lj = np.clip(np.searchsorted(tlons, lons), 0, len(tlons) - 1)
        # 面积平均降采样(块平均), 网格粗于源数据时更平滑
        fy, fx = topo.shape[0] // nlat, topo.shape[1] // nlon
        if fy >= 2 and fx >= 2:
            ty, tx = nlat * fy, nlon * fx
            t = topo[:ty, :tx].reshape(nlat, fy, nlon, fx).mean(axis=(1, 3))
            elev = t
        else:
            elev = topo[np.ix_(li, lj)]
        return elev.astype(np.float32), (elev > 0).astype(np.float32)
    # ---- 退化: 程序化近似大陆(仅在没有数据文件时) ----
    lats, lons = sim_grid(nlat, nlon)
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    elev = -3000 + 3500 * (np.sin(np.radians(LO) * 1.5) *
                           np.cos(np.radians(LA) * 2) > 0.3)
    return elev.astype(np.float32), (elev > 0).astype(np.float32)


def make_base_texture(topo_path, out_png, width=1080):
    """由原始 ETOPO 生成地形底图 PNG (北在上)。"""
    from PIL import Image
    z = np.load(topo_path)
    topo, tlats = z["topo"], z["lats"]
    tlons = np.mod(z["lons"], 360.0)
    order = np.argsort(tlons)
    _, uniq = np.unique(tlons[order], return_index=True)
    topo = topo[:, order][:, uniq]
    if tlats[0] > tlats[-1]:
        topo = topo[::-1]
    h, w = topo.shape
    img = np.zeros((h, w, 3), np.float32)

    ocean = topo <= 0
    d = np.clip(-topo / 6000.0, 0, 1) ** 0.5
    img[ocean] = (np.stack([0.09 - 0.05 * d, 0.22 - 0.12 * d,
                            0.38 - 0.18 * d], -1)[ocean])

    e = np.clip(topo / 4500.0, 0, 1)
    lo = np.stack([0.32 + 0.1 * e, 0.42 - 0.05 * e, 0.24 - 0.02 * e], -1)
    hi = np.stack([0.45 + 0.4 * e, 0.38 + 0.4 * e, 0.30 + 0.45 * e], -1)
    t = np.clip((e - 0.25) / 0.5, 0, 1)[..., None]
    land_col = lo * (1 - t) + hi * t
    img[~ocean] = land_col[~ocean]

    # 简单东西向晕渲
    shade = np.clip((np.roll(topo, 1, 1) - topo) / 800.0, -0.5, 0.5)
    img *= (1.0 + 0.35 * shade[..., None] * (~ocean)[..., None])

    img = (np.clip(img, 0, 1)[::-1] * 255).astype(np.uint8)  # 北在上
    im = Image.fromarray(img)
    if width and width != w:
        im = im.resize((width, int(width * h / w)), Image.LANCZOS)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    im.save(out_png)
    return out_png
