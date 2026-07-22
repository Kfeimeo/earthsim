"""Global topography helpers.

The model stores terrain as compressed ``npz`` files with three arrays:
``topo`` in meters, ``lats`` in degrees, and ``lons`` in degrees.  When a
configuration uses ``topo_file: auto`` or ``topo_files: [...]``, the highest
resolution available file is selected at startup.
"""
from __future__ import annotations

import glob
import os
import time
from pathlib import Path

import numpy as np


def sim_grid(nlat, nlon):
    """Cell-centre latitude/longitude. Row 0 is south, longitude is 0..360."""
    dlat = 180.0 / nlat
    dlon = 360.0 / nlon
    lats = -90.0 + dlat * (np.arange(nlat) + 0.5)
    lons = dlon * np.arange(nlon)
    return lats, lons


def _repo_data_dir():
    return Path(__file__).resolve().parent.parent / "data"


def _flatten_spec(spec):
    if spec is None:
        return []
    if isinstance(spec, (list, tuple)):
        out = []
        for item in spec:
            out.extend(_flatten_spec(item))
        return out
    text = str(spec).strip()
    return [text] if text else []


def _auto_candidates():
    dirs = [Path.cwd() / "data", _repo_data_dir()]
    seen_dirs = set()
    out = []
    for data_dir in dirs:
        key = str(data_dir.resolve()) if data_dir.exists() else str(data_dir)
        if key in seen_dirs:
            continue
        seen_dirs.add(key)
        for pat in ("etopo*.npz", "topo*.npz"):
            out.extend(Path(p) for p in glob.glob(str(data_dir / pat)))
    return out


def _candidate_paths(spec):
    out = []
    for item in _flatten_spec(spec):
        if item.lower() == "auto":
            out.extend(_auto_candidates())
        elif any(ch in item for ch in "*?[]"):
            out.extend(Path(p) for p in glob.glob(item))
        else:
            out.append(Path(item))

    deduped = []
    seen = set()
    for path in out:
        key = str(path.resolve()) if path.exists() else str(path)
        if key not in seen:
            seen.add(key)
            deduped.append(path)
    return deduped


def _topo_rank(path):
    try:
        with np.load(path) as z:
            shape = tuple(z["topo"].shape)
    except Exception:
        return (-1, str(path))
    cells = int(np.prod(shape))
    return (cells, str(path))


def resolve_topo_path(spec):
    """Return the best existing topography file for a string/list spec."""
    started = time.perf_counter()
    existing = [p for p in _candidate_paths(spec) if p.exists()]
    if not existing:
        print(f"[startup] topography candidates resolved (none found): "
              f"{time.perf_counter() - started:.3f}s", flush=True)
        return None
    selected = str(max(existing, key=_topo_rank))
    print(f"[startup] topography candidate selected "
          f"({len(existing)} found, {selected}): "
          f"{time.perf_counter() - started:.3f}s", flush=True)
    return selected


def _read_topo(path):
    started = time.perf_counter()
    with np.load(path) as z:
        result = (np.asarray(z["topo"], dtype=np.float32),
                  np.asarray(z["lats"], dtype=np.float64),
                  np.asarray(z["lons"], dtype=np.float64))
    print(f"[startup] topography archive read "
          f"({result[0].shape[0]}x{result[0].shape[1]}): "
          f"{time.perf_counter() - started:.3f}s", flush=True)
    return result


def _resample_topo(topo, tlats, tlons, nlat, nlon):
    started = time.perf_counter()
    source_shape = topo.shape
    tlons = np.mod(np.asarray(tlons, dtype=np.float64), 360.0)
    order = np.argsort(tlons)
    tlons, uniq = np.unique(tlons[order], return_index=True)
    topo = np.asarray(topo, dtype=np.float32)[:, order][:, uniq]
    if tlats[0] > tlats[-1]:
        tlats, topo = tlats[::-1], topo[::-1]

    lats, lons = sim_grid(nlat, nlon)
    lon_ext = np.concatenate(([tlons[-1] - 360.0], tlons,
                              [tlons[0] + 360.0]))
    topo_ext = np.concatenate((topo[:, -1:], topo, topo[:, :1]), axis=1)
    qlon = np.asarray(lons, dtype=np.float64)
    qlon = np.where(qlon < lon_ext[0], qlon + 360.0, qlon)
    qlon = np.where(qlon > lon_ext[-1], qlon - 360.0, qlon)
    along = np.stack([np.interp(qlon, lon_ext, row)
                      for row in topo_ext], axis=0)
    elev = np.stack([np.interp(lats, tlats, along[:, j])
                     for j in range(nlon)], axis=1)
    result = elev.astype(np.float32)
    print(f"[startup] topography resampled "
          f"({source_shape[0]}x{source_shape[1]} -> {nlat}x{nlon}): "
          f"{time.perf_counter() - started:.3f}s", flush=True)
    return result


def _procedural_topo(nlat, nlon):
    lats, lons = sim_grid(nlat, nlon)
    la, lo = np.meshgrid(lats, lons, indexing="ij")
    elev = -3000 + 3500 * (np.sin(np.radians(lo) * 1.5) *
                           np.cos(np.radians(la) * 2) > 0.3)
    return elev.astype(np.float32)


def load_topo(path, nlat, nlon):
    """Return ``(elev[nlat,nlon], land_mask[nlat,nlon])``."""
    started = time.perf_counter()
    resolved = resolve_topo_path(path)
    if resolved:
        topo, tlats, tlons = _read_topo(resolved)
        elev = _resample_topo(topo, tlats, tlons, nlat, nlon)
        result = elev, (elev > 0).astype(np.float32)
        print(f"[startup] topography load complete: "
              f"{time.perf_counter() - started:.3f}s", flush=True)
        return result

    elev = _procedural_topo(nlat, nlon)
    result = elev, (elev > 0).astype(np.float32)
    print(f"[startup] procedural topography generated: "
          f"{time.perf_counter() - started:.3f}s", flush=True)
    return result


def make_base_texture(topo_path, out_png, width=1080):
    """Create the UI base-map PNG from the best available terrain file."""
    started = time.perf_counter()
    from PIL import Image

    resolved = resolve_topo_path(topo_path)
    if resolved:
        topo, tlats, tlons = _read_topo(resolved)
        h = int(round(width / 2)) if width else topo.shape[0]
        w = int(width) if width else topo.shape[1]
        topo = _resample_topo(topo, tlats, tlons, h, w)
    else:
        w = int(width or 1080)
        h = max(1, int(round(w / 2)))
        topo = _procedural_topo(h, w)

    img = np.zeros((*topo.shape, 3), np.float32)
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

    shade = np.clip((np.roll(topo, 1, 1) - topo) / 800.0, -0.5, 0.5)
    img *= (1.0 + 0.35 * shade[..., None] * (~ocean)[..., None])

    img = (np.clip(img, 0, 1)[::-1] * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    Image.fromarray(img).save(out_png)
    print(f"[startup] basemap texture written ({w}x{h}): "
          f"{time.perf_counter() - started:.3f}s", flush=True)
    return out_png
