"""Download and convert global topography to EarthSim npz format.

Default keeps the original lightweight ETOPO 20 arc-minute source.  For a
finer source use NOAA ETOPO 2022:

    python scripts/get_topo.py --resolution 60s
    python scripts/get_topo.py --resolution 30s

The 60s file is roughly 491 MB before compression; 30s is roughly 1.7 GB and
needs substantially more temporary disk and memory during conversion.
"""
from __future__ import annotations

import argparse
import gzip
import io
import pathlib
import tempfile
import urllib.request

import numpy as np


BASMAP20_BASE = (
    "https://raw.githubusercontent.com/matplotlib/basemap/master/doc/examples/"
)

ETOPO2022 = {
    "60s": {
        "url": (
            "https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/"
            "60s/60s_bed_elev_netcdf/ETOPO_2022_v1_60s_N90W180_bed.nc"
        ),
        "out": "etopo2022_60s.npz",
    },
    "30s": {
        "url": (
            "https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/"
            "30s/30s_bed_elev_netcdf/ETOPO_2022_v1_30s_N90W180_bed.nc"
        ),
        "out": "etopo2022_30s.npz",
    },
}


def data_dir():
    return pathlib.Path(__file__).resolve().parent.parent / "data"


def fetch_bytes(url, timeout=120):
    print("download", url)
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def fetch_file(url, out_path, timeout=120):
    print("download", url)
    with urllib.request.urlopen(url, timeout=timeout) as r:
        with open(out_path, "wb") as f:
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    return out_path


def save_basemap20(out):
    def fetch_gz(name):
        return gzip.decompress(fetch_bytes(BASMAP20_BASE + name))

    topo = np.loadtxt(io.BytesIO(fetch_gz("etopo20data.gz")))
    lats = np.loadtxt(io.BytesIO(fetch_gz("etopo20lats.gz")))
    lons = np.loadtxt(io.BytesIO(fetch_gz("etopo20lons.gz")))
    np.savez_compressed(out, topo=topo, lats=lats, lons=lons)
    print("saved", out, topo.shape)


def _coord(ds, names):
    for name in names:
        if name in ds.coords:
            return np.asarray(ds.coords[name].values)
        if name in ds.variables and ds[name].ndim == 1:
            return np.asarray(ds[name].values)
    raise KeyError(f"missing coordinate: one of {names}")


def _topo_var(ds):
    for name in ("z", "elevation", "Band1", "bedrock_topography"):
        if name in ds.variables and ds[name].ndim >= 2:
            return ds[name]
    for da in ds.data_vars.values():
        if da.ndim >= 2:
            return da
    raise KeyError("no 2-D topography variable found")


def save_etopo2022(resolution, out, keep_netcdf=False):
    try:
        import xarray as xr
    except ImportError as exc:
        raise SystemExit("xarray is required: pip install xarray netCDF4") from exc

    meta = ETOPO2022[resolution]
    out.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        nc_path = out.with_suffix(".nc") if keep_netcdf else pathlib.Path(td) / "topo.nc"
        fetch_file(meta["url"], nc_path, timeout=300)
        with xr.open_dataset(nc_path) as ds:
            da = _topo_var(ds).squeeze()
            topo = np.asarray(da.values, dtype=np.float32)
            lats = _coord(ds, ("lat", "latitude", "y")).astype(np.float64)
            lons = _coord(ds, ("lon", "longitude", "x")).astype(np.float64)
        np.savez_compressed(out, topo=topo, lats=lats, lons=lons)
        print("saved", out, topo.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resolution", default="20m", choices=["20m", "60s", "30s"],
        help="20m is small; 60s/30s use NOAA ETOPO 2022 NetCDF sources.",
    )
    parser.add_argument("--out", default="", help="output .npz path")
    parser.add_argument(
        "--keep-netcdf", action="store_true",
        help="keep the downloaded NOAA NetCDF next to the output file.",
    )
    args = parser.parse_args()

    root = data_dir()
    root.mkdir(exist_ok=True)
    if args.out:
        out = pathlib.Path(args.out)
    elif args.resolution == "20m":
        out = root / "etopo20.npz"
    else:
        out = root / ETOPO2022[args.resolution]["out"]

    if args.resolution == "20m":
        save_basemap20(out)
    else:
        save_etopo2022(args.resolution, out, keep_netcdf=args.keep_netcdf)


if __name__ == "__main__":
    main()
