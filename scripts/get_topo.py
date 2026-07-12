"""下载真实 ETOPO 20' 地形数据并转为 data/etopo20.npz"""
import gzip, io, urllib.request, numpy as np, pathlib

BASE = "https://raw.githubusercontent.com/matplotlib/basemap/master/doc/examples/"

def fetch(name):
    print("下载", name)
    with urllib.request.urlopen(BASE + name, timeout=60) as r:
        return gzip.decompress(r.read())

def main():
    topo = np.loadtxt(io.BytesIO(fetch("etopo20data.gz")))
    lats = np.loadtxt(io.BytesIO(fetch("etopo20lats.gz")))
    lons = np.loadtxt(io.BytesIO(fetch("etopo20lons.gz")))
    out = pathlib.Path(__file__).resolve().parent.parent / "data" / "etopo20.npz"
    out.parent.mkdir(exist_ok=True)
    np.savez_compressed(out, topo=topo, lats=lats, lons=lons)
    print("保存", out, topo.shape)

if __name__ == "__main__":
    main()
