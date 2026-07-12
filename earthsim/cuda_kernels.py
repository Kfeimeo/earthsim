"""加载并包装 kernels.cu 中的手写 CUDA kernel。仅在 cuda 后端下使用。"""
import os

_module = None
_adv_diff = None


def load():
    global _module, _adv_diff
    if _module is not None:
        return True
    try:
        import cupy as cp
        src = open(os.path.join(os.path.dirname(__file__), "kernels.cu"),
                   encoding="utf-8").read()
        _module = cp.RawModule(code=src, options=("--use_fast_math",))
        _adv_diff = _module.get_function("adv_diff")
        return True
    except Exception:
        _module = None
        return False


def adv_diff(F, u, v, invdx, invdy, K, dt):
    """返回 F + dt*(-adv + K*lap)。F/u/v 为 float32 cupy 数组 (nlat,nlon)。"""
    import cupy as cp
    nlat, nlon = F.shape
    out = cp.empty_like(F)
    block = (16, 16)
    grid = ((nlon + 15) // 16, (nlat + 15) // 16)
    _adv_diff(grid, block,
              (F, u, v, invdx, cp.float32(invdy), cp.float32(K),
               cp.float32(dt), out, cp.int32(nlat), cp.int32(nlon)))
    return out


def available():
    return _module is not None
