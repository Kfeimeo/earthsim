"""加载并包装 kernels.cu 中的手写 CUDA kernel。仅在 cuda 后端下使用。"""
import os

_module = None
_adv_diff = None
_polar_filter = None
_ADV_BLOCK = (16, 16)  # Must match the static shared-memory tile in kernels.cu.
_POLAR_BLOCK = (256,)


def load():
    global _module, _adv_diff, _polar_filter
    if _module is not None:
        return True
    try:
        import cupy as cp
        with open(os.path.join(os.path.dirname(__file__), "kernels.cu"),
                  encoding="utf-8") as source_file:
            src = source_file.read()
        _module = cp.RawModule(code=src, options=("--use_fast_math",))
        _adv_diff = _module.get_function("adv_diff")
        _polar_filter = _module.get_function("polar_filter")
        return True
    except Exception:
        _module = None
        return False


def adv_diff(F, u, v, invdx, invdy, K, dt):
    """返回 F + dt*(-adv + K*lap)。F/u/v 为 float32 cupy 数组 (nlat,nlon)。"""
    import cupy as cp
    nlat, nlon = F.shape
    out = cp.empty_like(F)
    block = _ADV_BLOCK
    grid = ((nlon + block[0] - 1) // block[0],
            (nlat + block[1] - 1) // block[1])
    _adv_diff(grid, block,
              (F, u, v, invdx, cp.float32(invdy), cp.float32(K),
               cp.float32(dt), out, cp.int32(nlat), cp.int32(nlon)))
    return out


def polar_filter(F, weights, passes):
    """Apply all zonal filter passes to a 2-D or batched 3-D float32 field."""
    import cupy as cp
    if F.ndim < 2 or F.shape[-2] != weights.size:
        raise ValueError("polar filter field and latitude weights do not match")
    if F.dtype != cp.float32:
        raise TypeError("polar filter CUDA kernel requires float32 input")

    F = cp.ascontiguousarray(F)
    nlat, nlon = F.shape[-2:]
    rings = F.size // nlon
    shared_mem = 2 * nlon * cp.dtype(cp.float32).itemsize
    max_shared = cp.cuda.Device().attributes["MaxSharedMemoryPerBlock"]
    if shared_mem > max_shared:
        raise ValueError(
            f"polar filter requires {shared_mem} bytes of shared memory, "
            f"but this GPU supports {max_shared} bytes per block"
        )

    out = cp.empty_like(F)
    _polar_filter((rings,), _POLAR_BLOCK,
                  (F, weights, out, cp.int32(nlat), cp.int32(nlon),
                   cp.int32(rings), cp.int32(passes)),
                  shared_mem=shared_mem)
    return out


def available():
    return _module is not None
