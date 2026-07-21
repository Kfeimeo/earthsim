"""加载并包装 kernels.cu 中的手写 CUDA kernel。仅在 cuda 后端下使用。"""
import os

_module = None
_adv_diff = None
_gradient = None
_divergence = None
_polar_filter = None
_ADV_BLOCK = (16, 16)  # Must match the static shared-memory tile in kernels.cu.
_POLAR_BLOCK = (256,)


def load():
    global _module, _adv_diff, _gradient, _divergence, _polar_filter
    if _module is not None:
        return True
    try:
        import cupy as cp
        with open(os.path.join(os.path.dirname(__file__), "kernels.cu"),
                  encoding="utf-8") as source_file:
            src = source_file.read()
        _module = cp.RawModule(code=src, options=("--use_fast_math",))
        _adv_diff = _module.get_function("adv_diff")
        _gradient = _module.get_function("gradient")
        _divergence = _module.get_function("divergence")
        _polar_filter = _module.get_function("polar_filter")
        return True
    except Exception:
        _module = None
        return False


def adv_diff(F, u, v, invdx, invdy, K, dt):
    """返回批量平流扩散结果；末两维为 (nlat, nlon)。"""
    import cupy as cp
    if F.ndim < 2:
        raise ValueError("advection field must have at least two dimensions")
    if F.shape != u.shape or F.shape != v.shape:
        raise ValueError("F, u, and v must have identical shapes")
    if F.dtype != cp.float32 or u.dtype != cp.float32 or v.dtype != cp.float32:
        raise TypeError("advection CUDA kernel requires float32 inputs")

    F = cp.ascontiguousarray(F)
    u = cp.ascontiguousarray(u)
    v = cp.ascontiguousarray(v)
    invdx = cp.ascontiguousarray(invdx, dtype=cp.float32)
    nlat, nlon = F.shape[-2:]
    if invdx.size != nlat:
        raise ValueError("invdx length must match the latitude dimension")
    planes = F.size // (nlat * nlon)
    out = cp.empty_like(F)
    block = _ADV_BLOCK
    grid = ((nlon + block[0] - 1) // block[0],
            (nlat + block[1] - 1) // block[1],
            planes)
    _adv_diff(grid, block,
              (F, u, v, invdx, cp.float32(invdy), cp.float32(K),
               cp.float32(dt), out, cp.int32(nlat), cp.int32(nlon)))
    return out


def _field_layout(cp, F):
    if F.ndim < 2:
        raise ValueError("field must have at least two dimensions")
    if F.dtype != cp.float32:
        raise TypeError("CUDA stencil kernels require float32 inputs")
    F = cp.ascontiguousarray(F)
    nlat, nlon = F.shape[-2:]
    planes = F.size // (nlat * nlon)
    block = _ADV_BLOCK
    grid = ((nlon + block[0] - 1) // block[0],
            (nlat + block[1] - 1) // block[1], planes)
    return F, nlat, nlon, block, grid


def gradient(F, invdx, invdy):
    """Return centred ddx and ddy for a 2-D or batched float32 field."""
    import cupy as cp
    F, nlat, nlon, block, grid = _field_layout(cp, F)
    invdx = cp.ascontiguousarray(invdx, dtype=cp.float32)
    if invdx.size != nlat:
        raise ValueError("invdx length must match the latitude dimension")
    out_x = cp.empty_like(F)
    out_y = cp.empty_like(F)
    _gradient(grid, block,
              (F, invdx, cp.float32(invdy), out_x, out_y,
               cp.int32(nlat), cp.int32(nlon)))
    return out_x, out_y


def divergence(u, v, invdx, invdy, coslat, invcoslat):
    """Return spherical divergence for 2-D or batched wind fields."""
    import cupy as cp
    u, nlat, nlon, block, grid = _field_layout(cp, u)
    if v.shape != u.shape or v.dtype != cp.float32:
        raise ValueError("u and v must be identically shaped float32 arrays")
    v = cp.ascontiguousarray(v)
    invdx = cp.ascontiguousarray(invdx, dtype=cp.float32)
    coslat = cp.ascontiguousarray(coslat, dtype=cp.float32)
    invcoslat = cp.ascontiguousarray(invcoslat, dtype=cp.float32)
    if invdx.size != nlat or coslat.size != nlat or invcoslat.size != nlat:
        raise ValueError("latitude metric arrays must match the field")
    out = cp.empty_like(u)
    _divergence(grid, block,
                (u, v, invdx, coslat, invcoslat, cp.float32(invdy), out,
                 cp.int32(nlat), cp.int32(nlon)))
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
