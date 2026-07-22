"""加载并包装 kernels.cu 中的手写 CUDA kernel。仅在 cuda 后端下使用。"""
import os

_module = None
_adv_diff = None
_gradient = None
_divergence = None
_mass_transport = None
_hydrostatic_column = None
_polar_filter = None
_ADV_BLOCK = (16, 16)  # Must match the static shared-memory tile in kernels.cu.
_POLAR_BLOCK = (256,)


def load():
    global _module, _adv_diff, _gradient, _divergence
    global _mass_transport, _hydrostatic_column, _polar_filter
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
        _mass_transport = _module.get_function("mass_transport")
        _hydrostatic_column = _module.get_function("hydrostatic_column")
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


def mass_transport(field, u, v, mu_old, mu_new, sigma_thickness,
                   div_mass, interface_flux, invdx, invdy,
                   diffusivity, dt):
    """Flux-form horizontal/vertical transport for a batched 3-D field."""
    import cupy as cp
    field, nlat, nlon, block, grid = _field_layout(cp, field)
    if field.ndim != 3:
        raise ValueError("mass transport requires a 3-D [level, lat, lon] field")
    nz = field.shape[0]
    expected = field.shape
    for name, array in (("u", u), ("v", v), ("div_mass", div_mass)):
        if array.shape != expected or array.dtype != cp.float32:
            raise ValueError(f"{name} must match the float32 field shape")
    if interface_flux.shape != (nz + 1, nlat, nlon):
        raise ValueError("interface_flux must have shape [nz+1, nlat, nlon]")
    if interface_flux.dtype != cp.float32:
        raise TypeError("interface_flux must be float32")
    if mu_old.shape != (nlat, nlon) or mu_new.shape != (nlat, nlon):
        raise ValueError("column mass arrays must match [nlat, nlon]")

    u = cp.ascontiguousarray(u)
    v = cp.ascontiguousarray(v)
    mu_old = cp.ascontiguousarray(mu_old, dtype=cp.float32)
    mu_new = cp.ascontiguousarray(mu_new, dtype=cp.float32)
    sigma_thickness = cp.ascontiguousarray(
        cp.asarray(sigma_thickness, dtype=cp.float32))
    div_mass = cp.ascontiguousarray(div_mass)
    interface_flux = cp.ascontiguousarray(interface_flux)
    invdx = cp.ascontiguousarray(invdx, dtype=cp.float32)
    if sigma_thickness.size != nz or invdx.size != nlat:
        raise ValueError("transport metric arrays do not match the field")

    out = cp.empty_like(field)
    _mass_transport(
        grid, block,
        (field, u, v, mu_old, mu_new, sigma_thickness,
         div_mass, interface_flux, invdx, cp.float32(invdy),
         cp.float32(diffusivity), cp.float32(dt), out,
         cp.int32(nz), cp.int32(nlat), cp.int32(nlon)))
    return out


def hydrostatic_state(temperature, humidity, surface_pressure,
                      sigma_interfaces, top_pressure,
                      surface_geopotential, gas_constant):
    """Integrate hydrostatic pressure/geopotential for every GPU column."""
    import cupy as cp
    temperature, nlat, nlon, _, _ = _field_layout(cp, temperature)
    if temperature.ndim != 3:
        raise ValueError("hydrostatic state requires [level, lat, lon] arrays")
    nz = temperature.shape[0]
    if humidity.shape != temperature.shape or humidity.dtype != cp.float32:
        raise ValueError("humidity must match the float32 temperature field")
    if surface_pressure.shape != (nlat, nlon):
        raise ValueError("surface pressure must match [nlat, nlon]")
    if surface_geopotential.shape != (nlat, nlon):
        raise ValueError("surface geopotential must match [nlat, nlon]")

    humidity = cp.ascontiguousarray(humidity)
    surface_pressure = cp.ascontiguousarray(
        surface_pressure, dtype=cp.float32)
    surface_geopotential = cp.ascontiguousarray(
        surface_geopotential, dtype=cp.float32)
    sigma_interfaces = cp.ascontiguousarray(
        cp.asarray(sigma_interfaces, dtype=cp.float32))
    if sigma_interfaces.size != nz + 1:
        raise ValueError("sigma interface count must be nz + 1")

    interface_shape = (nz + 1, nlat, nlon)
    pressure_interfaces = cp.empty(interface_shape, dtype=cp.float32)
    pressure_layers = cp.empty_like(temperature)
    geopotential_layers = cp.empty_like(temperature)
    geopotential_interfaces = cp.empty(interface_shape, dtype=cp.float32)
    virtual_temperature = cp.empty_like(temperature)
    cells = nlat * nlon
    block = (256,)
    grid = ((cells + block[0] - 1) // block[0],)
    _hydrostatic_column(
        grid, block,
        (temperature, humidity, surface_pressure, sigma_interfaces,
         surface_geopotential, cp.float32(top_pressure),
         cp.float32(gas_constant), pressure_interfaces, pressure_layers,
         geopotential_layers, geopotential_interfaces, virtual_temperature,
         cp.int32(nz), cp.int32(cells)))
    return (pressure_interfaces, pressure_layers, geopotential_layers,
            geopotential_interfaces, virtual_temperature)


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
