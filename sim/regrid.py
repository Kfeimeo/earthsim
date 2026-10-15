"""Regridding between regular latitude/longitude grids.

The model grid has cell centres at ``lat = -90 + dlat (i + 1/2)`` and
``lon = dlon j`` (periodic).  A field is moved between two such grids with a
separable linear operator ``W_lat @ F @ W_lon.T``.  Along each axis the 1-D
operator is chosen by direction:

* target coarser than source: conservative aggregation, i.e. the
  area-weighted mean of every source cell overlapping the target cell
  (latitude weights include ``cos(lat)``), so means and column budgets of
  the recorded fields are preserved;
* target finer than source: linear interpolation between neighbouring
  source cell centres (periodic in longitude, clamped at the poles), which
  smooths rather than replicating blocks.

Because the two axes are treated independently, a source that is finer in
one direction and coarser in the other is handled correctly too.
"""
import numpy as np


def grid_centres(nlat, nlon):
    dlat = 180.0 / nlat
    dlon = 360.0 / nlon
    return (-90.0 + dlat * (np.arange(nlat) + 0.5), dlon * np.arange(nlon))


def _edges_from_centres(centres, periodic, span):
    centres = np.asarray(centres, dtype=np.float64)
    if centres.size == 1:
        step = span
    else:
        step = float(np.mean(np.diff(centres)))
    return np.concatenate([centres - 0.5 * step, [centres[-1] + 0.5 * step]])


def _overlap_matrix(src_centres, dst_centres, periodic, span, weights=None):
    """Fraction of each source cell that lies inside each target cell."""
    src_edges = _edges_from_centres(src_centres, periodic, span)
    dst_edges = _edges_from_centres(dst_centres, periodic, span)
    nsrc, ndst = len(src_centres), len(dst_centres)
    if weights is None:
        weights = np.ones(nsrc)
    matrix = np.zeros((ndst, nsrc))
    shifts = (-span, 0.0, span) if periodic else (0.0,)
    for shift in shifts:
        lo = np.maximum(dst_edges[:-1, None], src_edges[None, :-1] + shift)
        hi = np.minimum(dst_edges[1:, None], src_edges[None, 1:] + shift)
        matrix += np.clip(hi - lo, 0.0, None)
    matrix *= weights[None, :]
    total = matrix.sum(axis=1, keepdims=True)
    return matrix / np.where(total > 0.0, total, 1.0)


def _interpolation_matrix(src_centres, dst_centres, periodic, span):
    """Linear interpolation weights from source centres to target centres."""
    src = np.asarray(src_centres, dtype=np.float64)
    dst = np.asarray(dst_centres, dtype=np.float64)
    nsrc, ndst = src.size, dst.size
    matrix = np.zeros((ndst, nsrc))
    if nsrc == 1:
        matrix[:, 0] = 1.0
        return matrix
    if periodic:
        extended = np.concatenate([src, [src[0] + span]])
        x = np.mod(dst - src[0], span) + src[0]
        index = np.clip(np.searchsorted(extended, x, side="right") - 1, 0, nsrc - 1)
        upper = (index + 1) % nsrc
        fraction = (x - extended[index]) / (extended[index + 1] - extended[index])
    else:
        x = np.clip(dst, src[0], src[-1])
        index = np.clip(np.searchsorted(src, x, side="right") - 1, 0, nsrc - 2)
        upper = index + 1
        fraction = (x - src[index]) / (src[upper] - src[index])
    fraction = np.clip(fraction, 0.0, 1.0)
    matrix[np.arange(ndst), index] += 1.0 - fraction
    matrix[np.arange(ndst), upper] += fraction
    return matrix


def axis_operator(src_centres, dst_centres, periodic, span, weights=None):
    """1-D operator: aggregation when coarsening, interpolation when refining."""
    if len(dst_centres) < len(src_centres):
        return _overlap_matrix(src_centres, dst_centres, periodic, span, weights)
    return _interpolation_matrix(src_centres, dst_centres, periodic, span)


class Regridder:
    """Reusable operator between two regular latitude/longitude grids."""

    def __init__(self, src_shape, dst_shape):
        self.src_shape = (int(src_shape[0]), int(src_shape[1]))
        self.dst_shape = (int(dst_shape[0]), int(dst_shape[1]))
        self.identity = self.src_shape == self.dst_shape
        src_lat, src_lon = grid_centres(*self.src_shape)
        dst_lat, dst_lon = grid_centres(*self.dst_shape)
        self.lat_operator = axis_operator(
            src_lat, dst_lat, periodic=False, span=180.0,
            weights=np.cos(np.radians(src_lat)))
        self.lon_operator = axis_operator(
            src_lon, dst_lon, periodic=True, span=360.0)
        self.coarsens_lat = self.dst_shape[0] < self.src_shape[0]
        self.coarsens_lon = self.dst_shape[1] < self.src_shape[1]

    def describe(self):
        if self.identity:
            return "identity"
        lat = "aggregate" if self.coarsens_lat else "interpolate"
        lon = "aggregate" if self.coarsens_lon else "interpolate"
        return (f"{self.src_shape[0]}x{self.src_shape[1]} -> "
                f"{self.dst_shape[0]}x{self.dst_shape[1]} (lat: {lat}, lon: {lon})")

    def __call__(self, field):
        """Regrid an array whose last two axes are (lat, lon)."""
        field = np.asarray(field)
        if field.ndim < 2 or field.shape[-2:] != self.src_shape:
            raise ValueError(
                f"field shape {field.shape} does not end with {self.src_shape}")
        if self.identity:
            return field
        dtype = field.dtype if np.issubdtype(field.dtype, np.floating) else np.float32
        out = np.einsum("ab,...bc,dc->...ad", self.lat_operator,
                        field.astype(np.float64), self.lon_operator)
        return out.astype(dtype)


def regrid_fields(fields, dst_shape, skip=("subsolar",)):
    """Regrid every (…, nlat, nlon) array of a frame dictionary."""
    sample = next((v for k, v in fields.items()
                   if k not in skip and getattr(v, "ndim", 0) >= 2), None)
    if sample is None:
        return dict(fields)
    regridder = Regridder(sample.shape[-2:], dst_shape)
    out = {}
    for key, value in fields.items():
        value = np.asarray(value)
        if key in skip or value.ndim < 2 or value.shape[-2:] != regridder.src_shape:
            out[key] = value
        else:
            out[key] = regridder(value)
    return out


def remap_levels(field, src_pressure, dst_pressure):
    """Linear-in-ln(p) vertical remap of a (nz_src, nlat, nlon) field.

    ``src_pressure``/``dst_pressure`` are the layer pressures of the two
    grids, either 1-D (one column for all cells) or full 3-D arrays, ordered
    from the surface upward.  Values beyond the source range are clamped to
    the nearest source layer.  Vectorised over all columns.
    """
    field = np.asarray(field, dtype=np.float64)
    nz_src = field.shape[0]
    src = np.log(np.asarray(src_pressure, dtype=np.float64))
    dst = np.log(np.asarray(dst_pressure, dtype=np.float64))
    if src.ndim == 1:
        src = src[:, None, None] * np.ones((1,) + field.shape[1:])
    if dst.ndim == 1:
        dst = dst[:, None, None] * np.ones((1,) + field.shape[1:])
    if nz_src == 1:
        return np.repeat(field, dst.shape[0], axis=0).astype(np.float32)
    # Source pressure decreases with level: ln p is descending along axis 0.
    out = np.empty((dst.shape[0],) + field.shape[1:], dtype=np.float64)
    for k in range(dst.shape[0]):
        target = dst[k]
        # Number of source layers with pressure above (greater than) target.
        below = np.clip((src > target).sum(axis=0) - 1, 0, nz_src - 2)
        upper = below + 1
        p_lo = np.take_along_axis(src, below[None], axis=0)[0]
        p_hi = np.take_along_axis(src, upper[None], axis=0)[0]
        f_lo = np.take_along_axis(field, below[None], axis=0)[0]
        f_hi = np.take_along_axis(field, upper[None], axis=0)[0]
        fraction = (p_lo - target) / np.maximum(p_lo - p_hi, 1e-9)
        fraction = np.clip(fraction, 0.0, 1.0)
        out[k] = f_lo + fraction * (f_hi - f_lo)
    return out.astype(np.float32)
