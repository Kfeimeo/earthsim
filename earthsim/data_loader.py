"""Load a real analysis/reanalysis snapshot for model initialization.

The model grid is a regular cell-centred latitude/longitude grid.  This
module deliberately only handles the one-time remapping needed at startup;
it is not a data-assimilation or forecast component.

The preferred input is NetCDF opened by xarray.  A pressure-level file and a
single-level file may be supplied separately.  Common ERA5 and Copernicus
Marine variable names are accepted, while the explicit aliases in the
configuration-independent tables below also make small test datasets easy to
use.
"""
from __future__ import annotations

import os
from typing import Iterable

import numpy as np


class RealDataError(RuntimeError):
    """Raised when a requested real-data initialization is not usable."""


_LAT_NAMES = ("latitude", "lat", "nav_lat", "y")
_LON_NAMES = ("longitude", "lon", "nav_lon", "x")
_TIME_NAMES = ("time", "valid_time", "forecast_reference_time", "date")
_LEVEL_NAMES = ("level", "pressure_level", "isobaricInhPa", "plev",
                "height", "heightAboveGround", "altitude", "z")


def _as_paths(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v]
    return [str(value)]


def _open_datasets(paths: Iterable[str]):
    try:
        import xarray as xr
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise RealDataError(
            "real initialization requires xarray; install requirements.txt"
        ) from exc

    datasets = []
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"real-data file not found: {path}")
        try:
            datasets.append(xr.open_dataset(path))
        except Exception as exc:
            raise RealDataError(f"cannot open NetCDF file {path}: {exc}") from exc
    if not datasets:
        return None
    if len(datasets) == 1:
        return datasets[0]
    try:
        return xr.merge(datasets, compat="override", join="outer")
    except Exception as exc:
        for ds in datasets:
            ds.close()
        raise RealDataError(f"cannot merge real-data files: {exc}") from exc


def _coord_name(ds, candidates, kind: str):
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    for name, coord in ds.coords.items():
        standard = str(coord.attrs.get("standard_name", "")).lower()
        axis = str(coord.attrs.get("axis", "")).upper()
        if kind == "lat" and (standard == "latitude" or axis == "Y"):
            return name
        if kind == "lon" and (standard == "longitude" or axis == "X"):
            return name
        if kind == "time" and (standard == "time" or axis == "T"):
            return name
    return None


def _find_variable(ds, aliases, *, exclude=()):
    excluded = {str(v).lower() for v in exclude}
    for alias in aliases:
        for name in ds.variables:
            if name.lower() == alias.lower() and name.lower() not in excluded:
                return ds[name]
    for name, var in ds.data_vars.items():
        if name.lower() in excluded:
            continue
        standard = str(var.attrs.get("standard_name", "")).lower()
        long_name = str(var.attrs.get("long_name", "")).lower()
        for alias in aliases:
            a = alias.lower().replace("_", " ")
            if a == standard or a in long_name:
                return var
    return None


def _select_time(da, ds, target):
    time_name = _coord_name(ds, _TIME_NAMES, "time")
    if not time_name or time_name not in da.dims:
        return da
    if target:
        try:
            return da.sel({time_name: np.datetime64(str(target))}, method="nearest")
        except Exception:
            pass
    return da.isel({time_name: 0})


def _unit_factor(da, default=1.0):
    unit = str(da.attrs.get("units", "")).lower().replace(" ", "")
    if unit in {"c", "°c", "degc", "celsius", "degrees_celsius"}:
        return "celsius"
    if unit in {"hpa", "mb", "millibar", "millibars"}:
        return 100.0
    if unit in {"pa", "pascal", "pascals"}:
        return 1.0
    if unit in {"g/kg", "gkg-1", "gkg^-1"}:
        return 1e-3
    if unit in {"mm/day", "mmd-1"}:
        return 1.0 / 86400.0
    return default


def _to_numpy(da):
    values = np.asarray(da.values, dtype=np.float64)
    values = np.squeeze(values)
    if values.ndim != 2:
        raise RealDataError(
            f"variable {da.name!r} must be a 2-D horizontal field after selection"
        )
    return values


def _prepare_horizontal(da, ds, target_time):
    lat_name = _coord_name(ds, _LAT_NAMES, "lat")
    lon_name = _coord_name(ds, _LON_NAMES, "lon")
    if not lat_name or not lon_name or lat_name not in da.dims or lon_name not in da.dims:
        raise RealDataError(f"variable {da.name!r} has no recognizable lat/lon dimensions")
    da = _select_time(da, ds, target_time)
    extra = [dim for dim in da.dims if dim not in (lat_name, lon_name)]
    for dim in extra:
        # Surface ocean products often retain a singleton depth dimension;
        # for a surface field the first depth is the intended one.
        if dim in ds.dims and int(ds.sizes[dim]) >= 1:
            da = da.isel({dim: 0})
    extra = [dim for dim in da.dims if dim not in (lat_name, lon_name)]
    if extra:
        raise RealDataError(f"variable {da.name!r} still has dimensions {extra}")
    da = da.transpose(lat_name, lon_name)
    lat = np.asarray(ds[lat_name].values, dtype=np.float64).reshape(-1)
    lon = np.mod(np.asarray(ds[lon_name].values, dtype=np.float64).reshape(-1), 360.0)
    values = np.asarray(da.values, dtype=np.float64)
    if lat.size != values.shape[0] or lon.size != values.shape[1]:
        raise RealDataError(f"coordinate shape mismatch for variable {da.name!r}")
    return _prepare_values(lat, lon, values)


def _prepare_values(lat, lon, values):
    """Sort and de-duplicate coordinates for a raw 2-D field."""
    lat = np.asarray(lat, dtype=np.float64).reshape(-1)
    lon = np.mod(np.asarray(lon, dtype=np.float64).reshape(-1), 360.0)
    values = np.asarray(values, dtype=np.float64)
    lat_order = np.argsort(lat)
    lon_order = np.argsort(lon)
    lat, values = lat[lat_order], values[lat_order]
    lon, values = lon[lon_order], values[:, lon_order]
    lon, unique = np.unique(lon, return_index=True)
    values = values[:, unique]
    return lat, lon, values


def _interp_horizontal(lat, lon, values, target_lats, target_lons):
    """Bilinearly remap a regular field, with periodic longitude."""
    if len(lat) < 2 or len(lon) < 2:
        raise RealDataError("real-data grid needs at least two latitudes and longitudes")
    lon_ext = np.concatenate(([lon[-1] - 360.0], lon, [lon[0] + 360.0]))
    values_ext = np.concatenate((values[:, -1:], values, values[:, :1]), axis=1)
    qlon = np.mod(np.asarray(target_lons, dtype=np.float64), 360.0)
    qlon = np.where(qlon < lon_ext[0], qlon + 360.0, qlon)
    qlon = np.where(qlon > lon_ext[-1], qlon - 360.0, qlon)
    along = np.empty((len(lat), len(qlon)), dtype=np.float64)
    for i in range(len(lat)):
        along[i] = np.interp(qlon, lon_ext, values_ext[i])
    out = np.empty((len(target_lats), len(target_lons)), dtype=np.float64)
    for j in range(len(target_lons)):
        out[:, j] = np.interp(target_lats, lat, along[:, j])
    return out.astype(np.float32)


def _horizontal(da, ds, target_time, target_lats, target_lons):
    lat, lon, values = _prepare_horizontal(da, ds, target_time)
    return _interp_horizontal(lat, lon, values, target_lats, target_lons)


def _pressure_or_height(da, ds):
    for name in _LEVEL_NAMES:
        if name in da.dims and name in ds:
            return name, np.asarray(ds[name].values, dtype=np.float64).reshape(-1)
    for dim in da.dims:
        if dim not in (_coord_name(ds, _LAT_NAMES, "lat"),
                       _coord_name(ds, _LON_NAMES, "lon")):
            if dim in ds.coords:
                return dim, np.asarray(ds[dim].values, dtype=np.float64).reshape(-1)
    return None, None


def _level_kind(ds, name):
    if not name:
        return None
    units = str(ds[name].attrs.get("units", "")).lower()
    standard = str(ds[name].attrs.get("standard_name", "")).lower()
    if "pressure" in standard or units in {"pa", "hpa", "mb", "millibar"}:
        return "pressure"
    if "height" in standard or "altitude" in standard or "m" in units:
        return "height"
    # ERA5's conventional pressure coordinate is often missing units.
    values = np.asarray(ds[name].values, dtype=float)
    return "pressure" if np.nanmax(values) > 150 else "height"


def _profile(da, ds, target_time, target_lats, target_lons, levels_m):
    level_name, source_levels = _pressure_or_height(da, ds)
    if not level_name:
        return np.stack([_horizontal(da, ds, target_time, target_lats, target_lons)
                          for _ in levels_m])
    da = _select_time(da, ds, target_time)
    lat_name = _coord_name(ds, _LAT_NAMES, "lat")
    lon_name = _coord_name(ds, _LON_NAMES, "lon")
    da = da.transpose(level_name, lat_name, lon_name)
    source_levels = np.array(source_levels, dtype=np.float64, copy=True)
    kind = _level_kind(ds, level_name)
    if kind == "pressure":
        units = str(ds[level_name].attrs.get("units", "")).lower()
        if units in {"hpa", "mb", "millibar"} or np.nanmax(source_levels) < 2000:
            source_levels *= 100.0
        target_coord = 101325.0 * np.exp(-np.asarray(levels_m) / 8400.0)
    else:
        target_coord = np.asarray(levels_m, dtype=np.float64)
    order = np.argsort(source_levels)
    source_levels = source_levels[order]
    data = np.asarray(da.values, dtype=np.float64)[order]
    remapped = []
    for k in range(len(source_levels)):
        # Reuse the horizontal remapper, without rebuilding an xarray object.
        lat = np.asarray(ds[lat_name].values, dtype=np.float64).reshape(-1)
        lon = np.asarray(ds[lon_name].values, dtype=np.float64).reshape(-1)
        lat, lon, values = _prepare_values(lat, lon, data[k])
        remapped.append(_interp_horizontal(lat, lon, values,
                                           target_lats, target_lons))
    source = np.stack(remapped, axis=0).astype(np.float64)
    # np.interp is applied independently to every target grid point.
    out = np.empty((len(levels_m), len(target_lats), len(target_lons)), np.float32)
    for y in range(len(target_lats)):
        for x in range(len(target_lons)):
            out[:, y, x] = np.interp(
                target_coord, source_levels, source[:, y, x])
    return out


def _field(ds, aliases, target_time, target_lats, target_lons):
    da = _find_variable(ds, aliases)
    if da is None:
        return None
    return _horizontal(da, ds, target_time, target_lats, target_lons)


def _temperature(values, da):
    if values is None:
        return None
    factor = _unit_factor(da)
    return values + 273.15 if factor == "celsius" else values


def _convert(values, da, default=1.0):
    if values is None:
        return None
    factor = _unit_factor(da, default)
    if factor == "celsius":
        return values + 273.15
    return values * factor


def _find_and_field(ds, aliases, target_time, target_lats, target_lons,
                    *, profile=False, levels_m=()):
    da = _find_variable(ds, aliases)
    if da is None:
        return None, None
    result = (_profile(da, ds, target_time, target_lats, target_lons, levels_m)
              if profile else _horizontal(da, ds, target_time, target_lats, target_lons))
    return result, da


def load_real_initialization(cfg, lats, lons, levels_m):
    """Return model-shaped initial fields from configured NetCDF snapshots."""
    data_cfg = cfg.data
    atmosphere_paths = _as_paths(getattr(data_cfg, "atmosphere_file", ""))
    surface_paths = _as_paths(getattr(data_cfg, "surface_file", ""))
    ocean_paths = _as_paths(getattr(data_cfg, "ocean_file", ""))
    atmosphere = _open_datasets(atmosphere_paths + surface_paths)
    ocean = _open_datasets(ocean_paths)
    if atmosphere is None:
        raise FileNotFoundError("data.atmosphere_file is empty")
    target_time = getattr(data_cfg, "init_time", "") or str(cfg.time.start)
    fields = {}
    try:
        temp, temp_da = _find_and_field(
            atmosphere, ("t", "temperature", "air_temperature"), target_time,
            lats, lons, profile=True, levels_m=levels_m)
        t2m, t2m_da = _find_and_field(
            atmosphere, ("t2m", "2m_temperature", "air_temperature_2m"),
            target_time, lats, lons)
        if temp is None and t2m is None:
            raise RealDataError("real atmosphere data has no temperature variable")
        if temp is None:
            t2m = _temperature(t2m, t2m_da)
            temp = np.stack([t2m - 0.0065 * (z - levels_m[0])
                             for z in levels_m]).astype(np.float32)
        else:
            temp = _temperature(temp, temp_da)

        q, q_da = _find_and_field(
            atmosphere, ("q", "specific_humidity", "specific_humidity_kgkg"),
            target_time, lats, lons, profile=True, levels_m=levels_m)
        q2, q2_da = _find_and_field(
            atmosphere, ("q2m", "2m_specific_humidity", "specific_humidity_2m"),
            target_time, lats, lons)
        d2m, d2m_da = _find_and_field(
            atmosphere, ("d2m", "2m_dewpoint_temperature", "dewpoint_2m"),
            target_time, lats, lons)
        if q is None:
            if q2 is not None:
                q = np.stack([q2] * len(levels_m)).astype(np.float32)
            elif d2m is not None:
                td = _temperature(d2m, d2m_da)
                # The qsat approximation used by the model is also the
                # safest conversion for matching its moisture definition.
                es = 610.78 * np.exp(17.27 * (td - 273.15)
                                     / np.maximum(td - 35.85, 1.0))
                q = np.stack([0.8 * 0.622 * es / 1e5] * len(levels_m))
            else:
                q = np.empty_like(temp)
                es = 610.78 * np.exp(17.27 * (temp - 273.15)
                                     / np.maximum(temp - 35.85, 1.0))
                q[:] = 0.65 * 0.622 * es / 1e5
        else:
            q = _convert(q, q_da)

        u, u_da = _find_and_field(
            atmosphere, ("u", "u_component_of_wind", "eastward_wind"),
            target_time, lats, lons, profile=True, levels_m=levels_m)
        v, v_da = _find_and_field(
            atmosphere, ("v", "v_component_of_wind", "northward_wind"),
            target_time, lats, lons, profile=True, levels_m=levels_m)
        u10, u10_da = _find_and_field(
            atmosphere, ("u10", "10m_u_component_of_wind", "10m_eastward_wind"),
            target_time, lats, lons)
        v10, v10_da = _find_and_field(
            atmosphere, ("v10", "10m_v_component_of_wind", "10m_northward_wind"),
            target_time, lats, lons)
        if u is None:
            u = np.stack([u10 if u10 is not None else np.zeros_like(temp[0])]
                         * len(levels_m))
        if v is None:
            v = np.stack([v10 if v10 is not None else np.zeros_like(temp[0])]
                         * len(levels_m))

        mslp, mslp_da = _find_and_field(
            atmosphere, ("msl", "mslp", "mean_sea_level_pressure",
                         "msl_pressure"), target_time, lats, lons)
        if mslp is not None:
            mslp = _convert(mslp, mslp_da)
            mslp_hpa = mslp / 100.0
        else:
            mslp_hpa = np.full((len(lats), len(lons)), 1013.0, np.float32)

        skin, skin_da = _find_and_field(
            atmosphere, ("skt", "skin_temperature", "surface_temperature"),
            target_time, lats, lons)
        sst, sst_da = _find_and_field(
            atmosphere, ("sst", "sea_surface_temperature", "tos"),
            target_time, lats, lons)
        surface = _temperature(skin, skin_da) if skin is not None else None
        ocean_surface = None
        if sst is not None:
            sst = _temperature(sst, sst_da)
            ocean_surface = sst
        if surface is None:
            surface = temp[0].copy()

        cloud, _ = _find_and_field(
            atmosphere, ("tcc", "total_cloud_cover", "cloud_cover"),
            target_time, lats, lons)
        precip, precip_da = _find_and_field(
            atmosphere, ("tp", "total_precipitation", "precipitation"),
            target_time, lats, lons)
        fields.update(temp=temp, q=q, u=u, v=v, mslp_hpa=mslp_hpa,
                      surface=surface, cloud=cloud,
                      ocean_surface=ocean_surface,
                      precip=_convert(precip, precip_da) if precip is not None else None)

        if ocean is not None:
            ou, _ = _find_and_field(
                ocean, ("uo", "uo_surface", "eastward_sea_water_velocity",
                        "surface_eastward_current"), target_time, lats, lons)
            ov, _ = _find_and_field(
                ocean, ("vo", "vo_surface", "northward_sea_water_velocity",
                        "surface_northward_current"), target_time, lats, lons)
            osst, osst_da = _find_and_field(
                ocean, ("thetao", "sst", "sea_surface_temperature", "tos"),
                target_time, lats, lons)
            sea_ice, _ = _find_and_field(
                ocean, ("siconc", "sea_ice_fraction", "ice_concentration"),
                target_time, lats, lons)
            fields.update(ou=ou, ov=ov,
                          ocean_surface=(_temperature(osst, osst_da)
                                         if osst is not None else ocean_surface),
                          sea_ice=sea_ice)
    finally:
        atmosphere.close()
        if ocean is not None:
            ocean.close()
    return fields
