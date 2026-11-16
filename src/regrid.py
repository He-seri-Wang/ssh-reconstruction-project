"""scipy drop-in for the challenge's pyinterp-based regrid (pyinterp is
conda-only and has no wheel on this stack): trilinear (lon, lat, time)
interpolation on the regular source grid + nearest-neighbour extrapolation for
any NaNs. pyinterp's gauss_seidel fill is only exercised when the mapping
output has holes — reconstructions here are gap-free.
"""
import numpy
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt


def oi_regrid(ds_source, ds_target):
    lon = ds_source["lon"].values; lat = ds_source["lat"].values
    t = ds_source["time"].values.astype("datetime64[ns]").astype(numpy.int64)
    v = ds_source["gssh"].values                    # (time, lat, lon)
    interp = RegularGridInterpolator(
        (lon, lat, t), v.transpose(2, 1, 0),        # (lon, lat, time)
        method="linear", bounds_error=False, fill_value=numpy.nan)
    q_lon, q_lat, q_t = numpy.meshgrid(
        ds_target["lon"].values, ds_target["lat"].values,
        ds_target["time"].values.astype("datetime64[ns]").astype(numpy.int64),
        indexing="ij")
    out = interp(numpy.stack([q_lon, q_lat, q_t], -1)).transpose(2, 1, 0)
    if numpy.isnan(out).any():
        idx = distance_transform_edt(numpy.isnan(out),
                                     return_distances=False, return_indices=True)
        out = out[tuple(idx)]
    return xr.Dataset({"sossheig": (("time", "lat", "lon"), out)},
                      coords={"time": ds_target["time"].values,
                              "lon": ds_target["lon"].values,
                              "lat": ds_target["lat"].values})
