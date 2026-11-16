#!/usr/bin/env python3
"""Per-mission obs fields for the L2 multimodal-fusion arm: 4 nadir missions kept
as separate (obs, mask) channel stacks instead of pooled (same binning convention
as prep_data.py). Saves ssh_pilot_data_missions.npz."""
import os
import numpy as np
import xarray as xr

CH = os.environ.get("SSH_CHALLENGE_DIR", "/scratch/zceehw4/ssh_challenge")
WORK = os.environ.get("SSH_WORK_DIR", "/scratch/zceehw4/ssh_pilot")
OUT = f"{WORK}/ssh_pilot_data_missions.npz"
T0 = np.datetime64("2012-10-01")
NDAYS = 365
MISSIONS = ("envisat", "geosat2", "topex-poseidon_interleaved", "jason1")

d = np.load(os.environ.get("SSH_DATA_NPZ", f"{WORK}/ssh_pilot_data.npz"), allow_pickle=True)
lat, lon = d["lat"], d["lon"]


def cell_idx(vals, centers):
    edges = np.empty(len(centers) + 1)
    edges[1:-1] = 0.5 * (centers[1:] + centers[:-1])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return np.searchsorted(edges, vals, side="right") - 1


obs_m = np.full((4, NDAYS, 200, 200), np.nan, dtype=np.float32)
for k, m in enumerate(MISSIONS):
    ds = xr.open_dataset(f"{CH}/dc_obs/2020a_SSH_mapping_NATL60_{m}.nc")
    t = ds.time.values
    day = ((t - np.datetime64(T0)) / np.timedelta64(1, "D")).astype(int)
    keep = (day >= 0) & (day < NDAYS)
    di = day[keep]
    o_lon = ds.lon.values[keep].astype(np.float64)
    o_lon = np.where(o_lon < 0, o_lon + 360.0, o_lon)
    ii = cell_idx(ds.lat.values[keep].astype(np.float64), lat)
    jj = cell_idx(o_lon, lon % 360.0)
    vv = ds.ssh_obs.values[keep].astype(np.float32)
    ok = (ii >= 0) & (ii < 200) & (jj >= 0) & (jj < 200) & np.isfinite(vv)
    flat = di[ok] * 40000 + ii[ok] * 200 + jj[ok]
    sums = np.bincount(flat, weights=vv[ok], minlength=NDAYS * 40000).astype(np.float64)
    cnts = np.bincount(flat, minlength=NDAYS * 40000)
    hit = cnts > 0
    o = obs_m[k].reshape(-1)
    o[hit] = (sums[hit] / cnts[hit]).astype(np.float32)
    ds.close()
    print(f"{m}: {int(hit.sum())} obs-cells", flush=True)

mask_m = ~np.isnan(obs_m)
# sanity: pooled union must equal the pooled prep mask
pooled = mask_m.any(0)
d_mask = d["mask"].astype(bool)
print("union == pooled mask:", bool((pooled == d_mask).all()), flush=True)
np.savez_compressed(OUT, obs_m=obs_m, mask_m=mask_m, missions=np.array(MISSIONS))
print("saved", OUT, flush=True)
