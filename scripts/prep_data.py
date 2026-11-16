#!/usr/bin/env python3
"""SSH pilot data prep: gridded gt + daily-binned nadir obs on the 1/20deg challenge grid.

Grid = dc_ref daily-mean coarsened 3x (600 -> 200), i.e. the 4dvarnet-core gf-domain
200x200 convention at 1/20deg over [-65,-55]x[33,43]. Obs = 4-nadir along-track
ssh_obs binned per day per cell (paper: "aggregate the nadir pseudo-observations
on a daily basis"). Saves npz consumed by ssh_pilot_train.py / ssh_pilot_oi_year.py.
"""
import glob, os
import numpy as np
import xarray as xr

CH = os.environ.get("SSH_CHALLENGE_DIR", "/scratch/zceehw4/ssh_challenge")
OUT = os.path.join(os.environ.get("SSH_WORK_DIR", "/scratch/zceehw4/ssh_pilot"),
                   "ssh_pilot_data.npz")
T0 = np.datetime64("2012-10-01")
NDAYS = 365  # 2012-10-01 .. 2013-09-30
FOUR_NADIRS = [f"{CH}/dc_obs/2020a_SSH_mapping_NATL60_{m}.nc" for m in
               ("envisat", "geosat2", "topex-poseidon_interleaved", "jason1")]

# ---- gt: 365 daily files -> daily mean -> coarsen 600->200 -----------------
ref_files = sorted(glob.glob(f"{CH}/dc_ref/*.nc"))
assert len(ref_files) == NDAYS, f"expected {NDAYS} ref files, got {len(ref_files)}"
gt = np.empty((NDAYS, 200, 200), dtype=np.float32)
for i, f in enumerate(ref_files):
    ds = xr.open_dataset(f, decode_times=True)
    day = ds.sossheig.mean("time")                      # daily mean, 600x600
    day = day.coarsen(lat=3, lon=3).mean()              # 200x200 @ ~1/20deg
    gt[i] = day.values.astype(np.float32)
    ds.close()
lat = day.lat.values.astype(np.float64)
lon = day.lon.values.astype(np.float64)
print(f"gt {gt.shape} lat[{lat[0]:.3f},{lat[-1]:.3f}] lon[{lon[0]:.3f},{lon[-1]:.3f}] "
      f"nan%={100*np.isnan(gt).mean():.3f}", flush=True)

# ---- obs: bin along-track ssh_obs to (day, cell) ----------------------------
def cell_idx(vals, centers):
    # nearest-center index via searchsorted on midpoints (edges between centers)
    edges = np.empty(len(centers) + 1)
    edges[1:-1] = 0.5 * (centers[1:] + centers[:-1])
    edges[0], edges[-1] = centers[0] - 0.5 * (centers[1] - centers[0]), centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return np.searchsorted(edges, vals, side="right") - 1

obs = np.full((NDAYS, 200, 200), np.nan, dtype=np.float32)
for f in FOUR_NADIRS:
    ds = xr.open_dataset(f)
    t = ds.time.values
    day_idx = ((t - np.datetime64(T0)) / np.timedelta64(1, "D")).astype(int)
    keep = (day_idx >= 0) & (day_idx < NDAYS)
    di = day_idx[keep]
    o_lon = ds.lon.values[keep].astype(np.float64)
    o_lon = np.where(o_lon < 0, o_lon + 360.0, o_lon)   # obs files use 0..360
    ii = cell_idx(ds.lat.values[keep].astype(np.float64), lat)
    jj = cell_idx(o_lon, lon % 360.0)
    vv = ds.ssh_obs.values[keep].astype(np.float32)
    ok = (ii >= 0) & (ii < 200) & (jj >= 0) & (jj < 200) & np.isfinite(vv)
    # accumulate sums/counts per (day, cell), then divide — handles multiple
    # along-track samples falling in one 0.05deg cell on one day
    flat = di[ok] * 40000 + ii[ok] * 200 + jj[ok]
    sums = np.bincount(flat, weights=vv[ok], minlength=NDAYS * 40000).astype(np.float64)
    cnts = np.bincount(flat, minlength=NDAYS * 40000)
    hit = cnts > 0
    o = obs.reshape(-1)
    o[hit] = (sums[hit] / cnts[hit]).astype(np.float32)
    n = int(hit.sum())
    print(f"{f.split('_')[-1][:-3]}: {n} obs-cells/day avg {n/NDAYS:.0f}", flush=True)
    ds.close()

mask = ~np.isnan(obs)
print(f"obs coverage: {100*mask.mean():.2f}% cells/day, daily obs pts total {int(mask.sum())}", flush=True)

dates = T0 + np.arange(NDAYS).astype("timedelta64[D]")
np.savez_compressed(OUT, gt=gt, obs=obs, mask=mask, lat=lat, lon=lon, dates=dates.astype("datetime64[D]"))
print("saved", OUT, flush=True)
