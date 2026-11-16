#!/usr/bin/env python3
"""Full-year OI first guess at 1/20deg for the SSH pilot (4dvarnet input channel).

Challenge quickstart OI verbatim (Lx=Ly=1, Lt=7, noise=0.05, obs time coarsen 5)
on the 4-nadir obs, but dx=dy=0.05 (4dvarnet gf-domain native resolution) and the
full challenge year 2012-10-01..2013-09-30. Output -> /scratch/zceehw4/ssh_pilot/oi_year.npz
"""
import os, sys, time
import numpy
import xarray as xr

numpy.float = float            # challenge code predates numpy 1.24 alias removal
numpy.int = int

_orig_mfdataset = xr.open_mfdataset          # dask-threaded segfault on this host
def _mfdataset(*a, **k):
    pp = k.pop("preprocess", None)
    if pp is not None:
        k["preprocess"] = lambda ds: pp(ds).load()
    return _orig_mfdataset(*a, **{**k, "parallel": False})
xr.open_mfdataset = _mfdataset

CH = os.environ.get("SSH_CHALLENGE_DIR", "/scratch/zceehw4/ssh_challenge")
WORK = os.environ.get("SSH_WORK_DIR", "/scratch/zceehw4/ssh_pilot")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "third_party", "2020a_ssh_challenge"))
from src.mod_oi import oi_grid, oi_param, oi_core          # noqa: E402
from src.mod_inout import read_obs                          # noqa: E402

lon_min, lon_max = -65., -55.
lat_min, lat_max = 33., 43.
time_min = numpy.datetime64('2012-10-01')
time_max = numpy.datetime64('2013-09-30')
dx = dy = 0.05
dt = numpy.timedelta64(1, 'D')
glon = numpy.arange(lon_min, lon_max + dx, dx)
glat = numpy.arange(lat_min, lat_max + dy, dy)
gtime = numpy.arange(time_min, time_max + dt, dt)
Lx = Ly = 1.
Lt = 7.
noise = 0.05
simu_start_date = '2012-10-01T00:00:00'

FOUR_NADIRS = [f"{CH}/dc_obs/2020a_SSH_mapping_NATL60_{m}.nc" for m in
               ("envisat", "geosat2", "topex-poseidon_interleaved", "jason1")]

def main():
    smoke = "--smoke" in sys.argv
    ds_oi_param = oi_param(Lx, Ly, Lt, noise)
    ds_oi_grid = oi_grid(glon, glat, gtime, simu_start_date)
    ds_obs = read_obs(FOUR_NADIRS, ds_oi_grid, ds_oi_param, simu_start_date, {'time': 5})
    print(f"obs after read_obs: {ds_obs.time.size} pts", flush=True)

    t0 = time.time()
    steps = range(1) if smoke else range(len(gtime))
    for it in steps:
        ts = time.time()
        oi_core(it, ds_oi_grid, ds_oi_param, ds_obs)
        if it % 20 == 0 or smoke:
            print(f"step {it}/{len(gtime)}: nobs={int(ds_oi_grid.nobs.values[it])} {time.time()-ts:.1f}s "
                  f"elapsed {time.time()-t0:.0f}s", flush=True)
    if smoke:
        print("SMOKE DONE", flush=True)
        return

    oi = numpy.asarray(ds_oi_grid.gssh.values, dtype=numpy.float32)
    numpy.savez_compressed(f"{WORK}/oi_year.npz",
                           oi=oi, glon=glon, glat=glat, gtime=gtime.astype("datetime64[D]"))
    print(f"saved oi {oi.shape} in {time.time()-t0:.0f}s total", flush=True)

if __name__ == "__main__":
    main()
