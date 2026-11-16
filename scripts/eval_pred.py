#!/usr/bin/env python3
"""SSH pilot eval: score a prediction npz (1/20deg daily, test window) with the
challenge's own mod_eval on the official protocol — identical chain to the
validated ssh_oi_gate.py (scipy regrid shim for pyinterp, full box, ref daily resample).

Usage: python eval_pred.py [--tag L1]
Writes ssh_pilot/eval_{tag}.json with mu/mu_std/lx/lt vs leaderboard anchors.
"""
import os, sys, json, glob
import numpy
import xarray as xr

numpy.float = float
numpy.int = int

_orig_mfdataset = xr.open_mfdataset
def _mfdataset(*a, **k):
    pp = k.pop("preprocess", None)
    if pp is not None:
        k["preprocess"] = lambda ds: pp(ds).load()
    return _orig_mfdataset(*a, **{**k, "parallel": False})
xr.open_mfdataset = _mfdataset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CH = os.environ.get("SSH_CHALLENGE_DIR", "/scratch/zceehw4/ssh_challenge")
WORK = os.environ.get("SSH_WORK_DIR", "/scratch/zceehw4/ssh_pilot")
sys.path.insert(0, os.path.join(REPO, "third_party", "2020a_ssh_challenge"))
sys.path.insert(0, REPO)
from src.mod_eval import rmse_based_scores, psd_based_scores  # noqa: E402
from src.regrid import oi_regrid                               # noqa: E402

ANCHORS = {
    "leaderboard_4dvarnet_v2022_4nadir": {"mu": 0.94, "mu_std": 0.01, "lx": 0.83, "lt": 8.03},
    "leaderboard_oi_4nadir": {"mu": 0.83, "mu_std": 0.04, "lx": 2.25, "lt": 15.67},
}


def main():
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "L1"
    d = numpy.load(f"{WORK}/pred_{tag}.npz", allow_pickle=True)
    pred_da = xr.Dataset({"gssh": (("time", "lat", "lon"), d["pred"])},
                         coords={"time": d["dates"], "lat": d["lat"], "lon": d["lon"]})

    t0 = numpy.datetime64("2012-10-22")
    t1 = numpy.datetime64("2012-12-02")
    ref_files = sorted(glob.glob(f"{CH}/dc_ref/*.nc"))
    dc_ref = xr.open_mfdataset(ref_files, combine="nested", concat_dim="time")
    ref = dc_ref.sel(time=slice(t0, t1)).resample(time="1D").mean()
    dc_ref.close()
    print(f"ref: {ref.time.size} days {ref.lon.size}x{ref.lat.size}", flush=True)

    regrid = oi_regrid(pred_da, ref)
    _, _, mu, mu_std = rmse_based_scores(regrid, ref)
    psd_map, lx, lt = psd_based_scores(regrid, ref)
    out = {"tag": tag,
           "mu": round(float(mu), 3), "mu_std": round(float(mu_std), 3),
           "lx_deg": round(float(lx), 2), "lt_days": round(float(lt), 2),
           "anchors": ANCHORS,
           "mu_delta_vs_v2022": round(float(mu) - ANCHORS["leaderboard_4dvarnet_v2022_4nadir"]["mu"], 3)}
    os.makedirs(f"{REPO}/results", exist_ok=True)
    dest = f"{REPO}/results/eval_{tag}.json"
    json.dump(out, open(dest, "w"), indent=2)
    print(json.dumps(out, indent=1), flush=True)
    print("->", dest, flush=True)


if __name__ == "__main__":
    main()
