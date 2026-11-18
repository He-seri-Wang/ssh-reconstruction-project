#!/usr/bin/env python3
"""Generate the repo's summary figures from results JSONs + saved prediction fields.

figs/scores_and_training.png   mu vs leaderboard anchors + L1 training curve
figs/advantage_snapshot.png    test day where the reconstruction beats OI by the
                               largest margin: truth / OI / ours + error maps

Usage: python scripts/make_figs.py
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
WORK = os.environ.get("SSH_WORK_DIR", "/scratch/zceehw4/ssh_pilot")
FIGS = os.path.join(os.path.dirname(HERE), "figs")
os.makedirs(FIGS, exist_ok=True)

# ---------------- fig 1: scores + training curve ----------------
def fig_scores():
    e1 = json.load(open(f"{RESULTS}/eval_L1.json"))
    e2 = json.load(open(f"{RESULTS}/eval_L2.json"))
    hist = json.load(open(f"{RESULTS}/train_L1.json"))["history"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    names = ["OI\n(leaderboard)", "ours L1\n(single obs set)", "ours L2\n(+ multi-obs fusion)",
             "4DVarNet v2022\n(leaderboard)"]
    mus = [0.83, e1["mu"], e2["mu"], 0.94]
    stds = [0.04, e1["mu_std"], e2["mu_std"], 0.01]
    cols = ["tab:gray", "tab:red", "tab:orange", "tab:blue"]
    ax = axes[0]
    ax.bar(names, mus, yerr=stds, capsize=4, color=cols)
    for i, m in enumerate(mus):
        ax.text(i, m + 0.005, f"{m:.2f}", ha="center", fontsize=10)
    ax.set_ylabel(r"$\mu_{RMS}$ (lower = better)")
    ax.set_ylim(0.75, 1.0)
    ax.set_title("2020a SSH challenge, official eval chain")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ep = [h["ep"] for h in hist]
    ax.plot(ep, [h["tr_loss"] for h in hist], label="train loss", color="tab:blue")
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss", color="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(ep, [h["val_mse_center"] for h in hist], label="val MSE (center)",
             color="tab:red")
    ax2.set_ylabel("val MSE (normalized)", color="tab:red")
    ax.set_title("L1 training (120 epochs, from scratch)")
    fig.tight_layout()
    fig.savefig(f"{FIGS}/scores_and_training.png", dpi=150, bbox_inches="tight")
    print("-> figs/scores_and_training.png")

# ---------------- fig 2: advantage snapshot ----------------
def fig_snapshot():
    pred = np.load(f"{WORK}/pred_L1.npz", allow_pickle=True)
    oi = np.load(f"{WORK}/oi_year.npz", allow_pickle=True)
    dat = np.load(f"{WORK}/ssh_pilot_data.npz", allow_pickle=True)

    gt, obs_mask, dates = dat["gt"], dat["mask"], dat["dates"]
    date_idx = {str(d)[:10]: i for i, d in enumerate(dates)}

    # OI lives on its own 201x201 grid -> interpolate to the 200x200 field grid
    oi_idx = {str(t)[:10]: i for i, t in enumerate(oi["gtime"])}
    latq, lonq = np.meshgrid(dat["lat"], dat["lon"], indexing="ij")
    oi_interp = np.empty((len(pred["dates"]), 200, 200))
    for k, d in enumerate(pred["dates"]):
        oi_day = oi["oi"][oi_idx[str(d)[:10]]][::-1, :]
        f_oi = RegularGridInterpolator((oi["glat"][::-1], oi["glon"]), oi_day,
                                       bounds_error=False, fill_value=np.nan)
        oi_interp[k] = f_oi(np.stack([latq, lonq], -1))

    # per-day RMSE gap (OI - ours); pick the day our reconstruction wins biggest
    gaps = []
    for k, d in enumerate(pred["dates"]):
        di = date_idx[str(d)[:10]]
        g = gt[di]
        v = np.isfinite(g) & np.isfinite(pred["pred"][k]) & np.isfinite(oi_interp[k])
        if v.sum() < 10000: continue
        r_oi = float(np.sqrt(np.mean((oi_interp[k][v] - g[v]) ** 2)))
        r_us = float(np.sqrt(np.mean((pred["pred"][k][v] - g[v]) ** 2)))
        gaps.append((r_oi - r_us, r_oi, r_us, di, k))
    gaps.sort(reverse=True)
    gain, r_oi, r_us, di, k = gaps[0]
    d = pred["dates"][k]
    print(f"advantage day: {str(d)[:10]} OI RMSE {r_oi*100:.2f} cm, ours {r_us*100:.2f} cm, gain {gain*100:.2f} cm")

    g, ours, o = gt[di], pred["pred"][k], oi_interp[k]
    vmax = float(np.nanpercentile(np.abs(g), 99))
    emax = float(np.nanpercentile(np.stack([np.abs(o - g), np.abs(ours - g)]), 99))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    for ax, field, title, cmap, vm in (
            (axes[0, 0], g, f"NATL60 truth\n{str(d)[:10]}", "RdBu_r", vmax),
            (axes[0, 1], o, f"OI  (RMSE {r_oi*100:.1f} cm)", "RdBu_r", vmax),
            (axes[0, 2], ours, f"ours L1  (RMSE {r_us*100:.1f} cm)", "RdBu_r", vmax),
            (axes[1, 1], np.abs(o - g), "|OI - truth|", "magma", emax),
            (axes[1, 2], np.abs(ours - g), "|ours - truth|", "magma", emax)):
        im = ax.imshow(field, cmap=cmap, vmin=-vm if cmap == "RdBu_r" else 0, vmax=vm)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8)
    axes[1, 0].imshow(obs_mask[di], cmap="gray")
    axes[1, 0].set_title("observation mask (4 nadirs)")
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Largest OI-beating margin day: gain {gain*100:.1f} cm RMSE", y=0.995)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/advantage_snapshot.png", dpi=150, bbox_inches="tight")
    print("-> figs/advantage_snapshot.png")

if __name__ == "__main__":
    fig_scores()
    fig_snapshot()
