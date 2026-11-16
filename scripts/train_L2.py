#!/usr/bin/env python3
"""SSH pilot L2: identical v2022 recipe as train_L1.py, plus OUR multimodal
adaptive-fusion module at the observation-increment entry (T4 hook).

The 4 nadir missions (currently pooled into one obs field by the official
pipeline, and hand-weighted per mission in DUACS) enter as separate channels;
a small conv gate produces per-pixel softmax mission weights from local
[mask, increment] context; the fused increment then feeds the exact same
augmented-state v2022 solver, loss, schedule, and budget as L1.

kill gate: L2 mu >= L1 mu + 2*se (se from rmse_based_scores mu_std over daily
scores; final comparison in eval JSONs).

Usage: python train_L2.py [--smoke] [--epochs N] [--gpu 2] [--tag L2]
"""
import os, sys, json, time
import numpy as np
import torch
import kornia

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "third_party", "4dvarnet-core"))
import solver as NN_4DVar
from src.models_ssh import Phi_r, Model_H, ModelLR

WORK = os.environ.get("SSH_WORK_DIR", "/scratch/zceehw4/ssh_pilot")
DATA = os.environ.get("SSH_DATA_NPZ", f"{WORK}/ssh_pilot_data.npz")
MISSIONS = os.environ.get("SSH_MISSIONS_NPZ", f"{WORK}/ssh_pilot_data_missions.npz")
OI = os.environ.get("OI_NPZ", f"{WORK}/oi_year.npz")
OUTDIR = WORK
DT = 5
SHAPE_STATE = (3 * DT, 200, 200)
ITER_UPDATE = [0, 20, 40, 60, 100, 150]
NB_GRAD = [5, 5, 10, 10, 15, 15]
LR_UPD = [1e-3, 1e-4, 1e-3, 1e-4, 1e-4, 1e-5]
TRAIN_SLICE = (np.datetime64("2013-02-04"), np.datetime64("2013-09-30"))
VAL_SLICE = (np.datetime64("2013-01-01"), np.datetime64("2013-02-02"))
TEST_SLICE = (np.datetime64("2012-10-22"), np.datetime64("2012-12-02"))


class MissionFusion(torch.nn.Module):
    """Per-pixel adaptive mission weighting from local [mask, increment] context."""

    def __init__(self, n_missions=4, dT=DT, k=5):
        super().__init__()
        self.M, self.T = n_missions, dT
        self.conv = torch.nn.Conv2d(2 * n_missions * dT, n_missions * dT, k, padding=k // 2)

    def forward(self, ms, ds):
        # ms, ds: (B, M, T, H, W) -> fused increment & union mask: (B, T, H, W)
        B, M, T, H, W = ms.shape
        x = torch.cat([ms.reshape(B, M * T, H, W), ds.reshape(B, M * T, H, W)], dim=1)
        w = torch.softmax(self.conv(x).reshape(B, M, T, H, W), dim=1)
        union = ms.max(dim=1).values
        fused = (w * ds).sum(dim=1) * union
        return fused, union


def load_data():
    d = np.load(DATA, allow_pickle=True)
    dm = np.load(MISSIONS, allow_pickle=True)
    oi_d = np.load(OI)
    from scipy.interpolate import RegularGridInterpolator
    LAT, LON = np.meshgrid(d["lat"], d["lon"], indexing="ij")
    pts = np.stack([LAT.ravel(), LON.ravel()], axis=-1)
    oi = np.empty((len(oi_d["gtime"]), 200, 200), dtype=np.float32)
    for k in range(len(oi_d["gtime"])):
        itp = RegularGridInterpolator((oi_d["glat"], oi_d["glon"]), oi_d["oi"][k],
                                      method="linear", bounds_error=False, fill_value=None)
        oi[k] = itp(pts).reshape(200, 200).astype(np.float32)
    return d["gt"], oi, dm["obs_m"], dm["mask_m"], d["dates"]


def day_index(dates, a, b):
    return np.where((dates >= a) & (dates <= b))[0]


class Pilot4DVarNetFusion:
    """train_L1.Pilot4DVarNet + MissionFusion at the increment entry; everything
    else (solver, state, loss, schedule) verbatim identical to L1."""

    def __init__(self, mean, std, device):
        self.mean, self.std, self.device = mean, std, device
        self.model = NN_4DVar.Solver_Grad_4DVarNN(
            Phi_r(SHAPE_STATE[0], DimAE=50, dw=3, dw2=1, ss=4, nb_blocks=1,
                  rateDr=0.0, stochastic=False),
            Model_H(SHAPE_STATE[0]),
            NN_4DVar.model_GradUpdateLSTM(SHAPE_STATE, False, 150, 0.25),
            'l2', 'l2', SHAPE_STATE, 5,
        ).to(device)
        self.fusion = MissionFusion().to(device)
        self.model_LR = ModelLR().to(device)
        w = np.zeros((DT, 200, 200), dtype=np.float32)
        w[DT // 2] = 1.0
        self.patch_weight = torch.from_numpy(w).to(device)
        self.grad_crop = lambda t: t[..., 1:-1, 1:-1]
        self.gradient_img = lambda t: torch.unbind(
            self.grad_crop(2. * kornia.filters.spatial_gradient(t, normalized=True)), 2)

    def wloss(self, residual, weight):
        return NN_4DVar.compute_spatio_temp_weighted_loss(residual, weight)

    def compute_loss(self, batch):
        targets_OI, inputs_MaskM, inputs_obsM, targets_GT = batch   # mask/obs: (B,M,T,H,W)
        if inputs_MaskM.sum().item() == 0:
            z = torch.zeros_like(targets_GT)
            return (None, z, dict(mse=torch.tensor(0.), mseGrad=torch.tensor(0.)))
        targets_GT_wo_nan = targets_GT.where(~targets_GT.isnan(), targets_OI)

        # OUR MODULE: adaptive per-mission fusion of the increments
        per_mission_incr = inputs_MaskM * (inputs_obsM - targets_OI[:, None])
        obs_incr, inputs_Mask = self.fusion(inputs_MaskM, per_mission_incr)

        init_state = torch.cat((targets_OI, obs_incr, obs_incr), dim=1)
        obs = torch.cat((targets_OI, obs_incr, 0. * targets_OI), dim=1)
        new_masks = torch.cat((torch.ones_like(inputs_Mask), inputs_Mask,
                               torch.zeros_like(inputs_Mask)), dim=1)

        with torch.set_grad_enabled(True):
            state = torch.autograd.Variable(init_state, requires_grad=True)
            outputsSLRHR, _, _, _ = self.model(state, obs, new_masks)
        outputsSLR = outputsSLRHR[:, 0:DT]
        outputs = outputsSLR + outputsSLRHR[:, 2 * DT:]

        g_ox, g_oy = self.gradient_img(outputs)
        g_gx, g_gy = self.gradient_img(targets_GT_wo_nan)
        loss_All = self.wloss(outputs - targets_GT_wo_nan, self.patch_weight)
        gw = self.grad_crop(self.patch_weight)
        loss_GAll = self.wloss(g_ox - g_gx, gw) + self.wloss(g_oy - g_gy, gw)

        l_ae = torch.mean((self.model.phi_r(outputsSLRHR) - outputsSLRHR) ** 2)
        yGT = torch.cat((targets_OI, targets_GT_wo_nan - outputsSLR,
                         targets_GT_wo_nan - outputsSLR), dim=1)
        l_ae_gt = torch.mean((self.model.phi_r(yGT) - yGT) ** 2)
        l_sr = self.wloss(outputsSLR - targets_OI, self.patch_weight)
        l_lr = self.wloss(self.model_LR(outputs) - self.model_LR(targets_OI),
                          self.model_LR(self.patch_weight))

        loss = 10.0 * loss_All + 1.0 * loss_GAll
        loss += 0.25 * (l_ae + l_ae_gt) + 0.5 * l_lr + 0.5 * l_sr
        return loss, outputs, dict(mse=loss_All.detach(), mseGrad=loss_GAll.detach())


def main():
    smoke = "--smoke" in sys.argv
    epochs = 2 if smoke else (int(sys.argv[sys.argv.index("--epochs") + 1])
                              if "--epochs" in sys.argv else 120)
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "L2"
    gpu = int(sys.argv[sys.argv.index("--gpu") + 1]) if "--gpu" in sys.argv else 0
    device = torch.device(f"cuda:{gpu}")
    torch.manual_seed(0)

    gt, oi, obs_m, mask_m, dates = load_data()
    tr_i = day_index(dates, *TRAIN_SLICE)
    va_i = day_index(dates, *VAL_SLICE)
    te_i = day_index(dates, *TEST_SLICE)
    print(f"loaded gt{gt.shape} oi{oi.shape} obs_m{obs_m.shape} | "
          f"train {len(tr_i)}d val {len(va_i)}d test {len(te_i)}d", flush=True)

    mean = float(np.nanmean(gt[tr_i]))
    std = float(np.nanstd(gt[tr_i]))
    P = Pilot4DVarNetFusion(mean, std, device)

    def norm_split(arr_i):
        o = ((oi[arr_i] - mean) / std).astype(np.float32)
        ob = np.where(mask_m[:, arr_i], (obs_m[:, arr_i] - mean) / std, 0.0).astype(np.float32)
        m = mask_m[:, arr_i].astype(np.float32)
        g = ((gt[arr_i] - mean) / std).astype(np.float32)
        return o, m.transpose(1, 0, 2, 3), ob.transpose(1, 0, 2, 3), g

    def make_windows(o, m, ob, g):
        # mission arrays arrive day-major (len, M, H, W); windows need (M, DT, H, W)
        xs = []
        for s in range(len(o) - DT + 1):
            xs.append((torch.from_numpy(o[s:s + DT]),
                       torch.from_numpy(m[s:s + DT].transpose(1, 0, 2, 3)),
                       torch.from_numpy(ob[s:s + DT].transpose(1, 0, 2, 3)),
                       torch.from_numpy(g[s:s + DT])))
        return xs

    tr_w = make_windows(*norm_split(tr_i))
    va_w = make_windows(*norm_split(va_i))
    print(f"windows: train {len(tr_w)} val {len(va_w)}", flush=True)

    def batches(windows, bs=2, shuffle=False):
        order = np.random.permutation(len(windows)) if shuffle else np.arange(len(windows))
        for k in range(0, len(order), bs):
            sel = [windows[j] for j in order[k:k + bs]]
            yield tuple(torch.stack(t).to(device) for t in zip(*sel))

    opt = torch.optim.Adam([
        {'params': P.model.model_Grad.parameters()},
        {'params': P.model.model_VarCost.parameters()},
        {'params': P.model.model_H.parameters()},
        {'params': P.model.phi_r.parameters()},
        {'params': P.fusion.parameters()},          # our module, solver-lr group
    ], lr=1e-3)

    log = {"tag": tag, "recipe": "augstate_v2022_gf + mission_fusion", "mean": mean, "std": std,
           "train_days": len(tr_i), "epochs": epochs, "history": []}
    best_val = float("inf")
    start_ep = 0
    if "--resume" in sys.argv and os.path.exists(f"{OUTDIR}/ckpt_{tag}.pt"):
        ck = torch.load(f"{OUTDIR}/ckpt_{tag}.pt", map_location="cpu")
        P.model.load_state_dict(ck["model"])
        P.fusion.load_state_dict(ck["fusion"])
        start_ep = ck["ep"] + 1
        # ckpts saved before the best_val edit lack the key: recover it (and full
        # history) from train_{tag}.json so a worse post-resume epoch cannot
        # overwrite the true-best checkpoint
        best_val = ck.get("best_val")
        json_path = f"{OUTDIR}/train_{tag}.json"
        if os.path.exists(json_path):
            prev = json.load(open(json_path)).get("history", [])
            log["history"] = [h for h in prev if h["ep"] < start_ep]
            if best_val is None:
                best_val = min((h["val_mse_center"] for h in prev),
                               default=float("inf"))
        if best_val is None:
            best_val = float("inf")
        print(f"RESUME from ep {ck['ep']} (best_val {best_val:.4f})", flush=True)
    t00 = time.time()
    for ep in range(start_ep, epochs):
        sched = max(i for i, e in enumerate(ITER_UPDATE) if e <= ep)
        n_grad, lr = NB_GRAD[sched], LR_UPD[sched]
        P.model.n_grad = n_grad
        for gi, pg in enumerate(opt.param_groups):
            pg["lr"] = lr * (0.5 if gi == 3 else 1.0)
        P.model.train(); P.fusion.train()
        tr_losses = []
        for batch in batches(tr_w, shuffle=True):
            opt.zero_grad()
            loss, _, _ = P.compute_loss(batch)
            if loss is None:
                continue
            loss.backward()
            opt.step()
            tr_losses.append(float(loss))
        P.model.eval(); P.fusion.eval()
        c = DT // 2
        vals = []
        with torch.no_grad():
            for batch in batches(va_w):
                _, out, _ = P.compute_loss(batch)
                vals.append(float(torch.mean((out[:, c] - batch[3][:, c]) ** 2)))
        vl = float(np.mean(vals))
        log["history"].append({"ep": ep, "n_grad": n_grad, "lr": lr,
                               "tr_loss": float(np.mean(tr_losses)), "val_mse_center": vl})
        marker = ""
        if vl < best_val:
            best_val = vl
            torch.save({"model": P.model.state_dict(), "fusion": P.fusion.state_dict(),
                        "mean": mean, "std": std, "ep": ep, "best_val": best_val},
                       f"{OUTDIR}/ckpt_{tag}.pt")
            marker = " *"
        print(f"ep {ep:3d} ngrad {n_grad:2d} lr {lr:.0e} tr {np.mean(tr_losses):.4f} "
              f"val {vl:.4f}{marker} [{time.time()-t00:.0f}s]", flush=True)
        json.dump(log, open(f"{OUTDIR}/train_{tag}.json", "w"), indent=1)
    print(f"TRAIN DONE best val {best_val:.4f} in {time.time()-t00:.0f}s", flush=True)

    ck = torch.load(f"{OUTDIR}/ckpt_{tag}.pt", map_location=device)
    P.model.load_state_dict(ck["model"])
    P.fusion.load_state_dict(ck["fusion"])
    P.model.n_grad = NB_GRAD[max(i for i, e in enumerate(ITER_UPDATE) if e <= epochs - 1)]
    P.model.eval(); P.fusion.eval()
    o, m, ob, g = norm_split(te_i)
    pred = np.full((len(te_i), 200, 200), np.nan, dtype=np.float32)
    with torch.no_grad():
        for s in range(len(te_i) - DT + 1):
            batch = (torch.from_numpy(o[s:s + DT])[None].float().to(device),
                     torch.from_numpy(m[s:s + DT].transpose(1, 0, 2, 3))[None].float().to(device),
                     torch.from_numpy(ob[s:s + DT].transpose(1, 0, 2, 3))[None].float().to(device),
                     torch.from_numpy(g[s:s + DT])[None].float().to(device))
            _, out, _ = P.compute_loss(batch)
            pred[s + c] = out[0, c].cpu().numpy() * std + mean
    d = np.load(DATA, allow_pickle=True)
    np.savez_compressed(f"{OUTDIR}/pred_{tag}.npz", pred=pred,
                        lat=d["lat"], lon=d["lon"], dates=d["dates"][te_i])
    print(f"saved pred_{tag}.npz", flush=True)


if __name__ == "__main__":
    main()
