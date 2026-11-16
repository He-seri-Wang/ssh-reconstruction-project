#!/usr/bin/env python3
"""SSH pilot L1: 4DVarNet v2022 (augmented-state GF solver) trained from scratch
on local 2020a challenge data, per the published recipe.

Recipe provenance (4dvarnet-core @ CIA-Oceanix + GMD 2023 16.2119):
- split (official): train 2013-02-04..09-30 / val 2013-01-01..02-02 / test 2012-10-22..12-02
- windows dT=5 stride 1d, whole 200x200 gf domain (dl_gf2), patch_weight = constant crop time 2
  -> loss on center day only
- Solver_Grad_4DVarNN + Phi_r(DimAE=50,dW=3,dW2=1,sS=4,nbBlocks=1, dropout_phi 0.0) +
  model_GradUpdateLSTM(dim_grad_solver=150, dropout 0.25); aug_state 3x dT channels (v2022)
- Adam grouped lr (phi_r group 0.5x), schedule dec_lr_ngrad5_15_nocas:
  epochs [0,20,40,60,100,150] -> n_grad [5,5,10,10,15,15], lr [1e-3,1e-4,1e-3,1e-4,1e-4,1e-5]
- loss = 10*mse + 1*grad_mse + 0.25*(AE+AE_GT) + 0.5*LR + 0.5*SR
  (lit_model_augstate.compute_loss, alphas from qfebvre base_xp)
- normalization: train-GT mean/std (dataloaderading.mean_stds convention)

Usage: python train_L1.py [--smoke] [--epochs N] [--gpu 2] [--tag L1]
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


def load_data():
    d = np.load(DATA, allow_pickle=True)
    oi_d = np.load(OI)
    # regrid OI (201x201 quickstart grid) onto ref-cell centers (200x200)
    from scipy.interpolate import RegularGridInterpolator
    LAT, LON = np.meshgrid(d["lat"], d["lon"], indexing="ij")
    pts = np.stack([LAT.ravel(), LON.ravel()], axis=-1)
    oi = np.empty((len(oi_d["gtime"]), 200, 200), dtype=np.float32)
    for k in range(len(oi_d["gtime"])):
        itp = RegularGridInterpolator((oi_d["glat"], oi_d["glon"]), oi_d["oi"][k],
                                      method="linear", bounds_error=False, fill_value=None)
        oi[k] = itp(pts).reshape(200, 200).astype(np.float32)
    return d["gt"], oi, d["obs"], d["mask"], d["dates"]


def day_index(dates, a, b):
    return np.where((dates >= a) & (dates <= b))[0]


class Pilot4DVarNet:
    """Faithful port of lit_model_augstate.compute_loss (aug_state, no sst)."""

    def __init__(self, mean, std, device):
        self.mean, self.std, self.device = mean, std, device
        self.model = NN_4DVar.Solver_Grad_4DVarNN(
            Phi_r(SHAPE_STATE[0], DimAE=50, dw=3, dw2=1, ss=4, nb_blocks=1,
                  rateDr=0.0, stochastic=False),
            Model_H(SHAPE_STATE[0]),
            NN_4DVar.model_GradUpdateLSTM(SHAPE_STATE, False, 150, 0.25),
            'l2', 'l2', SHAPE_STATE, 5,
        ).to(device)
        self.model_LR = ModelLR().to(device)
        w = np.zeros((DT, 200, 200), dtype=np.float32)   # constant crop time 2 (dl_gf2)
        w[DT // 2] = 1.0
        self.patch_weight = torch.from_numpy(w).to(device)
        self.grad_crop = lambda t: t[..., 1:-1, 1:-1]
        self.gradient_img = lambda t: torch.unbind(
            self.grad_crop(2. * kornia.filters.spatial_gradient(t, normalized=True)), 2)

    def wloss(self, residual, weight):
        return NN_4DVar.compute_spatio_temp_weighted_loss(residual, weight)

    def compute_loss(self, batch):
        targets_OI, inputs_Mask, inputs_obs, targets_GT = batch
        if inputs_Mask.sum().item() == 0:
            z = torch.zeros_like(targets_GT)
            return (None, z, dict(mse=torch.tensor(0.), mseGrad=torch.tensor(0.)))
        targets_GT_wo_nan = targets_GT.where(~targets_GT.isnan(), targets_OI)

        # v2022 augmented state: [OI, m*(obs-OI), m*(obs-OI)] over the time dim
        obs_incr = inputs_Mask * (inputs_obs - targets_OI)
        init_state = torch.cat((targets_OI, obs_incr, obs_incr), dim=1)
        obs = torch.cat((targets_OI, obs_incr, 0. * targets_OI), dim=1)
        new_masks = torch.cat((torch.ones_like(inputs_Mask), inputs_Mask,
                               torch.zeros_like(inputs_Mask)), dim=1)

        # (verbatim from lit_model_augstate): the variational loop needs graph
        # building even under no_grad (val/test) — autograd.grad inside solver
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
                              if "--epochs" in sys.argv else 160)
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "L1"
    gpu = int(sys.argv[sys.argv.index("--gpu") + 1]) if "--gpu" in sys.argv else 2
    device = torch.device(f"cuda:{gpu}")
    torch.manual_seed(0)

    gt, oi, obs, mask, dates = load_data()
    tr_i = day_index(dates, *TRAIN_SLICE)
    va_i = day_index(dates, *VAL_SLICE)
    te_i = day_index(dates, *TEST_SLICE)
    print(f"loaded gt{gt.shape} oi{oi.shape} | train {len(tr_i)}d val {len(va_i)}d test {len(te_i)}d", flush=True)

    mean = float(np.nanmean(gt[tr_i]))
    std = float(np.nanstd(gt[tr_i]))
    P = Pilot4DVarNet(mean, std, device)

    def norm_split(arr_i):
        o = ((oi[arr_i] - mean) / std).astype(np.float32)
        ob = np.where(mask[arr_i], (obs[arr_i] - mean) / std, 0.0).astype(np.float32)
        g = ((gt[arr_i] - mean) / std).astype(np.float32)
        return o, ob, mask[arr_i].astype(np.float32), g

    def make_windows(o, ob, m, g):
        xs = []
        for s in range(len(o) - DT + 1):
            xs.append((torch.from_numpy(o[s:s + DT]),
                       torch.from_numpy(m[s:s + DT]),
                       torch.from_numpy(ob[s:s + DT]),
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
    ], lr=1e-3)

    log = {"tag": tag, "recipe": "augstate_v2022_gf", "mean": mean, "std": std,
           "train_days": len(tr_i), "epochs": epochs, "history": []}
    best_val = float("inf")
    start_ep = 0
    if "--resume" in sys.argv and os.path.exists(f"{OUTDIR}/ckpt_{tag}.pt"):
        ck = torch.load(f"{OUTDIR}/ckpt_{tag}.pt", map_location="cpu")
        P.model.load_state_dict(ck["model"])
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
        P.model.train()
        tr_losses = []
        for batch in batches(tr_w, shuffle=True):
            opt.zero_grad()
            loss, _, _ = P.compute_loss(batch)
            if loss is None:
                continue
            loss.backward()
            opt.step()
            tr_losses.append(float(loss))
        P.model.eval()
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
            torch.save({"model": P.model.state_dict(), "mean": mean, "std": std, "ep": ep,
                        "best_val": best_val},
                       f"{OUTDIR}/ckpt_{tag}.pt")
            marker = " *"
        print(f"ep {ep:3d} ngrad {n_grad:2d} lr {lr:.0e} tr {np.mean(tr_losses):.4f} "
              f"val {vl:.4f}{marker} [{time.time()-t00:.0f}s]", flush=True)
        json.dump(log, open(f"{OUTDIR}/train_{tag}.json", "w"), indent=1)
    print(f"TRAIN DONE best val {best_val:.4f} in {time.time()-t00:.0f}s", flush=True)

    # ---- test-window inference: sliding windows, center day --------------------
    ck = torch.load(f"{OUTDIR}/ckpt_{tag}.pt", map_location=device)
    P.model.load_state_dict(ck["model"])
    P.model.n_grad = NB_GRAD[max(i for i, e in enumerate(ITER_UPDATE) if e <= epochs - 1)]
    P.model.eval()
    o, ob, m, g = norm_split(te_i)
    pred = np.full((len(te_i), 200, 200), np.nan, dtype=np.float32)
    with torch.no_grad():
        for s in range(len(te_i) - DT + 1):
            batch = tuple(torch.from_numpy(a[s:s + DT])[None].float().to(device)
                          for a in (o, m, ob, g))
            _, out, _ = P.compute_loss(batch)
            pred[s + c] = out[0, c].cpu().numpy() * std + mean
    d = np.load(DATA, allow_pickle=True)
    np.savez_compressed(f"{OUTDIR}/pred_{tag}.npz", pred=pred,
                        lat=d["lat"], lon=d["lon"], dates=d["dates"][te_i])
    print(f"saved pred_{tag}.npz", flush=True)


if __name__ == "__main__":
    main()
