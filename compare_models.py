"""Part 2: does building physics into the network fix the extrapolation failures?

Two ground-truth materials x two network types x a 5-seed ensemble each.
  materials : Neo-Hookean, Ogden alpha=4 (stiffens much harder; not in the convex net's feature span)
  networks  : EnergyNet (plain MLP energy)  vs  ConvexEnergyNet (ICNN: convex + W->inf as lam->0)
Same FE bar, same load cases as run_demo.py. Writes results_v2.png, metrics_v2.json, prints a table.

    python compare_models.py
"""
import json
import warnings
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from materials import NeoHookean, Ogden, EnergyNet, ConvexEnergyNet
from fe_solver import Bar1D
from train import make_data, fit

warnings.filterwarnings("ignore", category=RuntimeWarning)
SEEDS = range(5)
LAM_TRAIN = (0.8, 1.8)
U_IN = 0.45
U_SWEEP = np.linspace(-0.3, 3.0, 34)
bar = Bar1D(L=1.0, n_el=40, taper=0.5)
MATERIALS = [NeoHookean(), Ogden(alpha=4.0)]
MODELS = [("plain MLP", EnergyNet), ("convex ICNN", ConvexEnergyNet)]


def sweep(mat):
    R, lmax = [], []
    for uL in U_SWEEP:
        r = bar.solve(mat, uL, control="displacement", n_steps=20)
        R.append(r["reaction"] if r["converged"] else np.nan)
        lmax.append(r["lam"].max() if r["converged"] else np.nan)
    return np.array(R), np.array(lmax)


results, curves = {}, {}
for mat in MATERIALS:
    lam_d, P_d = make_data(mat, LAM_TRAIN)
    ref = bar.solve(mat, U_IN, control="displacement", n_steps=10)
    R_true, lmax_true = sweep(mat)
    F_test = R_true[np.argmin(abs(U_SWEEP - 2.0))]      # force that needs u_L = 2 with the true law
    ref_F = bar.solve(mat, F_test, control="force", n_steps=20)
    curves[mat.name] = dict(lam_d=lam_d, P_d=P_d, R_true=R_true, lmax_true=lmax_true, models={})
    for mname, cls in MODELS:
        nets, rows, Rs = [], [], []
        for s in SEEDS:
            torch.manual_seed(s)
            net = cls()
            rmse = fit(net, lam_d, P_d)
            nets.append(net)
            r = bar.solve(net, U_IN, control="displacement", n_steps=10)
            e_u = np.linalg.norm(r["u"] - ref["u"]) / np.linalg.norm(ref["u"])
            rF = bar.solve(net, F_test, control="force", n_steps=20)
            e_uF = (abs(rF["u"][-1] - ref_F["u"][-1]) / abs(ref_F["u"][-1])
                    if rF["converged"] else np.nan)
            g = np.linspace(0.05, 50, 5000)
            R_nn, _ = sweep(net)
            Rs.append(R_nn)
            rows.append(dict(seed=s, rmse=rmse, err_u_in=e_u, force_conv=rF["converged"],
                             err_tip_force=e_uF, min_dP=float(net.dP(g).min()),
                             P_at_0p1=float(net.P(np.array([0.1]))[0])))
            print(f"  {mat.name:22s} {mname:12s} seed {s}: rmse {rmse:.4f}  in-range u err {e_u:.2e}  "
                  f"F={F_test:.2f} {'ok' if rF['converged'] else 'FAILED'}  min dP {rows[-1]['min_dP']:.3f}")
        Rs = np.array(Rs)
        curves[mat.name]["models"][mname] = dict(Rs=Rs, nets=nets)
        rel = np.abs(Rs - R_true) / np.abs(R_true)
        i2, i3 = np.argmin(abs(U_SWEEP - 2.0)), np.argmin(abs(U_SWEEP - 3.0))
        results[f"{mat.name} | {mname}"] = dict(
            err_u_in_mean=float(np.mean([r["err_u_in"] for r in rows])),
            err_R_uL2_mean=float(np.nanmean(rel[:, i2])), err_R_uL3_mean=float(np.nanmean(rel[:, i3])),
            spread_R_uL3=float(np.nanstd(Rs[:, i3]) / abs(R_true[i3])),
            force_control_converged=f"{sum(r['force_conv'] for r in rows)}/{len(rows)}",
            err_tip_force_mean=float(np.nanmean([r["err_tip_force"] for r in rows])),
            min_dP_worst=float(min(r["min_dP"] for r in rows)),
            P_at_0p1_mean=float(np.mean([r["P_at_0p1"] for r in rows])), P_at_0p1_true=float(mat.P(0.1)),
            lam_max_true_uL3=float(lmax_true[i3]), F_test=float(F_test))

print("\n" + f"{'case':36s} {'in-range u':>10s} {'R@uL=2':>8s} {'R@uL=3':>8s} {'spread@3':>9s} "
      f"{'F-ctrl':>6s} {'tip err':>8s} {'min dP':>7s} {'P(0.1)':>14s}")
for k, v in results.items():
    print(f"{k:36s} {v['err_u_in_mean']:10.2e} {v['err_R_uL2_mean']:8.1%} {v['err_R_uL3_mean']:8.1%} "
          f"{v['spread_R_uL3']:9.1%} {v['force_control_converged']:>6s} {v['err_tip_force_mean']:8.1%} "
          f"{v['min_dP_worst']:7.3f} {v['P_at_0p1_mean']:6.1f} ({v['P_at_0p1_true']:.0f})")
json.dump(results, open("metrics_v2.json", "w"), indent=1)

# ------------------------------------------------------------------ figure
C_TRUE, C_MLP, C_ICNN = "#0b0b0b", "#eb6834", "#2a78d6"
COL = {"plain MLP": C_MLP, "convex ICNN": C_ICNN}
INK2, GRID, BG = "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "lines.linewidth": 2})
fig, ax = plt.subplots(2, 3, figsize=(14, 8), facecolor=BG)
lam_g = np.linspace(0.3, 5.0, 300)
for row, mat in enumerate(MATERIALS):
    c = curves[mat.name]
    # (1) material law with ensemble band
    a = ax[row, 0]
    a.axvspan(*LAM_TRAIN, color="#d9d8d3", alpha=0.5, lw=0, label="training range")
    a.plot(lam_g, mat.P(lam_g), color=C_TRUE, lw=1.5, label="truth")
    for mname, d in c["models"].items():
        Ps = np.array([n.P(lam_g) for n in d["nets"]])
        a.fill_between(lam_g, Ps.min(0), Ps.max(0), color=COL[mname], alpha=0.2, lw=0)
        a.plot(lam_g, Ps.mean(0), color=COL[mname], ls="--", label=f"{mname} (5-seed band)")
    a.scatter(c["lam_d"], c["P_d"], s=6, color=INK2, zorder=3)
    ymax = 1.3 * mat.P(5.0) if mat.P(5.0) < 30 else 40
    a.set(xlabel="stretch λ", ylabel="P / μ", ylim=(-8, ymax), facecolor=BG,
          title=f"{mat.name.replace(' (true)', '')}: learned stress–stretch")
    a.legend(loc="upper left", fontsize=8)
    # (2) force-displacement with ensemble band
    a = ax[row, 1]
    a.plot(U_SWEEP, c["R_true"], color=C_TRUE, lw=1.5, label="truth")
    for mname, d in c["models"].items():
        a.fill_between(U_SWEEP, np.nanmin(d["Rs"], 0), np.nanmax(d["Rs"], 0), color=COL[mname],
                       alpha=0.2, lw=0)
        a.plot(U_SWEEP, np.nanmean(d["Rs"], 0), color=COL[mname], ls="--", label=mname)
    a.set(xlabel="tip displacement u_L / L", ylabel="reaction F / (μ A0)", facecolor=BG,
          title="FE force–displacement (bands = 5 seeds)")
    a.legend(loc="upper left", fontsize=8)
    # (3) is ensemble spread a usable warning? error vs spread along the sweep
    a = ax[row, 2]
    for mname, d in c["models"].items():
        err = np.abs(np.nanmean(d["Rs"], 0) - c["R_true"]) / np.abs(c["R_true"])
        spr = np.nanstd(d["Rs"], 0) / np.abs(c["R_true"])
        a.plot(c["lmax_true"], err, color=COL[mname], label=f"{mname}: actual error")
        a.plot(c["lmax_true"], spr, color=COL[mname], ls=":", label=f"{mname}: seed spread")
    a.axvline(LAM_TRAIN[1], color=INK2, lw=0.8, ls=":")
    a.set(xlabel="max stretch in the bar (true solution)", ylabel="relative to true reaction",
          yscale="log", ylim=(1e-4, 300), facecolor=BG, title="Does seed spread warn about the error?")
    a.legend(loc="upper center", fontsize=7.5, ncol=2)
fig.tight_layout()
fig.savefig("results_v2.png", dpi=150, facecolor=BG)
print("wrote results_v2.png, metrics_v2.json")
