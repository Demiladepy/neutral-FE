"""End-to-end demo: true law -> noisy 'lab' data -> neural energy -> FE solve -> compare.

    python run_demo.py            # writes results.png and prints the error numbers
"""
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from materials import NeoHookean, EnergyNet
from fe_solver import Bar1D

SEED = 0
LAM_TRAIN = (0.8, 1.8)        # stretch range covered by the "lab test"
N_DATA = 60
NOISE = 0.02                  # std of additive stress noise (mu = 1)
U_IN = 0.45                   # tip displacement of the in-range load case
U_SWEEP = np.linspace(-0.3, 3.0, 67)   # tip displacements for the extrapolation sweep

rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
true = NeoHookean(mu=1.0)
bar = Bar1D(L=1.0, n_el=40, taper=0.5)

# ---------------------------------------------------------------- 1. ground truth
ref = bar.solve(true, U_IN, control="displacement", n_steps=10)
print(f"[truth] in-range case u_L={U_IN}: stretch {ref['lam'].min():.3f}..{ref['lam'].max():.3f}, "
      f"reaction {ref['reaction']:.4f}, Newton iters/step {[h['iters'] for h in ref['history']]}")

# ---------------------------------------------------------------- 2. synthetic lab data
lam_d = rng.uniform(*LAM_TRAIN, N_DATA)
P_d = true.P(lam_d) + NOISE * rng.standard_normal(N_DATA)

# ---------------------------------------------------------------- 3. learn W(lam)
net = EnergyNet(width=16, depth=2)
lam_t = torch.tensor(lam_d)
P_t = torch.tensor(P_d)


def loss_fn():
    P_pred, _ = net.stress(lam_t)               # P = dW/dlam (autograd, differentiable)
    return torch.mean((P_pred - P_t) ** 2)


opt = torch.optim.Adam(net.parameters(), lr=1e-2)
for ep in range(3000):
    opt.zero_grad(); l = loss_fn(); l.backward(); opt.step()
lbfgs = torch.optim.LBFGS(net.parameters(), max_iter=500, line_search_fn="strong_wolfe")
def closure():
    lbfgs.zero_grad(); l = loss_fn(); l.backward(); return l
lbfgs.step(closure)
train_rmse = float(loss_fn().detach().sqrt())
print(f"[train] stress RMSE on noisy data = {train_rmse:.4f}  (noise std = {NOISE})")

lam_grid = np.linspace(0.3, 4.5, 400)
inr = (lam_grid >= LAM_TRAIN[0]) & (lam_grid <= LAM_TRAIN[1])
P_err_in = np.abs(net.P(lam_grid[inr]) - true.P(lam_grid[inr])).max()
dP_learned = net.dP(lam_grid)
print(f"[train] max |P_nn - P_true| inside training range = {P_err_in:.4f}")
print(f"[train] learned tangent dP/dlam min over [0.3,4.5] = {dP_learned.min():.3f} "
      f"at lam={lam_grid[dP_learned.argmin()]:.2f} (true min = {true.dP(lam_grid).min():.3f})")

# ---------------------------------------------------------------- 4. same load case, NN law
nn = bar.solve(net, U_IN, control="displacement", n_steps=10)
print(f"[nn]    Newton iters/step {[h['iters'] for h in nn['history']]}")

# ---------------------------------------------------------------- 5. compare
def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))

err_u = rel_l2(nn["u"], ref["u"])
err_P = rel_l2(nn["P"], ref["P"])
err_R = abs(nn["reaction"] - ref["reaction"]) / abs(ref["reaction"])
print(f"[compare in-range] rel. L2 error: displacement {err_u:.2e}, stress {err_P:.2e}, "
      f"reaction {err_R:.2e}")

# force-controlled version of the same case (the NN must now find the displacement)
F_in = ref["reaction"]
ref_f = bar.solve(true, F_in, control="force", n_steps=10)
nn_f = bar.solve(net, F_in, control="force", n_steps=10)
err_u_f = rel_l2(nn_f["u"], ref_f["u"])
print(f"[compare in-range, force control F={F_in:.3f}] rel. L2 displacement error {err_u_f:.2e}")

# extrapolation sweep (displacement control, each point solved from scratch)
sweep = []
for uL in U_SWEEP:
    r_t = bar.solve(true, uL, control="displacement", n_steps=20)
    r_n = bar.solve(net, uL, control="displacement", n_steps=20)
    sweep.append(dict(uL=uL, R_true=r_t["reaction"] if r_t["converged"] else np.nan,
                      R_nn=r_n["reaction"] if r_n["converged"] else np.nan,
                      lam_max=r_t["lam"].max(), lam_min=r_t["lam"].min()))
S = {k: np.array([s[k] for s in sweep]) for k in sweep[0]}
with np.errstate(divide="ignore", invalid="ignore"):
    rel = np.abs(S["R_nn"] - S["R_true"]) / np.abs(S["R_true"])
for uL_q in (-0.3, 0.45, 1.0, 2.0, 3.0):
    i = np.argmin(abs(S["uL"] - uL_q))
    print(f"[extrap] u_L={S['uL'][i]:.2f} max stretch {S['lam_max'][i]:.2f}: "
          f"min stretch {S['lam_min'][i]:.2f}: reaction true {S['R_true'][i]:.3f}  nn {S['R_nn'][i]:.3f}  rel err {rel[i]:.1%}")

# force control beyond training range: does Newton even converge?
F_big = 2.0
r_tb = bar.solve(true, F_big, control="force", n_steps=20)
r_nb = bar.solve(net, F_big, control="force", n_steps=20)
last_ok = r_nb["history"][-1] if r_nb["history"] else None
fail_msg = ("converged" if r_nb["converged"] else
            f"FAILED at {r_nb['failed_at']:.0%} of the load (F={r_nb['failed_at']*F_big:.2f}); "
            f"last converged F={last_ok['reaction']:.2f} with max stretch {last_ok['max_lam']:.2f} "
            f"(true law at that F: {bar.solve(true, last_ok['reaction'], 'force', 20)['lam'].max():.2f}); "
            f"last residuals {['%.1e' % x for x in r_nb['last_residuals'][-4:]]}")
print(f"[extrap force F={F_big}] true: converged, max stretch {r_tb['lam'].max():.2f}; NN: {fail_msg}")
P_10, P_inf = net.P(np.array([10.0, 1e4]))
print(f"[extrap] learned stress at lam=10: {P_10:.3f} (true {true.P(10.0):.3f}); "
      f"as lam->inf P_nn -> {P_inf:.3f} (softplus MLP becomes linear, so W ~ c*lam and P saturates). "
      f"")
lam_w = np.linspace(0.05, 50, 20000)
P_w, dP_w = net.P(lam_w), net.dP(lam_w)
neg = lam_w[dP_w < 0]
print(f"[extrap] learned P peaks at {P_w.max():.3f} (lam={lam_w[P_w.argmax()]:.1f}) -> the NN bar can never carry "
      f"more than A_min*P_max = {bar.A.min()*P_w.max():.3f}, so F={F_big} has NO equilibrium solution")
print(f"[extrap] learned tangent dP/dlam < 0 (W non-convex, spurious softening) for lam in "
      f"{'none' if neg.size == 0 else f'[{neg.min():.1f}, {neg.max():.1f}]'}; "
      f"compression: P_nn(0.05)={P_w[0]:.2f} vs true {true.P(0.05):.0f}")

# ---------------------------------------------------------------- figure
C_TRUE, C_NN = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "legend.frameon": False, "lines.linewidth": 2})
fig, ax = plt.subplots(2, 2, figsize=(10, 7.2), facecolor="#fcfcfb")
for a in ax.flat:
    a.set_facecolor("#fcfcfb")

a = ax[0, 0]
a.axvspan(*LAM_TRAIN, color="#d9d8d3", alpha=0.5, lw=0, label="training range")
a.plot(lam_grid, true.P(lam_grid), color=C_TRUE, label="true Neo-Hookean")
a.plot(lam_grid, net.P(lam_grid), color=C_NN, ls="--", label="NN: P = dW/dλ")
a.scatter(lam_d, P_d, s=10, color=INK2, zorder=3, label="noisy 'lab' data")
a.set(xlabel="stretch λ", ylabel="1st PK stress P / μ", ylim=(-3, 4.8),
      title="(a) Material law: learned inside the data, guessed outside")
a.legend(loc="upper left")

a = ax[0, 1]
a.plot(bar.X, ref["u"], color=C_TRUE, label="true law")
a.plot(bar.X, nn["u"], color=C_NN, ls="--", label="NN law")
a.plot(bar.X, ref_f["u"], color=C_TRUE, alpha=0.35)
a.plot(bar.X, nn_f["u"], color=C_NN, ls=":", label=f"NN law, force control (F={F_in:.2f})")
a.set(xlabel="position X / L", ylabel="displacement u / L",
      title=f"(b) Displacement, in-range case (rel. L2 err {err_u_f:.1e})")
a.legend(loc="upper left")

a = ax[1, 0]
a.step(bar.X, np.r_[ref["P"], ref["P"][-1]], where="post", color=C_TRUE, label="true law")
a.step(bar.X, np.r_[nn["P"], nn["P"][-1]], where="post", color=C_NN, ls="--", label="NN law")
a.set(xlabel="position X / L", ylabel="stress P / μ (per element)",
      title=f"(c) Stress along tapered bar (rel. L2 err {err_P:.1e})")
a.legend(loc="upper left")

a = ax[1, 1]
lam_edge = np.interp(LAM_TRAIN[1], S["lam_max"], S["uL"])
lam_edge_lo = np.interp(LAM_TRAIN[0], S["lam_min"], S["uL"])
a.axvspan(lam_edge_lo, lam_edge, color="#d9d8d3", alpha=0.5, lw=0,
          label="all stretches in training range")
a.plot(S["uL"], S["R_true"], color=C_TRUE, label="true law")
a.plot(S["uL"], S["R_nn"], color=C_NN, ls="--", label="NN law")
a.axhline(F_big, color=INK2, lw=0.8, ls=":")
a.text(0.6, F_big + 0.06, f"F = {F_big:g}: NN bar has no solution", ha="left", color=INK2, fontsize=8)
a.set(xlabel="prescribed tip displacement u_L / L", ylabel="reaction force F / (μ A0)",
      title=f"(d) Out of range: error {rel[-1]:.0%} at max stretch {S['lam_max'][-1]:.1f}")
a.legend(loc="lower right")

fig.tight_layout()
fig.savefig("results.png", dpi=160, facecolor=fig.get_facecolor())

json.dump(dict(train_rmse=train_rmse, P_err_in=float(P_err_in), err_u=err_u, err_P=err_P,
               err_R=float(err_R), err_u_force=err_u_f,
               extrap={f"{s['uL']:.2f}": dict(lam_max=float(s['lam_max']),
                       R_true=float(s['R_true']), R_nn=float(s['R_nn'])) for s in sweep},
               force_big=dict(F=F_big, nn_converged=r_nb["converged"],
                              nn_failed_at=r_nb.get("failed_at"))),
          open("metrics.json", "w"), indent=1)
print("wrote results.png, metrics.json")
