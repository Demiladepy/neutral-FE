"""Shared data generation and training (stress-only loss: P_pred = dW/dlam vs noisy P)."""
import numpy as np
import torch


def make_data(mat, lam_range=(0.8, 1.8), n=60, noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    lam = rng.uniform(*lam_range, n)
    return lam, mat.P(lam) + noise * rng.standard_normal(n)


def fit(net, lam, P, adam_epochs=3000, lr=1e-2):
    lam_t, P_t = torch.tensor(lam), torch.tensor(P)

    def loss_fn():
        P_pred, _ = net.stress(lam_t)
        return torch.mean((P_pred - P_t) ** 2)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for _ in range(adam_epochs):
        opt.zero_grad(); l = loss_fn(); l.backward(); opt.step()
    lbfgs = torch.optim.LBFGS(net.parameters(), max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        lbfgs.zero_grad(); l = loss_fn(); l.backward(); return l
    lbfgs.step(closure)
    return float(loss_fn().detach().sqrt())
