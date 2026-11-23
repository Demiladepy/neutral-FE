"""Tiny 1D nonlinear (total Lagrangian) finite-element solver for a hyperelastic bar.

Bar on X in [0, L], tapered cross-section A(X), clamped at X=0.
Linear 2-node elements, 1 Gauss point, stretch lam = 1 + du/dX (constant per element).
Equilibrium solved with Newton-Raphson under load stepping.

Load control:
  - "force":        tip force F applied at X=L
  - "displacement": tip displacement u_L prescribed, reaction force reported
"""
import numpy as np


class Bar1D:
    def __init__(self, L=1.0, n_el=40, A0=1.0, taper=0.5):
        self.L, self.n_el = L, n_el
        self.X = np.linspace(0.0, L, n_el + 1)
        self.h = np.diff(self.X)
        Xm = 0.5 * (self.X[:-1] + self.X[1:])          # element midpoints (Gauss point)
        self.Xm = Xm
        self.A = A0 * (1.0 - taper * Xm / L)            # thinner towards the loaded end

    def stretch(self, u):
        return 1.0 + np.diff(u) / self.h

    def assemble(self, u, mat):
        """Internal force vector and consistent tangent stiffness."""
        lam = self.stretch(u)
        P, dP = mat.P(lam), mat.dP(lam)
        n = self.n_el + 1
        f = np.zeros(n)
        K = np.zeros((n, n))
        fe = self.A * P                                   # element axial force
        ke = self.A * dP / self.h                         # element stiffness
        for e in range(self.n_el):
            i, j = e, e + 1
            f[i] -= fe[e]; f[j] += fe[e]
            K[i, i] += ke[e]; K[j, j] += ke[e]
            K[i, j] -= ke[e]; K[j, i] -= ke[e]
        return f, K, lam, P

    def solve(self, mat, load, control="force", n_steps=20, tol=1e-10, max_iter=30,
              verbose=False):
        """Returns dict with u, lam, P, reaction, converged, history (per load step)."""
        n = self.n_el + 1
        u = np.zeros(n)
        hist = []
        for step in range(1, n_steps + 1):
            t = step / n_steps
            if control == "force":
                free = np.arange(1, n)
                f_ext = np.zeros(n); f_ext[-1] = t * load
            else:
                free = np.arange(1, n - 1)
                f_ext = np.zeros(n)
                # predictor: spread the displacement increment linearly along the bar
                # (putting it all into the last element makes the first Newton step overshoot)
                u += (t * load - u[-1]) * self.X / self.L
            res_norms = []
            ok = False
            for it in range(max_iter):
                f_int, K, lam, P = self.assemble(u, mat)
                if np.any(lam <= 0) or not np.all(np.isfinite(f_int)):
                    break
                R = f_int[free] - f_ext[free]
                scale = max(1.0, abs(f_int[-1]))
                res_norms.append(np.linalg.norm(R) / scale)
                if res_norms[-1] < tol:
                    ok = True
                    break
                Kff = K[np.ix_(free, free)]
                try:
                    du = np.linalg.solve(Kff, -R)
                except np.linalg.LinAlgError:
                    break
                u[free] += du
            if verbose:
                print(f"step {step:2d}  t={t:.2f}  iters={len(res_norms)}  "
                      + "  ".join(f"{r:.1e}" for r in res_norms))
            if not ok:
                return dict(u=u, lam=self.stretch(u), converged=False, failed_at=t,
                            history=hist, last_residuals=res_norms)
            f_int, _, lam, P = self.assemble(u, mat)
            hist.append(dict(t=t, u_tip=u[-1], reaction=f_int[-1], max_lam=lam.max(),
                             iters=len(res_norms), residuals=res_norms))
        return dict(u=u.copy(), lam=lam, P=P, reaction=f_int[-1], converged=True,
                    history=hist)
