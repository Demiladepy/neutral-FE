"""Material laws for a 1D (uniaxial) bar.

Every material exposes two vectorised functions of the stretch lambda:
    P(lam)   first Piola-Kirchhoff stress  = dW/dlam
    dP(lam)  tangent stiffness             = d2W/dlam2   (needed by Newton)
"""
import numpy as np
import torch

torch.set_default_dtype(torch.float64)


class NeoHookean:
    """Incompressible Neo-Hookean in uniaxial tension/compression.

    W(lam) = mu/2 (lam^2 + 2/lam - 3)      (I1 = lam^2 + 2/lam for incompressible uniaxial)
    P(lam) = mu (lam - 1/lam^2)
    """
    name = "Neo-Hookean (true)"

    def __init__(self, mu=1.0):
        self.mu = mu

    def W(self, lam):
        return 0.5 * self.mu * (lam**2 + 2.0 / lam - 3.0)

    def P(self, lam):
        return self.mu * (lam - 1.0 / lam**2)

    def dP(self, lam):
        return self.mu * (1.0 + 2.0 / lam**3)


class EnergyNet(torch.nn.Module):
    """Neural strain energy W_theta(lam).

    Raw MLP N(lam) is normalised so the reference state is stress- and energy-free
    by construction (no data needed for that):
        W(lam) = N(lam) - N(1) - N'(1) (lam - 1)
    Stress and tangent come from autograd: P = dW/dlam, dP = d2W/dlam2.
    """
    name = "Neural energy (learned)"

    def __init__(self, width=16, depth=2):
        super().__init__()
        layers, n_in = [], 1
        for _ in range(depth):
            layers += [torch.nn.Linear(n_in, width), torch.nn.Softplus()]
            n_in = width
        layers += [torch.nn.Linear(n_in, 1)]
        self.mlp = torch.nn.Sequential(*layers)

    def _N(self, lam):
        return self.mlp(lam.reshape(-1, 1)).reshape(-1)

    def energy(self, lam):
        one = torch.ones(1, requires_grad=True)
        N1 = self._N(one)
        dN1, = torch.autograd.grad(N1.sum(), one, create_graph=True)
        return self._N(lam) - N1 - dN1 * (lam - 1.0)

    def stress(self, lam, create_graph=True):
        """P = dW/dlam via autograd. lam: 1D tensor."""
        if not lam.requires_grad:
            lam = lam.clone().requires_grad_(True)
        W = self.energy(lam)
        P, = torch.autograd.grad(W.sum(), lam, create_graph=create_graph)
        return P, lam

    # --- numpy interface used by the FE solver --------------------------------
    def W(self, lam_np):
        lam = torch.as_tensor(np.atleast_1d(lam_np), dtype=torch.float64).requires_grad_(True)
        return self.energy(lam).detach().numpy()

    def P(self, lam_np):
        lam = torch.as_tensor(np.atleast_1d(lam_np), dtype=torch.float64)
        P, _ = self.stress(lam, create_graph=False)
        return P.detach().numpy()

    def dP(self, lam_np):
        lam = torch.as_tensor(np.atleast_1d(lam_np), dtype=torch.float64)
        P, lam = self.stress(lam, create_graph=True)
        dP, = torch.autograd.grad(P.sum(), lam)   # elementwise: d P_i / d lam_i
        return dP.detach().numpy()


class Ogden:
    """One-term incompressible Ogden in uniaxial loading (alpha=2 recovers Neo-Hookean).

    W = 2 mu / alpha^2 (lam^alpha + 2 lam^(-alpha/2) - 3)
    P = 2 mu / alpha   (lam^(alpha-1) - lam^(-alpha/2-1))
    alpha = 4 stiffens much faster than Neo-Hookean (P ~ lam^3), so it is NOT in the span of the
    constrained network's features -- an honest test of extrapolation.
    """

    def __init__(self, mu=1.0, alpha=4.0):
        self.mu, self.a = mu, alpha
        self.name = f"Ogden alpha={alpha:g} (true)"

    def W(self, lam):
        a = self.a
        return 2 * self.mu / a**2 * (lam**a + 2 * lam**(-a / 2) - 3)

    def P(self, lam):
        a = self.a
        return 2 * self.mu / a * (lam**(a - 1) - lam**(-a / 2 - 1))

    def dP(self, lam):
        a = self.a
        return 2 * self.mu / a * ((a - 1) * lam**(a - 2) + (a / 2 + 1) * lam**(-a / 2 - 2))


class ConvexEnergyNet(EnergyNet):
    """Physics-constrained neural energy: convex in lam and W -> inf as lam -> 0, by construction.

    Features  phi = [lam^2, 1/lam]  are convex in lam (for lam > 0).
    Input-convex NN (ICNN): every weight acting on phi or on a hidden layer is kept >= 0
    (softplus of a raw parameter), activations are convex and non-decreasing (softplus).
    Convex non-decreasing function of convex functions + affine terms in lam  =>  convex in lam.
    The same normalisation as EnergyNet subtracts an affine function, which keeps convexity, so
    W >= 0, W(1) = 0, P(1) = 0, dP/dlam >= 0 everywhere -- inside AND outside the data.
    """
    name = "Convex neural energy (ICNN)"

    def __init__(self, width=16, depth=2):
        torch.nn.Module.__init__(self)
        sp = torch.nn.functional.softplus
        self._sp = sp
        self.Wphi = torch.nn.ParameterList()   # >= 0, acts on [lam^2, 1/lam]
        self.Wlam = torch.nn.ParameterList()   # free, acts on lam (affine => sign-free)
        self.Wz = torch.nn.ParameterList()     # >= 0, acts on previous hidden layer
        self.b = torch.nn.ParameterList()
        for k in range(depth):
            self.Wphi.append(torch.nn.Parameter(0.5 * torch.randn(2, width) - 1.0))
            self.Wlam.append(torch.nn.Parameter(0.5 * torch.randn(1, width)))
            self.b.append(torch.nn.Parameter(torch.zeros(width)))
            if k > 0:
                self.Wz.append(torch.nn.Parameter(0.5 * torch.randn(width, width) - 1.0))
        self.w_out = torch.nn.Parameter(0.5 * torch.randn(width) - 1.0)

    def _N(self, lam):
        sp = self._sp
        x = lam.reshape(-1, 1)
        phi = torch.cat([x**2, 1.0 / x], dim=1)
        z = None
        for k in range(len(self.b)):
            pre = phi @ sp(self.Wphi[k]) + x @ self.Wlam[k] + self.b[k]
            if z is not None:
                pre = pre + z @ sp(self.Wz[k - 1])
            z = sp(pre)
        return z @ sp(self.w_out)
