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
