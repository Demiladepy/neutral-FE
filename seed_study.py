"""Is the extrapolation a property of the data or of the random init? Train 5 seeds, same data."""
import numpy as np, torch
from materials import EnergyNet, NeoHookean
t=NeoHookean(); rng=np.random.default_rng(0)
lam=rng.uniform(0.8,1.8,60); P=t.P(lam)+0.02*rng.standard_normal(60)
L=torch.tensor(lam); Pt=torch.tensor(P)
q=np.array([0.4,0.6,1.3,2.5,4.0])
print("lam ", q)
print("true", np.round(t.P(q),3))
for s in range(5):
    torch.manual_seed(s); net=EnergyNet()
    opt=torch.optim.Adam(net.parameters(),1e-2)
    for _ in range(3000):
        opt.zero_grad(); l=((net.stress(L)[0]-Pt)**2).mean(); l.backward(); opt.step()
    print("seed",s, np.round(net.P(q),3), "min dP", round(net.dP(np.linspace(0.2,6,300)).min(),3))
