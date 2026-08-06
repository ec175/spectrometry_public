"""Measure the EFFECTIVE drag coefficient of a body held in this tunnel, in the configuration
the scene actually uses (blockage, slip walls, LES). Prints Cd_eff so CD_HINT can be set from a
measurement rather than a textbook value."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from wt import shapes
from wt.bodies import BodySystem, FreeBody, _ring, polygon_area
from wt.config import RenderConfig
from wt.gpu import asnumpy, xp
from wt.lbm import LBM

cfg = RenderConfig()
cfg.width, cfg.height, cfg.scale = 540, 960, 1.5
nx, ny, u0 = cfg.nx, cfg.ny, 0.09

for name, frac in (("circle", 0.16), ("circle", 0.095), ("square", 0.14), ("square", 0.088)):
    prof = shapes.PROFILES[name]()
    d = frac * ny
    lbm = LBM(nx, ny, u0=u0, re=2600.0, ref_len=d, csm=0.16)
    sysm = BodySystem(nx, ny)
    b = sysm.add(FreeBody(prof, d, x=0.55 * nx, y=0.5 * ny, name=name, density=2.0))
    lbm.set_solid(sysm.build_masks(0.0))
    for _ in range(4000):
        lbm.step()
    fxf, fyf = lbm.force_field()
    mask, x0, y0 = b.local_mask(nx, ny)
    sl = (slice(y0, y0 + mask.shape[0]), slice(x0, x0 + mask.shape[1]))
    gm = xp.asarray(_ring(mask).astype(np.float32))
    fx = float(asnumpy((fxf[sl] * gm).sum()))
    cd = fx / (0.5 * u0 ** 2 * d)
    print(f"{name:7s} d={d:5.1f} ({frac:.3f} of span)  Fx={fx:8.4f}  Cd_eff={cd:6.2f}  "
          f"max|u|={lbm.health():.4f}")
