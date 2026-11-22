# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy - https://github.com/ec175/spectrometry_public
"""harmonics.py - the real spherical harmonics, the WIREFRAME CURVE MESH, and the state table.

WHAT THE SOURCE ACTUALLY DOES (measured, `../../../tools/Video_Interpreter/downloads/orbital_src.mp4`)
------------------------------------------------------------------------------------------
716x1274 @30, 601 frames, 20.03 s. Its burnt-in readout shows NON-INTEGER `l` and `m`
(l=0.06, 0.56, 0.99, 1.00, 1.05, 1.54, 1.99, 2.27, 2.82 ... m=0.26, 0.82, 0.95, 1.05, 1.54)
which is the tell: it is not evaluating an associated Legendre function of continuous degree,
it is INTERPOLATING between integer |Y_lm| states and printing the interpolated pair. That is
what this module reproduces, and it is also the only version that is numerically safe -
`lpmv` of a non-integer degree is a hypergeometric evaluation with poles, and the picture
would be identical.

THE MESH IS CURVES, NOT A SURFACE SAMPLE. The source is plainly a lat/long wireframe: you can
count the meridians, and the bright knots are where a latitude circle collapses onto a NODAL
CONE (r -> 0 there, so the whole circle lands on one point). A uniform (theta, phi) scatter of
the same point count renders as fog and loses that entirely. So the mesh is two families of
dense polylines, and a curve's brightness comes for free from the additive accumulator: many
samples per pixel where the curve is slow or folded.

⚠ SCIPY ARGUMENT ORDER IS A SILENT FAILURE. `sph_harm(m, n, azimuth, polar)` (deprecated)
and `sph_harm_y(n, m, polar, azimuth)` (>=1.15) take their four arguments in DIFFERENT ORDERS
AND DIFFERENT ROLES. Getting it wrong does not raise - it renders a plausible, wrong orbital.
`_selftest()` at import asserts |Y_1,0| peaks at the POLE and |Y_1,1| peaks at the EQUATOR,
which is the cheapest statement that separates the two conventions.
"""
from __future__ import annotations

import numpy as np
from scipy import special as _sp

_HAS_Y = hasattr(_sp, "sph_harm_y")


def real_Y(l: int, m: int, theta, phi):
    """The REAL spherical harmonic Y_lm (cos-form for m>0), evaluated on (theta, phi).

    theta is the POLAR angle from +z (0..pi); phi is the AZIMUTH (0..2pi).
    Only m >= 0 is needed: the sin-form m<0 partners are the same shape rotated in phi, and
    the choreography never asks for one.
    """
    theta = np.asarray(theta, np.float64)
    phi = np.asarray(phi, np.float64)
    if _HAS_Y:
        Y = _sp.sph_harm_y(l, m, theta, phi)
    else:                                              # scipy < 1.15
        Y = _sp.sph_harm(m, l, phi, theta)
    if m == 0:
        return np.real(Y)
    return np.sqrt(2.0) * ((-1.0) ** m) * np.real(Y)


def _selftest():
    th = np.array([0.02, np.pi / 2])
    ph = np.array([0.0, 0.0])
    y10 = np.abs(real_Y(1, 0, th, ph))
    y11 = np.abs(real_Y(1, 1, th, ph))
    assert y10[0] > y10[1] * 10, f"Y(1,0) should peak at the POLE, got {y10}"
    assert y11[1] > y11[0] * 10, f"Y(1,1) should peak at the EQUATOR, got {y11}"


_selftest()


# --------------------------------------------------------------------------------------
# the state table - the choreography's vocabulary
# --------------------------------------------------------------------------------------
#: (l, m, name). Ends where it began, on `s`, which is also what stops this clip inheriting
#: the source's blown-out ending (its last seconds sprawl far outside the frame).
STATES = [
    (0, 0, "s"),
    (1, 0, "p_z"),
    (1, 1, "p_x"),
    (2, 0, "d_z2"),
    (2, 1, "d_xz"),
    (2, 2, "d_xy"),
    (3, 0, "f_z3"),
    (3, 1, "f_xz2"),
    (3, 2, "f_xyz"),
    (3, 3, "f_x3"),
    (4, 0, "g_z4"),
    (4, 2, "g_xz"),
    (4, 4, "g_xy"),
    (0, 0, "s"),
]


def amplitudes(theta, phi, states=STATES) -> np.ndarray:
    """(n_states, N) float32 of |Y_lm|, EACH STATE NORMALISED TO max = 1.

    ⚠ The per-state normalisation is load-bearing and has to happen BEFORE any interpolation.
    The raw normalisation constant of Y_lm grows with l (Y_44 peaks ~2.5x Y_00), so a mesh
    built on raw magnitudes swells as the choreography climbs - which is exactly the failure
    the brief asks to avoid, and almost certainly why the source's own ending leaves the
    frame. Normalised, every state has the same maximum reach and the size differences that
    remain are honest shape differences (a p_z really is twice as tall as it is wide).
    """
    out = np.empty((len(states), theta.size), np.float32)
    for i, (l, m, _n) in enumerate(states):
        a = np.abs(real_Y(l, m, theta, phi))
        out[i] = (a / max(a.max(), 1e-12)).astype(np.float32)
    return out


# --------------------------------------------------------------------------------------
# the mesh
# --------------------------------------------------------------------------------------
#: The mesh shape, filled in by `curve_mesh` and read by the arc-length weighting in
#: `render.py`. The two families are laid out as contiguous blocks - meridians first, one block
#: of `n_th` per meridian with theta varying fastest, then latitudes, one block of `n_ph` per
#: latitude with phi varying fastest AND WRAPPING - so a curve's neighbour along itself is its
#: neighbour in memory, and the per-point spacing is one reshape and one diff.
SHAPE = {}

#: How close a sample may get to a pole. Every meridian passes through both, so theta=0 would
#: stack all of them on one pixel and burn a hole in the accumulator. NAMED because `flow.py`
#: needs the same bound to wrap a marching ring inside it, and two copies of `2e-3` in two
#: files is a silent divergence waiting to happen.
POLE_EPS = 2.0e-3


def curve_mesh(n_mer=112, n_th=768, n_lat=64, n_ph=1152):
    """Two families of dense polylines: meridians (phi fixed) and latitudes (theta fixed).

    Sample counts are chosen so the spacing ALONG a curve is sub-pixel at the reference
    frame: a meridian spans ~pi*R ~ 630 ref px in 768 samples. Particle_Fields S7 paid for the
    opposite mistake (sub-pixel POINTS integrating into a continuum); here the points are
    deliberately sub-pixel-spaced so the curve reads as a continuous line, and `point_sigma`
    gives it width.

    The poles are excluded by an epsilon: every meridian passes through both, so theta=0 would
    stack 112 identical points on one pixel and burn a hole in the accumulator.
    """
    eps = POLE_EPS
    # ⚠ AN EMPTY MERIDIAN FAMILY IS LEGAL AND IS HOW DOT MODE IS BUILT. `n_mer=0` means the
    # latitude family carries every point, which turns the mesh from two crossing families of
    # polylines into ONE lat/long LATTICE - the grid's nodes with no connections. Guarded here
    # rather than at the call site because the naive expression divides by n_mer.
    if n_mer <= 0 or n_th <= 0:
        n_mer = n_th = 0
        th1 = np.zeros(0)
        TH1 = np.zeros(0)
        PH1 = np.zeros(0)
    else:
        ph1 = (np.arange(n_mer) + 0.5) * (2.0 * np.pi / n_mer)
        th1 = np.linspace(eps, np.pi - eps, n_th)
        TH1 = np.repeat(th1[None, :], n_mer, 0).ravel()
        PH1 = np.repeat(ph1[:, None], n_th, 1).ravel()

    th2 = np.linspace(0.0, np.pi, n_lat + 2)[1:-1]
    ph2 = np.arange(n_ph) * (2.0 * np.pi / n_ph)
    TH2 = np.repeat(th2[:, None], n_ph, 1).ravel()
    PH2 = np.repeat(ph2[None, :], n_lat, 0).ravel()

    SHAPE.clear()
    SHAPE.update(n_mer=n_mer, n_th=n_th, n_lat=n_lat, n_ph=n_ph, n1=n_mer * n_th)
    return (np.concatenate([TH1, TH2]).astype(np.float64),
            np.concatenate([PH1, PH2]).astype(np.float64))


def directions(theta, phi) -> np.ndarray:
    """(N,3) unit vectors, with the QUANTUM z-AXIS MAPPED TO WORLD +Y.

    The camera's up is world +y and the source stands every orbital upright (p_z is a vertical
    dumbbell), so the axis swap belongs here, once, rather than in a camera roll that would
    also have to be undone by every scale and framing number downstream.
    """
    st = np.sin(theta)
    return np.stack([st * np.cos(phi), np.cos(theta), st * np.sin(phi)],
                    axis=1).astype(np.float32)
