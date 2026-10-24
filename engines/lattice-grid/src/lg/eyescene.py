# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy Earl - https://github.com/ec175/spectrometry_public
"""eyescene.py - the holofoil eye.

A different composition from `circuit_*`, built on the same substrate. Four things change:

1. **The IMAGE is an eye** (`eye.py`), not a wandering field. The illumination field survives
   only as texture multiplied into it, so the lattice is drawing something rather than
   shimmering at random.

2. **Colour is holofoil** (`holo.py`), and the eye's `layer` field drives the hue offset - so
   sclera, iris and pupil land on different bands of the same foil and read as stacked sheets.

3. **The hexagons are a SURFACE** (`panel.py`). A diffuse holofoil source is composited under a
   static transmission map that is bright at the seams and near-black in the middle of a face,
   so the colour escapes at the hexagon edges. Nodes and links are splatted on top, so they
   sit on the surface while the colour sits under it.

4. **Connectivity follows intensity.** Edges are gated by `Content.blend(gate=...)` against the
   local brightness, so lit regions are visibly more wired-up. This is what makes the iris
   read as a dense mesh and the sclera as scattered dots.

Blacks are real blacks: `ambient` is ~0 and the pupil is subtracted, so the unlit parts of the
frame stop blinking at a low level and simply go out.
"""
from __future__ import annotations

import numpy as np

from . import lattice as L
from . import panel as PAN
from .atlas import Atlas
from .config import REF_H, REF_W, SOURCE_PITCH
from .content import GLYPH_NAME, Content
from .eye import Eye
from .field import Illumination, Twinkle, envelope
from .holo import Holofoil


class EyeScene:
    """Holofoil eye on a honeycomb. Subclass and override to make a graded variant."""

    name = "eye_hex"
    duration = 15.0
    lattice_kind = "hex"
    pad_shape = "hex"
    pitch = SOURCE_PITCH
    seed = 7
    bpm = 117.5

    # -- content -------------------------------------------------------------------------
    node_p = 0.62
    edge_p = 0.62          # a CEILING now; the intensity gate does the real selection
    churn = 0.40
    walks = 60
    walk_len = 6
    boxes = 0              # the panel layer draws the cells; outlines would double them
    endcap_p = 0.30
    conn_gain = 1.45       # edge survives where roll < conn_gain * local connectivity
    conn_bias = 0.06

    # -- eye -----------------------------------------------------------------------------
    eye_kw = {}
    field_mix = 0.42       # how much of the wandering field modulates the eye (0 = none)

    # -- illumination --------------------------------------------------------------------
    illum_lo = 0.10
    illum_hi = 0.92
    wavelength = 520.0
    drift = 0.30
    spin = 0.18
    n_spirals = 2
    n_flares = 2
    ambient = 0.0          # TRUE BLACK outside the eye - no low-level blinking
    hot_gain = 2.4
    flare_gain = 0.55
    gamma = 1.5

    # -- element exposure ----------------------------------------------------------------
    gain = 2.3
    link_mul = 1.30        # the seams are the light SOURCES - links dominate nodes
    node_mul = 0.72
    twinkle = 0.30
    twinkle_rate = (2.5, 9.0)

    # -- holofoil -------------------------------------------------------------------------
    holo_kw = dict(cycles=3.4, axis_deg=118.0, warp=0.34, drift=0.055,
                   sheen=0.55, depth_shift=0.30)

    # -- panel / under-surface source ------------------------------------------------------
    source_gain = 1.15
    source_div = 8
    source_smooth = 1.0
    panel_kw = dict(seam_sigma=3.4, face_floor=0.035, bevel=0.45, bevel_at=7.0,
                    bevel_sigma=2.6)

    # -- output ----------------------------------------------------------------------------
    exposure = 1.30
    glow_mul = 1.0
    glow2_mul = 1.0
    streak_mul = 1.0
    mip_gain = None        # None = use RenderConfig default

    # Grain is cut hard and pushed almost entirely into the LIT parts. The source-matching
    # 0.044/0.80 puts ~9 levels of per-frame noise across the whole frame, and in a
    # composition that is 85% black that noise IS the low-level blinking in the unlit
    # regions. Keep enough to stop the highlights looking like flat vector art; let the
    # blacks go out.
    cfg_overrides = dict(grain=0.026, grain_floor=0.10, bg=(0x01, 0x00, 0x02))

    def __init__(self):
        self.lat = L.build(self.lattice_kind, REF_W, REF_H, self.pitch)
        self.atlas = Atlas(self.pitch, self.lat.link_specs, pad_shape=self.pad_shape)
        self._atlas_cache = {1.0: self.atlas}
        self.content = Content(self.lat, self.duration, bpm=self.bpm, seed=self.seed,
                               node_p=self.node_p, edge_p=self.edge_p, churn=self.churn,
                               walks=self.walks, walk_len=self.walk_len, boxes=self.boxes,
                               endcap_p=self.endcap_p)
        self.illum = Illumination(REF_W, REF_H, seed=self.seed, wavelength=self.wavelength,
                                  n_spirals=self.n_spirals, n_flares=self.n_flares,
                                  drift=self.drift, spin=self.spin)
        self.eye = Eye(seed=self.seed, duration=self.duration, **self.eye_kw)
        self.holo = Holofoil(REF_W, REF_H, seed=self.seed, **self.holo_kw)
        self.mids = self.lat.edge_midpoints()
        self.tw_site = Twinkle(self.lat.n_sites, seed=self.seed,
                               depth=self.twinkle, rate=self.twinkle_rate)
        self.tw_edge = Twinkle(self.lat.n_edges, seed=self.seed + 1,
                               depth=self.twinkle, rate=self.twinkle_rate)
        self._grid = {}

    def atlas_for(self, cfg):
        k = round(float(cfg.scale), 6)
        if k not in self._atlas_cache:
            self._atlas_cache[k] = Atlas(
                self.pitch * cfg.scale,
                [(a, ln * cfg.scale) for a, ln in self.lat.link_specs],
                pad_shape=self.pad_shape)
        return self._atlas_cache[k]

    def grid_for(self, cfg):
        k = (cfg.width, cfg.height)
        if k not in self._grid:
            self._grid[k] = PAN.SourceGrid(cfg.width, cfg.height, div=self.source_div)
        return self._grid[k]

    # -- fields --------------------------------------------------------------------------
    def _field(self, P, t):
        """The wandering illumination, as a MULTIPLIER on the eye rather than a source."""
        v = self.illum(P, t, lo=self.illum_lo, hi=self.illum_hi, gamma=self.gamma,
                       ambient=0.0, hot_gain=self.hot_gain, flare_gain=self.flare_gain)
        return (1.0 - self.field_mix) + self.field_mix * v

    def fields(self, t):
        env = envelope(t, self.duration, fade_in=0.5, fade_out=0.8)
        es = self.eye(self.lat.sites, t)
        ee = self.eye(self.mids, t)
        ls = es["lit"] * self._field(self.lat.sites, t) * self.tw_site(t)
        le = ee["lit"] * self._field(self.mids, t) * self.tw_edge(t)
        ls = self.ambient + ls
        le = self.ambient + le
        cs = self.holo(self.lat.sites, t, layer=es["layer"],
                       jitter=self.content.site_jitter)
        ce = self.holo(self.mids, t, layer=ee["layer"], jitter=self.content.edge_jitter)
        return env, ls, le, cs, ce, es, ee

    # -- compose ---------------------------------------------------------------------------
    def compose(self, fr, t, cfg):
        env, ls, le, cs, ce, es, ee = self.fields(t)
        fr.clear()
        if env <= 1e-4:
            return

        # ---- 1. the UNDER-SURFACE source: holofoil x eye, through the seam map ------------
        if self.source_gain > 1e-4:
            g = self.grid_for(cfg)
            ge = self.eye(g.P, t)
            gi = ge["lit"] * self._field(g.P, t)
            gc = self.holo(g.P, t, layer=ge["layer"])
            lo = (gc * (gi * (self.source_gain * env))[:, None]).astype(np.float32)
            src = g.expand(lo, smooth=self.source_smooth)
            tr = PAN.transmission(self.lat, cfg.width, cfg.height, cfg.scale,
                                  **self.panel_kw)
            fr.add_source(src * tr[:, :, None])

        # ---- 2. elements on top -----------------------------------------------------------
        gate = np.clip(self.conn_bias + self.conn_gain * ee["conn"] * np.clip(le, 0, 1.5),
                       0.0, 1.0)
        gw = self.gain * env
        for node, edge, boxes, w in self.content.blend(t, gate=gate):
            ww = gw * w
            ei = np.nonzero(edge)[0]
            if len(ei):
                fr.splat_links(self.mids[ei], self.lat.edge_kind[ei],
                               ce[ei], le[ei] * ww * self.link_mul)
            for gid, gname in GLYPH_NAME.items():
                si = np.nonzero(node == gid)[0]
                if len(si):
                    fr.splat_nodes(self.lat.sites[si], gname, cs[si],
                                   ls[si] * ww * self.node_mul)

    def render_frame(self, fr, t, frame_index=None, cfg=None):
        cfg = cfg or fr.cfg
        self.compose(fr, t, cfg)
        return fr.to_rgb(exposure=self.exposure, glow_mul=self.glow_mul,
                         glow2_mul=self.glow2_mul, streak_mul=self.streak_mul,
                         mip_gain=self.mip_gain,
                         frame_index=int(round(t * 60)) if frame_index is None
                         else frame_index)


# ==========================================================================================
# VARIANTS. Each isolates ONE axis I cannot grade by measurement, so a side-by-side is a
# clean answer to a single question rather than a preference between four different pictures.
# ==========================================================================================

class EyeFoil(EyeScene):
    """A - baseline. Balanced foil, panels clearly readable, mid density."""
    name = "eye_foil"


class EyeFoilDeep(EyeScene):
    """B - DEEPER PANELS. Faces near-opaque, seams tight and hot: maximum "light from
    underneath". Tests how far the surface illusion can be pushed before the picture
    goes to lace."""
    name = "eye_foil_deep"
    panel_kw = dict(seam_sigma=2.3, face_floor=0.012, bevel=0.70, bevel_at=6.0,
                    bevel_sigma=2.0)
    source_gain = 1.75
    link_mul = 1.55
    node_mul = 0.55
    exposure = 1.42
    glow2_mul = 0.85


class EyeFoilWide(EyeScene):
    """C - WIDER FOIL BANDS + stronger depth shift. Fewer, broader colour bands and a big
    hue separation between sclera / iris / pupil: the layered read is carried by COLOUR
    rather than by brightness."""
    name = "eye_foil_wide"
    holo_kw = dict(cycles=1.9, axis_deg=104.0, warp=0.50, drift=0.040,
                   sheen=0.40, depth_shift=0.52)
    source_gain = 1.35
    field_mix = 0.30
    exposure = 1.34


class EyeFoilDense(EyeScene):
    """D - MAXIMUM CONNECTIVITY. Nodes wire up hard wherever it is bright, so the iris
    becomes a solid mesh and the sclera stays scattered. Tests "more nodes
    connected throughout the illuminated portions" at its limit."""
    name = "eye_foil_dense"
    conn_gain = 2.6
    conn_bias = 0.02
    node_p = 0.78
    walks = 130
    link_mul = 1.45
    source_gain = 0.95
    exposure = 1.26


class EyeFoilSparse(EyeScene):
    """E - QUIET. Low element density, source-dominant: the picture is mostly the glowing
    seams themselves, with nodes as accents. The opposite end from D."""
    name = "eye_foil_sparse"
    node_p = 0.30
    conn_gain = 0.95
    walks = 30
    source_gain = 1.90
    link_mul = 0.85
    node_mul = 0.95
    twinkle = 0.22
    exposure = 1.38
    panel_kw = dict(seam_sigma=4.2, face_floor=0.020, bevel=0.30, bevel_at=8.0,
                    bevel_sigma=3.0)


EYE_SCENES = {c.name: c for c in
              (EyeFoil, EyeFoilDeep, EyeFoilWide, EyeFoilDense, EyeFoilSparse)}
