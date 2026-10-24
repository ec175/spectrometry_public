# SPDX-License-Identifier: MIT
# From spectrometry.mp4 engines by Clancy Earl - https://github.com/ec175/spectrometry_public
"""scenes.py - the scenes, and the per-frame composition.

A scene is (substrate, content, illumination, palette, exposure). `compose` is shared by all of
them: swapping `lattice_kind` from "square" to "hex" is the ONLY difference between
`circuit_square` and `circuit_hex`, which is the whole point of the format.

Per-frame order, and why:

 1. Evaluate the illumination and palette fields ONCE per frame at the site positions and at
    the edge midpoints. They do not depend on which content epoch is being drawn, so a
    crossfade frame must not pay for them twice.
 2. Boxes first, then links, then nodes. Additive compositing is order-independent, but the
    intensities are tiered (boxes dimmest, nodes brightest) so that a node reads as sitting ON
    a trace rather than being lost in it - the same read as the source.
"""
from __future__ import annotations

import numpy as np

from . import lattice as L
from .atlas import Atlas
from .content import EMPTY, GLYPH_NAME, Content
from .field import Illumination, Palette, Twinkle, envelope
from .network import Network
from .config import REF_H, REF_W, SOURCE_PITCH


class LatticeScene:
    """Base scene. Subclasses set the class attributes; nothing else needs overriding."""

    name = "scene"
    duration = 15.0
    lattice_kind = "square"
    pitch = SOURCE_PITCH
    seed = 7
    bpm = 117.5                      # measured on reel_51d5e4f3 (vdx)

    # content density. Deliberately sparse: the source's lit regions BREATHE - there is black
    # between neighbouring glyphs inside a bright patch. Filling the lattice turns the region
    # into a solid carpet and the bloom then integrates it to a flat wash.
    node_p = 0.46
    edge_p = 0.085
    churn = 0.46
    walks = 200
    walk_len = 7
    walk_straight = 0.72
    boxes = 26
    endcap_p = 0.40
    xfade = 0.42          # share of a beat spent crossfading between content epochs

    # illumination. `illum_lo` is the most powerful control in the project: it sets how far up
    # the field a site must be to appear at all, so raising it thins the lit regions into arms
    # rather than shrinking them uniformly.
    # A NARROW gate band is what makes a cell swing. `cell_std` (67.3 in the source) needs an
    # element to be either off or blazing, not permanently half-lit; widening the band to buy
    # brightness instead flattens every cell's time series. Brightness comes from density and
    # exposure, contrast comes from here - they are separate controls and conflating them was
    # the first VFX pass's error.
    # A NARROW gate band is what makes the colour SATURATE. Under the RGB delay, saturation
    # is set by how ABRUPTLY an element switches: a hard 0->1 shows pure blue for three
    # frames, then cyan, then white; a smooth fade shows all three channels part-way at once
    # and comes out pastel. reel_466a0f52 runs mean_sat_bright 0.531 against reel_51d5e4f3's
    # 0.389, and that difference is almost entirely transition hardness.
    illum_lo = 0.20
    illum_hi = 0.55
    wavelength = 620.0
    drift = 0.54           # how fast the field sweeps -> how often a cell changes state
    spin = 0.33
    # A NARROW angular spread makes the lit regions elongated streaks rather than round
    # patches, which is most of the source's filament-ness (structure coherence 0.326).
    axis_deg = 52.0
    axis_spread = 20.0
    # The octaves are for SPATIAL detail and must now stay nearly still. A travelling wave
    # has a brightening leading edge and a dimming trailing edge, and under the RGB delay
    # those are OPPOSITE hues - so a moving octave flips the colour every half wavelength.
    # With the 155 px octave alive, hue correlation at eight pitches (154 px) went NEGATIVE
    # (-0.06) against the source's +0.484. Frozen, they add texture and no colour reversal.
    octave_speed = 0.75
    resp_lag = 0.0         # seconds; see the per-element response-lag note in __init__
    n_spirals = 2
    n_flares = 3
    # `ambient` is the `dark_frac` control of choice over more wide-bloom: the source's dark
    # regions carry faint STRUCTURE (dim elements), not a smooth wash, so lifting them with
    # the element floor is the faithful mechanism and lifting them with glow2 is not.
    ambient = 0.042        # floor, so the dark regions still carry faint structure
    hot_gain = 0.95        # how far past the gate a crest drives its elements. LOW now: a
                           # blown-out core is white by definition, and white is the one thing
                           # a colour-glitch look cannot afford (reel_466a0f52 blows out only
                           # 1.1% of pixels, against 5.4% for reel_51d5e4f3).
    flare_gain = 0.95

    # VFX layer. `twinkle` is per-element and independent; it is what carries the source's
    # high-frequency motion, which no smooth field can (field.Twinkle).
    # `twinkle` shares the high-frequency budget with GRAIN, which is per-frame white noise
    # and therefore pure HF. Raising grain to its measured level pushed hf_ratio from 0.178
    # to 0.195 on its own; the twinkle has to come down to make room.
    twinkle = 0.46
    twinkle_rate = (3.0, 11.0)
    streak_mul = 1.0

    # exposure. High `gamma` keeps mid-lit elements dim and saturated so only the field's
    # crests reach the intensity where the tonemap bleaches them to white - that split between
    # a coloured body and a white core is the source's most recognisable quality.
    gain = 1.9
    gamma = 2.1
    box_mul = 0.40
    link_mul = 0.78
    exposure = 1.34
    glow_mul = 1.0
    glow2_mul = 1.0
    pad_shape = "square"
    # MONO + RGB DELAY. The reference clips are a black-and-white render put through a glitch
    # pass that delays the channels (blue 0, green 3, red 6 frames at 30 fps) - confirmed by
    # cross-correlation, see rgbdelay.py. With this on, `Palette` is unused: the warm/cool
    # bimodality everything was tuned against is an OUTPUT of the delay, not an input.
    mono = True
    rgb_delay = None       # None = rgbdelay.ChannelDelay.DEFAULT (0.2 s, 0.1 s, 0 s)

    # NETWORK (network.py). The chain, not the node, is the unit that lights.
    chains = False
    chain_coverage = 0.26  # share of sites belonging to some chain
    chain_spine = 10       # max nodes on a chain's spine, before branches
    chain_period = (1.6, 5.5)
    chain_duty = 0.34
    chain_singles = 0.30   # share of chains that are a lone node
    loose_gain = 0.30      # how bright the UNCHAINED lattice gets
    loose_depth = 0.55
    field_floor = 0.35     # how much the illumination field still weights a lit chain
    # Unchained nodes TOUCHING a lit chain occasionally light with it - the circuit bleeding
    # into the dark lattice around it. The same behaviour kept from the glitch, applied to
    # the network instead. Keep it sparse or it just thickens every chain into a blob.
    border_flash = 0.0
    border_duty = 0.16

    # GLITCH overlay (glitch.py), applied to the mono frame BEFORE the RGB delay.
    glitch = False
    glitch_kw = {}

    # DARK REGIONS: a slow field that multiplies everything and reaches true zero.
    dark_depth = 0.0
    dark_wavelength = 1500.0
    dark_drift = 0.10
    dark_lo = 0.10
    dark_hi = 0.62
    # The mip-chain glow was added for the eye composition ("lighting doesn't flush well")
    # and is a REGRESSION here: it is a wide, smooth lift that smears neighbouring elements
    # into each other, and this format's whole character is discrete elements on black.
    # Measured on circuit_hex with it on: local contrast 0.359 against the source's 0.593.
    mip_gain = 0.0

    def __init__(self):
        self.lat = L.build(self.lattice_kind, REF_W, REF_H, self.pitch)
        # Reference-resolution atlas, for --dry reporting only. Rendering must go through
        # atlas_for(cfg) - see the note there.
        self.atlas = Atlas(self.pitch, self.lat.link_specs, pad_shape=self.pad_shape)
        self._atlas_cache = {1.0: self.atlas}
        self.content = Content(self.lat, self.duration, bpm=self.bpm, seed=self.seed,
                               node_p=self.node_p, edge_p=self.edge_p, churn=self.churn,
                               walks=self.walks, walk_len=self.walk_len, boxes=self.boxes,
                               endcap_p=self.endcap_p, walk_straight=self.walk_straight)
        self.illum = Illumination(REF_W, REF_H, seed=self.seed, wavelength=self.wavelength,
                                  n_spirals=self.n_spirals, n_flares=self.n_flares,
                                  drift=self.drift, spin=self.spin,
                                  axis_deg=self.axis_deg, axis_spread=self.axis_spread,
                                  octave_speed=self.octave_speed)
        self.pal = Palette(REF_W, REF_H, seed=self.seed)
        self.mids = self.lat.edge_midpoints()
        # Indexed by ELEMENT, not position - that is the point (see field.Twinkle).
        self.tw_site = Twinkle(self.lat.n_sites, seed=self.seed,
                               depth=self.twinkle, rate=self.twinkle_rate)
        self.tw_edge = Twinkle(self.lat.n_edges, seed=self.seed + 1,
                               depth=self.twinkle, rate=self.twinkle_rate)
        self._white_s = np.ones((self.lat.n_sites, 3), np.float32)
        self._white_e = np.ones((self.lat.n_edges, 3), np.float32)
        self.net = Network(self.lat, seed=self.seed, coverage=self.chain_coverage,
                           spine=self.chain_spine, period=self.chain_period,
                           duty=self.chain_duty, singles=self.chain_singles) \
            if self.chains else None
        # DARK REGIONS. A slow, very long-wavelength field that multiplies everything and
        # genuinely reaches zero, so whole areas of the frame go out and come back. The
        # illumination gate cannot do this on its own: it has a floor (`ambient`) that applies
        # everywhere by construction, which is exactly why the frame never went fully dark
        # anywhere.
        self.dark = Illumination(REF_W, REF_H, seed=self.seed + 991,
                                 wavelength=self.dark_wavelength, n_waves=3, n_spirals=1,
                                 n_flares=0, drift=self.dark_drift, spin=0.05,
                                 axis_spread=90.0, octaves=1, octave_gain=0.30)
        # Per-element RESPONSE LAG. Each element reacts to the field at its own moment,
        # spread over `resp_lag` seconds. Under the RGB delay this is what decorrelates hue
        # at SHORT range without touching the regional pattern: neighbours cross the same
        # sweeping wavefront a few frames apart, so they show different colours, while the
        # wavefront itself stays coherent across the frame. Implemented as a per-element
        # blend of the field at two times, which costs one extra field evaluation instead of
        # rebuilding it to accept per-element time.
        _r = np.random.default_rng(self.seed + 8123)
        self._lag_s = _r.random(self.lat.n_sites).astype(np.float32)
        self._lag_e = _r.random(self.lat.n_edges).astype(np.float32)

    def atlas_for(self, cfg):
        """The glyph atlas AT THE OUTPUT RESOLUTION, cached per scale.

        Stamps are baked in PIXELS. `Frame.splat` scales an element's POSITION by cfg.scale
        but composites the stamp as-is, so an atlas baked at reference pitch draws 19.25 px
        glyphs into a 540-wide frame whose pitch is only 9.6 px - every glyph twice its
        correct relative size, the lattice half as many cells across, and the frame far
        denser and brighter than the final it is supposed to preview.

        That bug made the whole first VFX tuning pass worthless: `look_check` measured
        mean_luma 58 at 540 wide against 27 in the 1080 final. Bake the atlas at
        pitch * scale and a preview is a preview again.
        """
        k = round(float(cfg.scale), 6)
        if k not in self._atlas_cache:
            self._atlas_cache[k] = Atlas(
                self.pitch * cfg.scale,
                [(a, ln * cfg.scale) for a, ln in self.lat.link_specs],
                pad_shape=self.pad_shape)
        return self._atlas_cache[k]

    # -- per-frame -------------------------------------------------------------------------
    def fields(self, t):
        """Illumination + colour at every site and every edge midpoint, once per frame."""
        env = envelope(t, self.duration)
        kw = dict(lo=self.illum_lo, hi=self.illum_hi, gamma=self.gamma,
                  ambient=self.ambient, hot_gain=self.hot_gain,
                  flare_gain=self.flare_gain)
        if self.net is not None:
            # The CHAIN is the unit that lights, not the node. The illumination field is
            # demoted to a slow regional weighting on top of it, and the dark field can take
            # a whole area to zero regardless of either.
            ns, ne = self.net.levels(t, loose_gain=self.loose_gain,
                                     loose_depth=self.loose_depth,
                                     border_flash=self.border_flash,
                                     border_duty=self.border_duty)
            fs = self.illum(self.lat.sites, t, **kw)
            fe = self.illum(self.mids, t, **kw)
            ds = self._dark(self.lat.sites, t)
            de = self._dark(self.mids, t)
            ls = ns * (self.field_floor + (1.0 - self.field_floor) * fs) * ds
            le = ne * (self.field_floor + (1.0 - self.field_floor) * fe) * de
            if self.twinkle > 1e-4:
                ls = ls * self.tw_site(t)
                le = le * self.tw_edge(t)
            if self.mono:
                return env, ls, le, self._white_s, self._white_e
            cs = self.pal(self.lat.sites, t, jitter=self.content.site_jitter,
                          scatter_u=self.content.site_hue)
            ce = self.pal(self.mids, t, jitter=self.content.edge_jitter,
                          scatter_u=self.content.edge_hue)
            return env, ls, le, cs, ce
        if self.resp_lag > 1e-4:
            tb = t - self.resp_lag
            ls = (self.illum(self.lat.sites, t, **kw) * (1.0 - self._lag_s)
                  + self.illum(self.lat.sites, tb, **kw) * self._lag_s) * self.tw_site(t)
            le = (self.illum(self.mids, t, **kw) * (1.0 - self._lag_e)
                  + self.illum(self.mids, tb, **kw) * self._lag_e) * self.tw_edge(t)
        else:
            ls = self.illum(self.lat.sites, t, **kw) * self.tw_site(t)
            le = self.illum(self.mids, t, **kw) * self.tw_edge(t)
        if self.mono:
            # BLACK AND WHITE. All the colour comes from the RGB-delay pass downstream
            # (rgbdelay.py) - the palette is an artefact of that, not an input. Full-length
            # arrays, because the compositor indexes colour per element.
            return env, ls, le, self._white_s, self._white_e
        cs = self.pal(self.lat.sites, t, jitter=self.content.site_jitter,
                      scatter_u=self.content.site_hue)
        ce = self.pal(self.mids, t, jitter=self.content.edge_jitter,
                      scatter_u=self.content.edge_hue)
        return env, ls, le, cs, ce

    def compose(self, fr, t):
        env, ls, le, cs, ce = self.fields(t)
        fr.clear()
        if env <= 1e-4:
            return
        g = self.gain * env

        for node, edge, boxes, w in self.content.blend(t, xfade=self.xfade):
            gw = g * w

            # -- boxes: outlined cells, one colour per box (its first edge) ----------------
            if boxes:
                bidx = np.concatenate(boxes) if len(boxes) > 1 else boxes[0]
                if len(bidx):
                    bcol = np.concatenate([
                        np.repeat(ce[b[:1]], len(b), 0) for b in boxes if len(b)])
                    fr.splat_links(self.mids[bidx], self.lat.edge_kind[bidx],
                                   bcol, le[bidx] * gw * self.box_mul)

            # -- links ---------------------------------------------------------------------
            # With a network, the WIRING IS THE NETWORK'S, not the content's. The chains have
            # to be stable to read as circuits - if their edges were re-rolled every beat
            # alongside the glyphs, a chain would never be visible as a connected run and the
            # whole point of the structure would be lost. The content still churns the node
            # glyphs on top.
            ei = (np.nonzero(self.net.edge_chain >= 0)[0] if self.net is not None
                  else np.nonzero(edge)[0])
            if len(ei):
                fr.splat_links(self.mids[ei], self.lat.edge_kind[ei],
                               ce[ei], le[ei] * gw * self.link_mul)

            # -- nodes ---------------------------------------------------------------------
            for gid, gname in GLYPH_NAME.items():
                si = np.nonzero(node == gid)[0]
                if len(si):
                    fr.splat_nodes(self.lat.sites[si], gname, cs[si], ls[si] * gw)

    def _dark(self, P, t):
        """(N,) in [0,1], reaching TRUE zero over whole regions."""
        if self.dark_depth <= 1e-4:
            return 1.0
        v = self.dark(P, t, lo=self.dark_lo, hi=self.dark_hi, gamma=1.0,
                      ambient=0.0, hot_gain=0.0, flare_gain=0.0)
        return 1.0 - self.dark_depth + self.dark_depth * np.clip(v, 0.0, 1.0)

    def glitch_for(self, cfg):
        if not self.glitch:
            return None
        if getattr(self, "_glitch", None) is None:
            from .glitch import Glitch
            self._glitch = Glitch(cfg.fps, seed=self.seed, **self.glitch_kw)
        return self._glitch

    def render_mono(self, fr, t, frame_index=None):
        """The black-and-white frame, pre-background and pre-grain. This is what the
        RGB-delay pass consumes; see rgbdelay.py.

        The GLITCH goes here - before the delay, not after. Applied afterwards it would
        displace all three channels together and produce grey blocks; applied here, a block
        that exists now and did not 0.1 s ago lands in the blue channel alone, which is the
        solid blue rectangle the reference is full of.
        """
        self.compose(fr, t)
        m = fr.to_rgb(exposure=self.exposure, glow_mul=self.glow_mul,
                      glow2_mul=self.glow2_mul, streak_mul=self.streak_mul,
                      mip_gain=self.mip_gain, mono=True)
        g = self.glitch_for(fr.cfg)
        if g is not None:
            fi = int(round(t * fr.cfg.fps)) if frame_index is None else frame_index
            m = g(m, fi)
        return m

    def render_frame(self, fr, t, frame_index=None):
        """`frame_index` seeds the grain. Defaults to the frame number t implies at 60 fps,
        so a still and the render's frame at the same t get the same noise.

        For a mono scene this renders the delay history too, so a single still is correct in
        isolation - the animation path in render.py streams it far more cheaply."""
        if self.mono:
            from .rgbdelay import ChannelDelay
            fps = fr.cfg.fps
            d = ChannelDelay(fps, self.rgb_delay)
            for k in range(d.n - 1, -1, -1):
                d.push(self.render_mono(fr, max(0.0, t - k / fps)))
            rgb = d.push(self.render_mono(fr, t))
            return fr.finish(rgb, int(round(t * 60)) if frame_index is None
                             else frame_index)
        self.compose(fr, t)
        return fr.to_rgb(exposure=self.exposure, glow_mul=self.glow_mul,
                         glow2_mul=self.glow2_mul, streak_mul=self.streak_mul,
                         mip_gain=self.mip_gain,
                         frame_index=int(round(t * 60)) if frame_index is None
                         else frame_index)


# ==========================================================================================
class CircuitSquare(LatticeScene):
    """The recreation: square cells, circular nodes, 0/45/90/135 links."""
    name = "circuit_square"
    lattice_kind = "square"
    pad_shape = "square"


class CircuitHex(LatticeScene):
    """The variation: hexagonal cells (honeycomb), same node + connector vocabulary.

    A honeycomb site has 3 neighbours where a square site has 8, so the lattice carries a
    third as many edges (7.8k vs 23k) and the density settings have to be re-derived, not
    reused. The right invariant is LIT EDGES PER NODE, not the lit-edge fraction: the square
    scene lights 8.2% of 8 neighbours = 0.66 per node, so the honeycomb wants ~22% of 3.

    Matching the fraction instead (the first attempt, at edge_p 0.13 plus 150 walks) put 38%
    of edges on = 1.15 per node, which is past the honeycomb's percolation point. Every cell
    joined up and the frame read as continuous chicken-wire with no distinguishable nodes,
    traces or empty cells - and it passed `look_check` on all eight metrics while doing it,
    because bulk luminance and palette statistics cannot see structure. That is the standing
    reminder that `look_check` is necessary and not sufficient.
    """
    name = "circuit_hex"
    lattice_kind = "hex"
    pad_shape = "hex"
    # Long, few walks rather than short, many: the source's lit structure is FILAMENTARY
    # (structure-tensor coherence 0.326 against 0.217 for short chains), and a filament is
    # one long trace, not a cluster of stubs.
    # DENSE and DIM, not sparse and blown. Under the RGB delay a clipped pixel is white by
    # definition - once the tonemap pins mono at 1.0, the three delayed copies are all 1.0 and
    # the colour is gone. So brightness has to come from COVERAGE rather than from exposure:
    # measured, dropping exposure 2.05 -> 0.65 took saturation 0.434 -> 0.590 and white
    # 0.355 -> 0.246, but also mean luma 37 -> 20. Many unclipped elements restore the luma
    # without restoring the clipping.
    edge_p = 0.115
    walks = 230
    walk_len = 18
    # A honeycomb walk cannot go straight - every second edge turns 60 deg - so the best it
    # can do is the ZIGZAG that keeps a constant net heading. A high straight-preference is
    # what picks that zigzag over an aimless one, and at blur scale it reads as a filament.
    walk_straight = 0.93
    boxes = 12            # complete hexagon outlines. A closed hexagon is ISOTROPIC, so
                          # papering the frame with them fights the filament structure -
                          # enough to read the geometry, not enough to become wallpaper.
    # Density has to come DOWN once the hue scatter is in. Scattered hues only read as
    # variety while elements stay separable; packed shoulder to shoulder under a sigma-9.9
    # halo, a green next to an orange next to a blue just averages, and the frame goes
    # white and desaturated (measured: white_of_bright 0.51 against the source's 0.32,
    # saturation 0.26 against 0.39). Variety and saturation are the same knob here.
    node_p = 0.82         # nodes, not edges, are how this lattice gets its ink back: they
    link_mul = 0.86       # add brightness without joining cells into chicken-wire
    # With the RGB delay in place, per-element twinkle is mostly DOUBLE-COUNTING: the delay
    # itself turns any frame-to-frame change into visible colour, and it was the measured
    # high-frequency energy that motivated a large twinkle in the first place. Left high, it
    # makes every element strobe between the three primaries - confetti, not glitch.
    # Under the delay, hue IS rate-of-change, so the SPATIAL COHERENCE of the change becomes
    # the spatial coherence of the colour. The source holds hue correlation 0.484 out at eight
    # pitches, which means whole regions change together - the illumination field sweeping
    # across. Per-element randomness (twinkle, churn) decorrelates that: at twinkle 0.09 /
    # churn 0.54 the 8-pitch correlation collapsed to 0.06. Push the change into the field.
    # Colour must appear EVERYWHERE, not just at the edges of the lit regions. Under the
    # delay a pixel is coloured only while it is changing, so if the only thing changing is
    # the advancing front of a sweeping field, only that front is coloured - which is exactly
    # what the frame looked like. Elements have to change at many DIFFERENT times, spread
    # across the whole frame.
    #
    # This is safe here in a way it was not for reel_51d5e4f3: that reel holds hue
    # correlation 0.484 at eight pitches, but reel_466a0f52 - the reference for this look -
    # sits at 0.236, i.e. its colour genuinely is spatially independent.
    # HIGH. Under the RGB delay a pixel is coloured only while it is changing, and
    # reel_466a0f52 has 81% of its bright pixels changing inside the 0.2 s delay window -
    # essentially everything is in motion all the time. Swept: at twinkle 0.30 the frame
    # comes out white_of_bright 0.33 / sat 0.42; at 0.80 it lands 0.19 / 0.53, which is the
    # reference almost exactly. The slower rate band beats the faster one at every setting.
    twinkle = 0.80
    # Capped near 5 Hz (period 0.2 s). Faster than that and an element completes a whole
    # cycle INSIDE the delay window, so the three channels sample it at unrelated phases and
    # the result is magenta/green rather than the blue-cyan-white-yellow-red progression a
    # monotonic transition gives.
    twinkle_rate = (1.6, 5.2)
    resp_lag = 0.46       # a wide spread of per-element response times, so neighbours cross
                          # the same wavefront at visibly different moments
    # BLOOM IS THE ENEMY OF THE GLITCH. A halo is an average over many elements, so it
    # changes slowly and smoothly even when the elements under it are switching hard - and
    # under the RGB delay anything slow-changing is WHITE. With the halos at their
    # reel_51d5e4f3 strengths, bright pixels were bloom rather than ink and white_of_bright
    # stuck at 0.54 against the glitch reel's 0.186 no matter how abrupt the content became.
    # reel_466a0f52 is crisp thin strokes with very little glow, and that is why its colour
    # survives.
    glow_mul = 0.85
    glow2_mul = 1.00
    # `gain` sets per-ELEMENT intensity and therefore what survives the splat cull, so it -
    # not `exposure` - is what decides how much of the lattice is present at all. With chain
    # scheduling most elements sit well below their old level, and at the inherited gain only
    # 3100 of 13000 survived; the frame was a scatter of circuits on black.
    gain = 4.6
    exposure = 4.35       # LOW on purpose - see the density note above
    # A floor lights EVERY edge, and on a honeycomb every edge joins up - so a floor that is
    # invisible on a square lattice draws a continuous faint mesh here, which the source's
    # dark regions (scattered dim dots, no grid) never show. Keep it low and buy the
    # dark_frac back with exposure instead.
    # reel_466a0f52 has almost NO pure black (dark_frac 0.011): the lattice is faintly
    # visible everywhere, which is most of what reads as "node details". A higher floor is
    # exactly that. It does re-introduce the honeycomb mesh in the darks, so this is a
    # deliberate trade toward the glitch reference rather than reel_51d5e4f3.
    # Near zero: the CHAIN schedule decides what is lit now, so a global floor would only
    # wash the un-wired lattice back in and undo the structure.
    ambient = 0.02

    # The network and the glitch overlay, both on. See network.py / glitch.py.
    chains = True
    glitch = True
    # Measured against the reference: it carries ~50 blue-dominant regions over 400 px per
    # frame; the first setting produced 9. More blocks, bigger, and more of them solid-lit.
    # SMALL, and synced to the delay: one block per event, jumping position every 0.1 s, so
    # the channel delay renders it as three identical displaced copies in blue/green/red.
    # `size` is a fraction of frame width - the previous build went up to 0.24, which is a
    # quarter of the frame, and read as huge unrelated rectangles rather than a glitch.
    # The jump has to be SMALL relative to the block, or the three delayed copies land far
    # apart and read as unrelated scattered rectangles instead of one glitch displaced three
    # times. A couple of block-widths is what makes the triple legible.
    glitch_kw = dict(n_events=(7, 18), size=(0.010, 0.045), life=(2, 5),
                     jump=(0.006, 0.035), gain=(0.6, 1.6), active=0.90)
    border_flash = 0.62
    # MUST be 1.0 to mean what it says. The multiplier is (1-depth) + depth*v, so anything
    # below 1.0 leaves a floor and the region never actually goes out - at 0.62 not one of 64
    # tiles ever dropped below mean luma 6 over the whole clip, i.e. "completely dark" was
    # simply not happening. `dark_lo` then sets how much of the field is pinned at zero.
    dark_depth = 1.0      # whole regions go genuinely out and come back
    dark_lo = 0.30
    field_floor = 0.62    # the field weights a lit chain but must not smother it
    # The un-wired lattice carries most of the frame's AREA - 74% of sites are loose - so it
    # sets the overall level even though the chains are the subject. Too low and the frame is
    # a few bright circuits on black with nothing between them.
    loose_gain = 0.78
    loose_depth = 0.62
    chain_period = (1.3, 4.4)
    illum_hi = 0.66       # ...and a narrower gate to keep cell_std up despite that floor
    # Most elements re-roll every beat, and they SNAP rather than crossfade. Both exist to
    # feed the RGB delay: an element that changes abruptly every 0.5 s spends a large share
    # of its life inside the 0.2 s delay window, which is where all the colour lives. A long
    # crossfade (the 0.42 default) moves all three channels together and stays white.
    churn = 0.86
    xfade = 0.07
    # Sweep RATE sets how much of the frame is coloured at all. An element that stays lit for
    # two seconds is only coloured during the first and last 0.2 s of that - the delay window
    # - and reads white in between; the source has only 32% of its bright pixels neutral, so
    # its elements must switch on a timescale close to the delay itself. Rate is independent
    # of spatial coherence (that is set by wavelength), so this buys colour for free.
    drift = 1.55
    spin = 0.88
    # of colour - see the twinkle note above


class CircuitTri(LatticeScene):
    """Hex-PACKED nodes: 6 neighbours, links at 0/60/120, triangular cells.

    The other reading of "hexagons" - a denser fabric closer to the square version's
    8-neighbour connectivity, with hexagonal rings as its box glyph.
    """
    name = "circuit_tri"
    lattice_kind = "triangular"
    pad_shape = "hex"
    edge_p = 0.055
    walks = 100
    boxes = 18


from .eyescene import EYE_SCENES                          # noqa: E402

SCENES = {
    "circuit_square": CircuitSquare,
    "circuit_hex": CircuitHex,
    "circuit_tri": CircuitTri,
    **EYE_SCENES,
}
