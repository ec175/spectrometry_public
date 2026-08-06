"""content.py - WHAT sits on the lattice, and when it changes.

The mechanism this whole project turns on, and the thing that took measuring to find:

    lighting autocorrelation half-life  0.50 s      (per-cell brightness, reel_51d5e4f3)
    BPM                                 117.5       (vdx spectral-flux autocorrelation)
    one beat at 117.5 BPM               0.511 s

Those are the same number. The lattice content is not drifting continuously and it is not
random per frame - it **re-rolls on the beat**. Occupancy over the whole clip is 100% (every
cell lights at some point) while any single frame is mostly empty, which is only consistent
with a re-rolling assignment rather than a fixed pattern being lit and unlit.

Only a FRACTION (`churn`) of the lattice re-rolls per beat. Re-rolling everything gives a
strobe; re-rolling nothing gives a static PCB with a light moving over it. The clip is
neither. That means the state at beat b depends on beat b-1, so the epoch timeline is
generated once at construction (iterating from beat 0) and cached - ~30 beats of int8/bool
arrays is under a megabyte, and it keeps `--still 7.5` byte-identical to frame 450 of a
render.

Glyph vocabulary, read off the source at native resolution (CLAUDE.md S2):

    EMPTY 0   most cells
    DOT   1   small filled circle
    RING  2   hollow circle - the most common non-empty glyph
    PAD   3   small filled square (hexagon on a hex lattice) - the SMD-pad look

plus lit EDGES (single links), WALKS (chains of links - the PCB traces) and BOXES (outlined
cells). A "dumbbell" - a link with a circle at each end - is not a glyph: it is what you get
when a lit edge forces nodes onto its endpoints, which is `endcap_p` below.
"""
from __future__ import annotations

import numpy as np

EMPTY, DOT, RING, PAD = 0, 1, 2, 3
GLYPH_NAME = {DOT: "dot", RING: "ring", PAD: "pad"}


class Content:
    """The epoch timeline for one lattice.

    node[b]  (N,) int8   glyph id per site at beat b
    edge[b]  (M,) bool   lit links at beat b
    boxes[b] list of (K,2) float32 closed loops
    """

    def __init__(self, lat, duration, bpm=117.5, seed=0,
                 node_p=0.30, edge_p=0.055, churn=0.34,
                 walks=90, walk_len=7, walk_straight=0.72,
                 boxes=14, endcap_p=0.55,
                 glyph_w=(0.30, 0.52, 0.18)):
        self.lat = lat
        self.bpm = float(bpm)
        self.beat = 60.0 / float(bpm)
        self.duration = float(duration)
        self.n_beats = int(np.ceil(duration / self.beat)) + 2
        self.endcap_p = endcap_p
        rng = np.random.default_rng(seed)

        n, m = lat.n_sites, lat.n_edges
        gw = np.array(glyph_w, np.float64)
        gw = gw / gw.sum()

        node = np.zeros(n, np.int8)
        edge = np.zeros(m, bool)
        self.node, self.edge, self.boxes = [], [], []

        for b in range(self.n_beats):
            # -- re-roll a random subset; the rest persists. `churn` is the memory knob. ----
            frac = 1.0 if b == 0 else churn
            sel = rng.random(n) < frac
            newn = np.where(rng.random(n) < node_p,
                            rng.choice([DOT, RING, PAD], size=n, p=gw).astype(np.int8),
                            np.int8(EMPTY))
            node = np.where(sel, newn, node)

            selm = rng.random(m) < frac
            edge = np.where(selm, rng.random(m) < edge_p, edge)

            # -- traces: momentum-biased walks light CHAINS of links ----------------------
            we = self._walks(rng, walks, walk_len, walk_straight)
            edge = edge.copy()
            edge[we] = True

            # -- a lit link usually terminates on a node; that is the dumbbell look --------
            ends = lat.edges[edge].ravel()
            if len(ends):
                cap = ends[rng.random(len(ends)) < endcap_p]
                blank = cap[node[cap] == EMPTY]
                if len(blank):
                    node[blank] = rng.choice([DOT, RING, PAD], size=len(blank),
                                             p=gw).astype(np.int8)

            self.node.append(node.copy())
            self.edge.append(edge.copy())
            self.boxes.append([lat.sample_box(rng) for _ in range(boxes)])

        self.node = np.array(self.node)
        self.edge = np.array(self.edge)
        # A stable per-edge uniform, used by scenes to gate connectivity ON INTENSITY at
        # render time: an edge is drawn only where `edge_roll < f(local brightness)`, so
        # bright regions come out visibly more wired-up than dim ones without the epoch
        # timeline needing to know anything about the illumination field. Fixed per epoch,
        # so connections do not crawl between beats.
        self.edge_roll = rng.random((self.n_beats, m)).astype(np.float32)
        # A stable per-site random number, used to jitter hue so neighbours are related but
        # not identical. Fixed for the whole clip - it is a property of the site, not the beat.
        self.site_jitter = rng.uniform(-1, 1, n).astype(np.float32)
        self.edge_jitter = rng.uniform(-1, 1, m).astype(np.float32)
        # A second, independent per-element uniform. `Palette` uses it both to DECIDE whether
        # an element ignores the regional hue field entirely and to pick what it gets instead
        # - the scattered odd-coloured elements that a smooth field plus a small offset can
        # never produce on a bimodal palette. Fixed for the clip: a property of the site.
        self.site_hue = rng.random(n).astype(np.float32)
        self.edge_hue = rng.random(m).astype(np.float32)

    # -- walks -----------------------------------------------------------------------------
    def _walks(self, rng, count, length, straight):
        """Momentum-biased random walks on the lattice -> edge ids.

        `straight` is the probability of taking the neighbour closest to the current heading.
        At 0 the traces are shapeless blobs; at 1 they are plain straight lines. The source's
        traces run several cells and turn by one lattice angle at a time, which is what a
        momentum bias produces and what a plain random walk does not.
        """
        lat = self.lat
        out = []
        starts = rng.integers(0, lat.n_sites, count)
        for s in starts:
            cur = int(s)
            head = None
            for _ in range(length):
                nb, ne = lat.neighbours(cur)
                if not len(nb):
                    break
                if head is not None and rng.random() < straight:
                    d = lat.sites[nb] - lat.sites[cur]
                    d = d / np.maximum(np.hypot(d[:, 0], d[:, 1])[:, None], 1e-6)
                    k = int(np.argmax(d @ head))
                else:
                    k = int(rng.integers(0, len(nb)))
                nxt = int(nb[k])
                out.append(int(ne[k]))
                v = lat.sites[nxt] - lat.sites[cur]
                nv = np.hypot(v[0], v[1])
                head = v / nv if nv > 1e-6 else head
                cur = nxt
        return np.array(out, np.int32) if out else np.zeros(0, np.int32)

    # -- sampling --------------------------------------------------------------------------
    def beat_pos(self, t):
        """(beat index, fraction into the beat)."""
        x = t / self.beat
        b = int(np.floor(x))
        return min(b, self.n_beats - 2), float(x - np.floor(x))

    def blend(self, t, xfade=0.42, gate=None):
        """Return [(node_state, edge_state, boxes, weight), ...] for this instant.

        `gate` (M,) in [0,1], optional: an edge survives only where its stable per-epoch roll
        is under the gate, so connectivity scales with local brightness.

        Outside the crossfade window: one entry at weight 1. Inside it, the re-roll is split
        into THREE disjoint groups rather than two overlapping ones -

            unchanged   drawn once at weight 1
            old only    fading out, weight 1-w
            new only    fading in,  weight w

        - instead of drawing the whole old state at 1-w and the whole new state at w. The two
        are arithmetically identical (additive compositing, and the weights sum to 1 on any
        element present in both), but the split matters twice over:

        1. Cost. Only `churn` of the lattice actually differs between consecutive beats, so
           this draws ~1.25x the elements during a crossfade instead of 2x.
        2. Correctness. It guarantees no site is splatted twice in one call, which is the
           precondition for `Frame.splat(disjoint=True)` on nodes. Without it an unchanged
           site would appear in both groups at the same position and the buffered fancy-index
           add would drop one of the two contributions.
        """
        b, f = self.beat_pos(t)
        if gate is not None:
            # intensity-driven connectivity: keep only edges whose stable per-epoch roll
            # falls under the local gate. `gate` is (M,) in [0,1].
            e_b = self.edge[b] & (self.edge_roll[b] < gate)
            e_b1 = self.edge[b + 1] & (self.edge_roll[b + 1] < gate)
        else:
            e_b, e_b1 = self.edge[b], self.edge[b + 1]
        if f >= xfade:
            return [(self.node[b + 1], e_b1, self.boxes[b + 1], 1.0)]
        w = f / xfade                                   # 0 -> old, 1 -> new
        w = w * w * (3.0 - 2.0 * w)
        n0, n1 = self.node[b], self.node[b + 1]
        e0, e1 = e_b, e_b1
        same_n = n0 == n1
        same_e = e0 == e1
        keep_n = np.where(same_n, n0, np.int8(EMPTY))
        old_n = np.where(same_n, np.int8(EMPTY), n0)
        new_n = np.where(same_n, np.int8(EMPTY), n1)
        return [(keep_n, e0 & same_e, self.boxes[b + 1], 1.0),
                (old_n, e0 & ~same_e, [], 1.0 - w),
                (new_n, e1 & ~same_e, [], w)]

    def stats(self):
        n = self.node.shape[1]
        m = self.edge.shape[1]
        occ = (self.node != EMPTY).mean(1)
        eon = self.edge.mean(1)
        ever = (self.node != EMPTY).any(0).mean()
        churn = (self.node[1:] != self.node[:-1]).mean()
        return dict(sites=n, edges=m, node_occ=float(occ.mean()), edge_on=float(eon.mean()),
                    ever_lit=float(ever), beat_churn=float(churn), beats=self.n_beats,
                    beat_s=self.beat)
