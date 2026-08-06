"""lattice.py - the SUBSTRATE. This is the swappable part of the format.

Everything downstream (content.py, atlas.py, draw.py) is written against this interface and
nothing else, which is what makes `circuit_square` and `circuit_hex` the same engine:

    sites        (N,2) float32   world-px node positions
    edges        (M,2) int32     index pairs, each an adjacency the content layer may light
    edge_kind    (M,)  int32     index into `link_specs`
    link_specs   [(angle_deg, length_px)]   the DISTINCT link stamps the atlas must bake
    nbr_*        CSR adjacency   used by content.walks() to grow multi-cell traces
    sample_box() EDGE IDS of one closed outlined cell - the box glyph

`sample_box` returns edge ids rather than a free polyline on purpose. Every box in the source
runs along lattice lines, so expressing it as a set of edges means it reuses the atlas's
anti-aliased link stamps and lands exactly on its nodes - where a free polyline would need its
own rasteriser and would stair-step on the hex lattice's 30/150-degree sides.

A lattice is generated ONCE per scene and never moves. The source clip's camera measured
`locked, 0.0 px/s drift` (vdx), and everything that changes on screen is the content layer
and the illumination field, not the substrate.

Two implementations:

  SquareLattice  - the source geometry. 8-neighbour: 4 orthogonal + 4 diagonal, so links run
                   at 0/45/90/135 deg. Diagonals are sqrt(2) longer, which is visible in the
                   clip and is why `link_specs` carries a length per kind rather than assuming
                   one.
  HexLattice     - the requested variation.
                   mode="honeycomb"  (default) nodes are hexagon VERTICES, 3 neighbours each,
                                     and the CELLS you see are hexagons. This is the literal
                                     read of "square cells -> hexagons".
                   mode="triangular" nodes are hex-PACKED points, 6 neighbours, links at
                                     0/60/120. Denser fabric, triangular cells. Kept because
                                     it is the closer analogue of the square version's
                                     8-neighbour connectivity.
"""
from __future__ import annotations

import numpy as np

# Loop closure tolerance when welding shared hexagon vertices, in units of edge length.
_WELD = 1e-3


class Lattice:
    """Base: owns the CSR adjacency build and the shared accessors."""

    name = "lattice"

    def __init__(self, sites, edges, edge_kind, link_specs, pitch):
        self.sites = np.asarray(sites, np.float32)
        self.edges = np.asarray(edges, np.int32)
        self.edge_kind = np.asarray(edge_kind, np.int32)
        self.link_specs = link_specs
        self.pitch = float(pitch)
        self._build_adjacency()
        self._build_edge_index()

    # -- adjacency ---------------------------------------------------------------------
    def _build_adjacency(self):
        """CSR neighbour lists, both directions. content.walks() steps this."""
        n = len(self.sites)
        a = np.concatenate([self.edges[:, 0], self.edges[:, 1]])
        b = np.concatenate([self.edges[:, 1], self.edges[:, 0]])
        e = np.concatenate([np.arange(len(self.edges)), np.arange(len(self.edges))])
        order = np.argsort(a, kind="stable")
        a, b, e = a[order], b[order], e[order]
        self.nbr_site = b.astype(np.int32)
        self.nbr_edge = e.astype(np.int32)
        self.nbr_start = np.zeros(n + 1, np.int32)
        np.add.at(self.nbr_start, a + 1, 1)
        self.nbr_start = np.cumsum(self.nbr_start).astype(np.int32)

    def neighbours(self, i):
        s, e = self.nbr_start[i], self.nbr_start[i + 1]
        return self.nbr_site[s:e], self.nbr_edge[s:e]

    # -- site-pair -> edge id ------------------------------------------------------------
    def _build_edge_index(self):
        n = len(self.sites)
        lo = np.minimum(self.edges[:, 0], self.edges[:, 1]).astype(np.int64)
        hi = np.maximum(self.edges[:, 0], self.edges[:, 1]).astype(np.int64)
        self._ekey = lo * n + hi
        self._eord = np.argsort(self._ekey)
        self._ekey_sorted = self._ekey[self._eord]

    def edge_ids(self, a, b):
        """Vectorised (a,b) site pairs -> edge ids. -1 where the pair is not an edge."""
        n = len(self.sites)
        a = np.asarray(a, np.int64)
        b = np.asarray(b, np.int64)
        k = np.minimum(a, b) * n + np.maximum(a, b)
        pos = np.searchsorted(self._ekey_sorted, k)
        pos = np.clip(pos, 0, len(self._ekey_sorted) - 1)
        ok = self._ekey_sorted[pos] == k
        return np.where(ok, self._eord[pos], -1)

    def loop_edges(self, loop):
        """Closed loop of site ids -> its edge ids (invalid pairs dropped)."""
        loop = np.asarray(loop, np.int64)
        e = self.edge_ids(loop, np.roll(loop, -1))
        return e[e >= 0]

    @property
    def n_sites(self):
        return len(self.sites)

    @property
    def n_edges(self):
        return len(self.edges)

    def edge_midpoints(self):
        return 0.5 * (self.sites[self.edges[:, 0]] + self.sites[self.edges[:, 1]])

    def sample_box(self, rng):
        raise NotImplementedError

    def __repr__(self):
        return (f"<{self.name} pitch={self.pitch:.2f} sites={self.n_sites} "
                f"edges={self.n_edges} kinds={len(self.link_specs)}>")


# ==========================================================================================
class SquareLattice(Lattice):
    """The measured source geometry: isotropic square grid, 8-neighbour."""

    name = "square"

    def __init__(self, w, h, pitch, margin=1, diagonals=True):
        nx = int(np.floor(w / pitch)) + 2 * margin
        ny = int(np.floor(h / pitch)) + 2 * margin
        # Centre the lattice on the frame so the margin overhang is symmetric.
        ox = 0.5 * (w - (nx - 1) * pitch)
        oy = 0.5 * (h - (ny - 1) * pitch)
        jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        sites = np.stack([ox + ii.ravel() * pitch, oy + jj.ravel() * pitch], 1)
        self.nx, self.ny = nx, ny

        def sid(i, j):
            return j * nx + i

        d = np.sqrt(2.0)
        # (di, dj, angle_deg, length) - four directions generate every unique edge once.
        dirs = [(1, 0, 0.0, pitch), (0, 1, 90.0, pitch)]
        if diagonals:
            dirs += [(1, 1, 45.0, pitch * d), (1, -1, 135.0, pitch * d)]

        edges, kinds, specs = [], [], []
        for k, (di, dj, ang, ln) in enumerate(dirs):
            specs.append((ang, ln))
            i0 = np.arange(max(0, -di), nx - max(0, di))
            j0 = np.arange(max(0, -dj), ny - max(0, dj))
            J, I = np.meshgrid(j0, i0, indexing="ij")
            a = sid(I.ravel(), J.ravel())
            b = sid(I.ravel() + di, J.ravel() + dj)
            edges.append(np.stack([a, b], 1))
            kinds.append(np.full(len(a), k, np.int32))

        super().__init__(sites, np.concatenate(edges), np.concatenate(kinds), specs, pitch)

    def sample_box(self, rng):
        """An axis-aligned outlined rectangle spanning 1-3 cells - the clip's box glyph.

        The perimeter is walked cell by cell so every consecutive site pair is an orthogonal
        lattice edge; a rect corner-to-corner would not be.
        """
        bw = int(rng.integers(1, 4))
        bh = int(rng.integers(1, 4))
        i = int(rng.integers(0, max(1, self.nx - bw - 1)))
        j = int(rng.integers(0, max(1, self.ny - bh - 1)))
        top = [(i + k, j) for k in range(bw)]
        right = [(i + bw, j + k) for k in range(bh)]
        bot = [(i + bw - k, j + bh) for k in range(bw)]
        left = [(i, j + bh - k) for k in range(bh)]
        loop = [jj * self.nx + ii for ii, jj in top + right + bot + left]
        return self.loop_edges(loop)


# ==========================================================================================
class HexLattice(Lattice):
    """Hexagonal geometry. `pitch` is the hexagon EDGE length (== circumradius).

    honeycomb: sites are hexagon vertices (3 neighbours), cells are hexagons.
    triangular: sites are hex-packed centres (6 neighbours), cells are triangles.
    """

    name = "hex"

    def __init__(self, w, h, pitch, margin=1, mode="honeycomb"):
        self.mode = mode
        if mode == "triangular":
            sites, edges, kinds, specs = self._triangular(w, h, pitch, margin)
        elif mode == "honeycomb":
            sites, edges, kinds, specs = self._honeycomb(w, h, pitch, margin)
        else:
            raise ValueError(f"unknown hex mode {mode!r}")
        super().__init__(sites, edges, kinds, specs, pitch)

    # -- pointy-top hexagon cell centres covering the frame -----------------------------
    def _centres(self, w, h, R, margin):
        dx = np.sqrt(3.0) * R          # horizontal centre spacing
        dy = 1.5 * R                   # vertical centre spacing (rows interlock)
        nx = int(np.ceil(w / dx)) + 2 * margin + 1
        ny = int(np.ceil(h / dy)) + 2 * margin + 1
        j, i = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        cx = (i + 0.5 * (j & 1)) * dx
        cy = j * dy
        # Centre the whole tiling on the frame.
        cx = cx - 0.5 * (cx.max() + cx.min()) + 0.5 * w
        cy = cy - 0.5 * (cy.max() + cy.min()) + 0.5 * h
        self.nx, self.ny, self.dx, self.dy = nx, ny, dx, dy
        return np.stack([cx.ravel(), cy.ravel()], 1)

    def _honeycomb(self, w, h, R, margin):
        centres = self._centres(w, h, R, margin)
        self.centres = centres
        ang = np.deg2rad(90.0 + 60.0 * np.arange(6))          # pointy-top
        vx = centres[:, 0:1] + R * np.cos(ang)[None, :]
        vy = centres[:, 1:2] + R * np.sin(ang)[None, :]

        # Weld shared vertices: three hexagons meet at every honeycomb node.
        q = np.round(np.stack([vx.ravel(), vy.ravel()], 1) / (R * _WELD)).astype(np.int64)
        _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
        sites = np.stack([vx.ravel(), vy.ravel()], 1)[first]
        inv = inv.reshape(vx.shape)                            # (ncell, 6) site ids
        self.cell_loops = inv

        # Each hexagon contributes its 6 sides; duplicates removed by sorted-pair unique.
        a = inv
        b = np.roll(inv, -1, axis=1)
        pairs = np.stack([np.minimum(a, b).ravel(), np.maximum(a, b).ravel()], 1)
        edges = np.unique(pairs, axis=0)

        # A honeycomb has 3 distinct edge directions; classify by the actual segment angle.
        d = sites[edges[:, 1]] - sites[edges[:, 0]]
        deg = np.rad2deg(np.arctan2(d[:, 1], d[:, 0])) % 180.0
        specs, kinds = self._classify(deg, R)
        return sites, edges, kinds, specs

    def _triangular(self, w, h, R, margin):
        nx = int(np.ceil(w / R)) + 2 * margin + 1
        ny = int(np.ceil(h / (R * np.sqrt(3) / 2))) + 2 * margin + 1
        j, i = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
        x = (i + 0.5 * (j & 1)) * R
        y = j * R * np.sqrt(3) / 2
        x = x - 0.5 * (x.max() + x.min()) + 0.5 * w
        y = y - 0.5 * (y.max() + y.min()) + 0.5 * h
        sites = np.stack([x.ravel(), y.ravel()], 1)
        self.nx, self.ny = nx, ny
        self.centres = sites

        def sid(i, j):
            return j * nx + i

        edges = []
        # +x, and the two links into the row below - generates every unique edge once.
        for di_even, di_odd, dj in ((1, 1, 0), (-1, 0, 1), (0, 1, 1)):
            for jj in range(ny - dj):
                di = di_odd if (jj & 1) else di_even
                i0 = np.arange(max(0, -di), nx - max(0, di))
                if not len(i0):
                    continue
                edges.append(np.stack([sid(i0, jj), sid(i0 + di, jj + dj)], 1))
        edges = np.concatenate(edges)
        d = sites[edges[:, 1]] - sites[edges[:, 0]]
        deg = np.rad2deg(np.arctan2(d[:, 1], d[:, 0])) % 180.0
        specs, kinds = self._classify(deg, R)
        return sites, edges, kinds, specs

    @staticmethod
    def _classify(deg, length):
        """Bucket measured segment angles into the distinct stamps the atlas must bake."""
        uniq = np.unique(np.round(deg, 3))
        specs = [(float(u), float(length)) for u in uniq]
        kinds = np.searchsorted(uniq, np.round(deg, 3)).astype(np.int32)
        return specs, kinds

    def sample_box(self, rng):
        """The outlined-cell glyph: one hexagon ring. Direct analogue of the square rect.

        honeycomb  - a cell IS a hexagon, so its six sides are six lattice edges.
        triangular - the six NEIGHBOURS of an interior site form a hexagon whose sides are
                     also lattice edges; sorting them by angle makes the ring closed.
        """
        if self.mode == "honeycomb":
            c = int(rng.integers(0, len(self.cell_loops)))
            return self.loop_edges(self.cell_loops[c])
        for _ in range(24):
            s = int(rng.integers(0, self.n_sites))
            nb, _ = self.neighbours(s)
            if len(nb) == 6:
                d = self.sites[nb] - self.sites[s]
                order = np.argsort(np.arctan2(d[:, 1], d[:, 0]))
                return self.loop_edges(np.asarray(nb)[order])
        return np.zeros(0, np.int32)


def build(kind, w, h, pitch, **kw):
    if kind == "square":
        return SquareLattice(w, h, pitch, **kw)
    if kind in ("hex", "honeycomb"):
        return HexLattice(w, h, pitch, mode="honeycomb", **kw)
    if kind == "triangular":
        return HexLattice(w, h, pitch, mode="triangular", **kw)
    raise ValueError(f"unknown lattice {kind!r}")
