"""network.py - the lattice as a set of CIRCUITS rather than a field of loose dots.

Ethan's note: "There should be distinct connected nodes (no more than 10 connected nodes,
with a few 2 or 3 node (and some 1) branches on the 10 node chain), so they shouldn't all
flicker independently. This needs to be a living, interconnected, changing network of hexes.
A lot of hexes shouldn't be connected still, but 20-30% should be in some chain somewhere."

Two things follow, and the second is the important one:

1. **Structure.** Chains are grown as walks on the lattice graph, capped at `spine` nodes,
   with a few short branches hanging off each spine. Around a quarter of all sites end up in
   a chain; the rest stay loose.

2. **Shared scheduling.** Every node and edge in a chain lights on ONE schedule - the chain
   is the unit that turns on and off, not the node. That is what makes it read as a circuit
   energising rather than as confetti. Per-element independent flicker was carrying the whole
   colour effect before this, and it looked like noise precisely because nothing was
   connected to anything.

A chain's schedule is a slow asymmetric pulse: chains spend most of their time off, come up
fast and decay slower, so at any instant a minority are lit and the frame keeps its structure.
"""
from __future__ import annotations

import numpy as np


class Network:
    def __init__(self, lat, seed=0, coverage=0.26, spine=10, branches=(2, 3),
                 n_branches=(1, 3), singles=0.30, period=(1.6, 5.5), duty=0.34,
                 rise=0.10, fall=0.42):
        self.lat = lat
        rng = np.random.default_rng(seed + 30011)
        n, m = lat.n_sites, lat.n_edges

        # adjacency: site -> list of (neighbour, edge index)
        adj = [[] for _ in range(n)]
        for e, (a, b) in enumerate(lat.edges):
            adj[a].append((b, e))
            adj[b].append((a, e))

        self.site_chain = np.full(n, -1, np.int32)
        self.edge_chain = np.full(m, -1, np.int32)
        target = int(coverage * n)
        cid = 0
        order = rng.permutation(n)
        oi = 0
        used = 0
        while used < target and oi < n:
            start = int(order[oi]); oi += 1
            if self.site_chain[start] >= 0:
                continue
            # A `singles` fraction are lone nodes - the note asks for some 1-node members.
            if rng.random() < singles:
                self.site_chain[start] = cid
                used += 1
                cid += 1
                continue
            body = self._grow(adj, start, rng.integers(4, spine + 1), rng, cid)
            if not body:
                continue
            for _ in range(int(rng.integers(n_branches[0], n_branches[1] + 1))):
                root = int(body[int(rng.integers(0, len(body)))])
                self._grow(adj, root, int(rng.integers(branches[0], branches[1] + 1)),
                           rng, cid, seed_is_member=True)
            used += int((self.site_chain == cid).sum())
            cid += 1
        self.n_chains = cid

        # -- per-chain schedule -----------------------------------------------------------
        self.period = rng.uniform(period[0], period[1], max(cid, 1)).astype(np.float32)
        self.phase = rng.random(max(cid, 1)).astype(np.float32)
        self.duty = float(duty)
        self.rise, self.fall = float(rise), float(fall)
        # loose (unchained) sites keep a slow individual shimmer, well below the chains
        self.loose_ph = rng.random(n).astype(np.float32)
        self.loose_pe = rng.uniform(2.4, 7.0, n).astype(np.float32)
        self.loose_ph_e = rng.random(m).astype(np.float32)
        self.loose_pe_e = rng.uniform(2.4, 7.0, m).astype(np.float32)

        # BORDER FLASH. Ethan liked that the glitch momentarily illuminates unlit nodes, and
        # asked for the same thing to happen at the edges of lit regions. So: every UNCHAINED
        # site that touches a chain remembers which chain, and occasionally lights with it -
        # the circuit bleeding into the dark lattice around it. Sparse and short, or it just
        # thickens every chain into a blob.
        self.border_of = np.full(n, -1, np.int32)
        for e, (a, b) in enumerate(lat.edges):
            if self.site_chain[a] >= 0 and self.site_chain[b] < 0:
                self.border_of[b] = self.site_chain[a]
            elif self.site_chain[b] >= 0 and self.site_chain[a] < 0:
                self.border_of[a] = self.site_chain[b]
        self.flash_ph = rng.random(n).astype(np.float32)
        self.flash_pe = rng.uniform(0.9, 3.4, n).astype(np.float32)

    def _grow(self, adj, start, want, rng, cid, seed_is_member=False):
        """Random walk claiming unclaimed sites for chain `cid`. Returns the sites taken."""
        if not seed_is_member and self.site_chain[start] >= 0:
            return []
        cur = start
        if not seed_is_member:
            self.site_chain[start] = cid
        out = [start]
        for _ in range(int(want) - 1):
            opts = [(nb, e) for nb, e in adj[cur] if self.site_chain[nb] < 0]
            if not opts:
                break
            nb, e = opts[int(rng.integers(0, len(opts)))]
            self.site_chain[nb] = cid
            self.edge_chain[e] = cid
            out.append(nb)
            cur = nb
        return out

    def chain_level(self, t):
        """(n_chains,) in [0,1] - the pulse each chain is currently at."""
        u = np.mod(t / self.period + self.phase, 1.0)
        # fast rise, slower fall, then dark for the rest of the cycle
        lv = np.where(u < self.rise, u / self.rise,
                      np.where(u < self.rise + self.fall,
                               1.0 - (u - self.rise) / self.fall, 0.0))
        return np.clip(lv, 0.0, 1.0).astype(np.float32)

    def levels(self, t, loose_gain=0.30, loose_depth=0.55, border_flash=0.0,
               border_duty=0.16):
        """(site_level, edge_level), both (N,) in [0,~1].

        Chained elements take their CHAIN's level - they light and die together. Loose ones
        get a weak independent shimmer so the un-wired lattice is present but clearly
        subordinate; without that separation everything reads as one uniform texture.
        """
        cl = self.chain_level(t)
        cl = np.concatenate([cl, np.zeros(1, np.float32)])   # index -1 -> 0
        s = cl[self.site_chain]
        e = cl[self.edge_chain]
        loose_s = loose_gain * (0.5 + 0.5 * np.sin(
            2 * np.pi * (t / self.loose_pe + self.loose_ph))) ** 2
        loose_e = loose_gain * (0.5 + 0.5 * np.sin(
            2 * np.pi * (t / self.loose_pe_e + self.loose_ph_e))) ** 2
        s = np.where(self.site_chain >= 0, s, loose_s * loose_depth)
        e = np.where(self.edge_chain >= 0, e, loose_e * loose_depth)

        # border flash: a short spike, gated on the neighbouring chain actually being lit
        if border_flash > 1e-4:
            u = np.mod(t / self.flash_pe + self.flash_ph, 1.0)
            spike = np.clip(1.0 - u / max(border_duty, 1e-6), 0.0, 1.0) ** 2
            nb = cl[self.border_of]
            s = np.maximum(s, border_flash * spike * nb *
                           (self.border_of >= 0) * (self.site_chain < 0))
        return s.astype(np.float32), e.astype(np.float32)

    def report(self):
        n = len(self.site_chain)
        inch = int((self.site_chain >= 0).sum())
        sizes = np.bincount(self.site_chain[self.site_chain >= 0],
                            minlength=max(self.n_chains, 1))
        sizes = sizes[sizes > 0]
        return (f"chains {self.n_chains}  sites in a chain {inch}/{n} "
                f"({inch/max(n,1):.1%})  size min/med/max "
                f"{sizes.min()}/{int(np.median(sizes))}/{sizes.max()}  "
                f"edges wired {int((self.edge_chain >= 0).sum())}")
