"""HMM + Viterbi route reconstruction on top of the classifier probabilities.

HIDDEN STATE   : which candidate arc the vehicle is on at GPS point t (the K candidates of that point).
                 State sets differ per time step; every state keeps its arc id, so the candidate->road
                 correspondence is preserved.
EMISSION       : w_emit * log P_model(arc is correct | features)  - the classifier's probability.
TRANSITION     : a PENALTY (log-space, <= 0) for going from arc a (t-1) to arc b (t), using the road graph:
                    same arc                          0
                    b reachable from a in h hops      -alpha * h         (h = 1..MAX_HOPS; 1 = directly connected)
                    b is the reverse twin of a        -uturn_pen         (a U-turn: rare)
                    otherwise (disconnected jump)     -jump_pen
                 After a time gap > gap_s the car may have travelled anywhere, so connectivity is not
                 informative: all transitions cost 0 and the emission decides (a soft "reset").
VITERBI        : delta_t(b) = emission_t(b) + max_a [ delta_{t-1}(a) + trans(a->b) ], with back-pointers.
MISSING        : a point with no candidates is skipped; its output is filled with the previous decision.

Why not make disconnected jumps impossible (-inf)?  The reference route itself has a few breaks, candidates can be
missing (0.5%), and a hard constraint would make the whole decode fail. A large but finite penalty lets the path
recover while still strongly preferring connected routes.
"""
from dataclasses import dataclass, asdict
import numpy as np

MAX_HOPS = 4


@dataclass
class HMMParams:
    w_emit: float = 1.0
    alpha: float = 0.3        # cost per graph hop
    jump_pen: float = 4.0     # cost of an unconnected jump
    uturn_pen: float = 4.0
    gap_s: float = 10.0
    eps: float = 1e-4

    def to_dict(self): return asdict(self)


class RoadGraph:
    """Directed arc graph with cached hop-reachability: arcs reachable from arc a in 1..MAX_HOPS steps."""
    def __init__(self, arcs):
        self.arcs = arcs
        n_nodes = int(arcs.max()) + 1
        order = np.argsort(arcs[:, 0], kind="stable")
        self.order = order
        self.ptr = np.r_[0, np.cumsum(np.bincount(arcs[:, 0], minlength=n_nodes))]
        self._cache = {}
        self._tcache = {}

    def code_matrix(self, prev_arcs, arcs):
        """int8 matrix of transition 'codes' (parameter-free, so cached across decoder settings):
        0 same arc, 1..MAX_HOPS graph hops, -1 U-turn (reverse twin), -2 disconnected jump."""
        key = (prev_arcs.tobytes(), arcs.tobytes())
        c = self._tcache.get(key)
        if c is None:
            c = np.full((len(prev_arcs), len(arcs)), -2, dtype=np.int8)
            for ia, a in enumerate(prev_arcs):
                a = int(a); r = self.reach(a)
                for ib, b in enumerate(arcs):
                    b = int(b)
                    if b == a: c[ia, ib] = 0
                    elif b in r: c[ia, ib] = r[b]
                    elif self.is_twin(a, b): c[ia, ib] = -1
            if len(self._tcache) > 200000: self._tcache.clear()
            self._tcache[key] = c
        return c

    def out_arcs(self, node):
        if node + 1 >= len(self.ptr): return self.order[0:0]
        return self.order[self.ptr[node]:self.ptr[node + 1]]

    def reach(self, a):
        r = self._cache.get(a)
        if r is None:
            r = {}; frontier = [a]
            for h in range(1, MAX_HOPS + 1):
                nxt = []
                for x in frontier:
                    for y in self.out_arcs(int(self.arcs[x, 1])):
                        y = int(y)
                        if y not in r and y != a:
                            r[y] = h; nxt.append(y)
                frontier = nxt
            self._cache[a] = r
        return r

    def is_twin(self, a, b):
        return self.arcs[a, 0] == self.arcs[b, 1] and self.arcs[a, 1] == self.arcs[b, 0]


def viterbi(cand_arcs, cand_logp, t, graph, prm: HMMParams):
    """cand_arcs / cand_logp: lists (len T) of arrays; t: (T,) seconds. Returns (T,) predicted arc ids (-1 = none)."""
    T = len(cand_arcs)
    out = np.full(T, -1, dtype=np.int64)
    live = [i for i in range(T) if len(cand_arcs[i])]
    if not live: return out
    back, prev_arcs, delta, prev_i = [], None, None, None
    steps = []
    for i in live:
        arcs, em = cand_arcs[i], prm.w_emit * cand_logp[i]
        if delta is None:
            delta = em.copy(); back.append(None)
        else:
            gap = (t[i] - t[prev_i]) > prm.gap_s
            K0, K1 = len(prev_arcs), len(arcs)
            if gap:
                trans = np.zeros((K0, K1))
            else:
                lut = np.array([-prm.jump_pen, -prm.uturn_pen, 0.0] + [-prm.alpha * h for h in range(1, MAX_HOPS + 1)])
                trans = lut[graph.code_matrix(prev_arcs, arcs).astype(np.int64) + 2]
            score = delta[:, None] + trans
            arg = score.argmax(0)
            delta = score[arg, np.arange(K1)] + em
            back.append(arg)
        prev_arcs, prev_i = arcs, i
        steps.append(arcs)
    # backtrack
    k = int(delta.argmax()); path = [0] * len(live)
    for s in range(len(live) - 1, -1, -1):
        path[s] = int(steps[s][k])
        if back[s] is not None: k = int(back[s][k])
    out[live] = path
    last = -1
    for i in range(T):                      # fill skipped points with the previous decision
        if out[i] == -1: out[i] = last
        else: last = out[i]
    return out


def group_candidates(pt, arc, p, T, eps):
    """Flat candidate arrays -> per-point lists (sorted by pt)."""
    order = np.argsort(pt, kind="stable"); pt, arc, p = pt[order], arc[order], p[order]
    bounds = np.searchsorted(pt, np.arange(T + 1))
    arcs = [arc[bounds[i]:bounds[i + 1]] for i in range(T)]
    lp = [np.log(np.clip(p[bounds[i]:bounds[i + 1]], eps, 1.0)) for i in range(T)]
    return arcs, lp


def argmax_arcs(pt, arc, p, T):
    """Classifier-only decision: highest-probability candidate per point (-1 if none)."""
    out = np.full(T, -1, dtype=np.int64); best = np.full(T, -1.0)
    for a_, b_, c_ in zip(pt, arc, p):
        if c_ > best[a_]: best[a_] = c_; out[a_] = b_
    return out
