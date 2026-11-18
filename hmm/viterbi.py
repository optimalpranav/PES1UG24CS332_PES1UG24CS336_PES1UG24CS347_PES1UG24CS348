"""
HMM + Viterbi route reconstruction (map matching).

Hidden states : the candidate arcs of each GPS point (rows of the candidate table).
Emissions     : log P(candidate is correct) from the trained classifier.
Transitions   : Newson & Krumm (2009) style. A move from candidate a (point t) to
                candidate b (point t+1) is likely when the driving distance along
                the road network is close to the straight-line distance between
                the two GPS points:
                    log P(b | a) = -|d_route - d_gps| / beta
                Moves that are impossible on the network (no path within the
                search limit, or driving backwards along an arc) get -inf.
Breaks        : if no candidate at t+1 is reachable from any candidate at t,
                the chain is restarted at t+1 (emission only) and counted.
GPS gaps      : beta and the search limit grow with the time gap dt.
"""
import heapq
from collections import defaultdict

import numpy as np
import pandas as pd
from features.candidates import arc_geometry
from utils.geo import lonlat_to_xy

EPS = 1e-9


class RoadGraph:
    """Directed node graph with arc lengths, plus a cached bounded Dijkstra."""

    def __init__(self, nodes, arcs):
        ax, ay, bx, by, length, _ = arc_geometry(nodes, arcs)
        self.from_node = arcs["from_node"].to_numpy()
        self.to_node = arcs["to_node"].to_numpy()
        self.length = length
        self.adj = defaultdict(list)
        for f, t, l in zip(self.from_node, self.to_node, length):
            self.adj[f].append((t, l))
        self._cache = {}   # source node -> (limit used, {node: dist})

    def dists_from(self, src, limit):
        """Shortest driving distance from node src to every node within `limit` metres."""
        cached = self._cache.get(src)
        if cached is not None and cached[0] >= limit:
            return cached[1]
        dist = {src: 0.0}
        heap = [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, np.inf) or d > limit:
                continue
            for v, w in self.adj[u]:
                nd = d + w
                if nd <= limit and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        self._cache[src] = (limit, dist)
        return dist

    def route_distance(self, a, fa, b, fb, limit, back_tol):
        """Driving distance from position fa on arc a to position fb on arc b (metres)."""
        if a == b:
            delta = (fb - fa) * self.length[a]
            return max(delta, 0.0) if delta >= -back_tol else np.inf
        rest_a = (1.0 - fa) * self.length[a]
        into_b = fb * self.length[b]
        budget = limit - rest_a - into_b
        if budget < 0:
            return np.inf
        d_nodes = self.dists_from(self.to_node[a], limit)
        mid = d_nodes.get(self.from_node[b], np.inf)
        return rest_a + mid + into_b if mid <= budget else np.inf


def viterbi_trip(cands, prob, track, graph, beta=10.0, emission_weight=1.0,
                 back_tol=15.0, min_limit=100.0, limit_factor=3.0):
    """
    Decode one trip.
    cands : candidate rows of ONE trip (needs obs_idx, arc_id, proj_frac)
    prob  : classifier probability for each row of cands (same order)
    track : the trip's GPS track (lon, lat, t)
    Returns (dict obs_idx -> decoded arc_id, number of chain breaks).
    """
    # Same point ordering as candidates.py (obs_idx indexes this track)
    track = track.drop_duplicates("t").sort_values("t").reset_index(drop=True)
    px, py = lonlat_to_xy(track["lon"].to_numpy(), track["lat"].to_numpy())
    t = track["t"].to_numpy()

    cands = cands.assign(_logp=emission_weight * np.log(np.clip(prob, EPS, 1.0)))
    groups = [g for _, g in cands.groupby("obs_idx", sort=True)]

    obs = [int(g["obs_idx"].iloc[0]) for g in groups]
    arcs_t = [g["arc_id"].to_numpy() for g in groups]
    frac_t = [g["proj_frac"].to_numpy() for g in groups]
    emit_t = [g["_logp"].to_numpy() for g in groups]

    score = emit_t[0].copy()
    scores = [score]                      # score vector after each step
    backptr = [np.full(len(arcs_t[0]), -1)]
    breaks = 0

    for k in range(1, len(groups)):
        i0, i1 = obs[k - 1], obs[k]
        d_gps = float(np.hypot(px[i1] - px[i0], py[i1] - py[i0]))
        dt = max(float(t[i1] - t[i0]), 1.0)
        beta_k = beta * dt                        # more lenient over longer gaps
        limit = max(min_limit, limit_factor * d_gps)

        prev_a, prev_f = arcs_t[k - 1], frac_t[k - 1]
        cur_a, cur_f = arcs_t[k], frac_t[k]
        trans = np.full((len(prev_a), len(cur_a)), -np.inf)
        for p, (a, fa) in enumerate(zip(prev_a, prev_f)):
            for c, (b, fb) in enumerate(zip(cur_a, cur_f)):
                d_route = graph.route_distance(a, fa, b, fb, limit, back_tol)
                if np.isfinite(d_route):
                    trans[p, c] = -abs(d_route - d_gps) / beta_k

        total = score[:, None] + trans
        best_prev = np.argmax(total, axis=0)
        best_val = total[best_prev, np.arange(len(cur_a))]

        if not np.isfinite(best_val).any():
            # no connected move from any candidate: restart the chain here
            breaks += 1
            score = emit_t[k].copy()
            backptr.append(np.full(len(cur_a), -1))
        else:
            score = best_val + emit_t[k]
            backptr.append(np.where(np.isfinite(best_val), best_prev, -2))
        scores.append(score)

    # Backtrack. -1 = start of a chain segment (jump to the best end of the
    # previous segment); -2 = state unreachable (never on the best path).
    path = np.empty(len(groups), dtype=np.int64)
    j = int(np.argmax(scores[-1]))
    for k in range(len(groups) - 1, -1, -1):
        path[k] = arcs_t[k][j]
        if k == 0:
            break
        bp = int(backptr[k][j])
        j = bp if bp >= 0 else int(np.argmax(scores[k - 1]))
    return dict(zip(obs, path.tolist())), breaks

def _best_beta(default=10.0):
    """Beta chosen by hmm/run_hmm.py on student_test (results/hmm_beta_search.csv)."""
    try:
        s = pd.read_csv("results/hmm_beta_search.csv")
        return float(s.loc[s["hmm_top1_acc"].idxmax(), "beta"])
    except (FileNotFoundError, KeyError, ValueError):
        return default


BETA = _best_beta()


def decode(cands, trip):
    """Interface used by api/decoder.py.
    cands: candidate rows of one trip with a 'prob' column; trip: track/nodes/arcs (no route).
    Returns pd.Series index=obs_idx, value=decoded arc_id."""
    graph = RoadGraph(trip["nodes"], trip["arcs"])
    mapping, _ = viterbi_trip(cands, cands["prob"].to_numpy(), trip["track"], graph, beta=BETA)
    return pd.Series(mapping).sort_index()