"""
Member 2 EDA: road network analysis.
Run from the repo root:  python -m data_processing.eda_network
Uses the TRAIN partition only.
Output: results/eda_network.csv, results/eda_network_*.png
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from data_processing.load import load_trip, list_trips
from utils.geo import projected_distance_m

TRAIN_DIR = Path("data/train")


def network_stats(nodes, arcs):
    n_nodes, n_arcs = len(nodes), len(arcs)
    frm, to = arcs["from_node"].to_numpy(), arcs["to_node"].to_numpy()

    bad_ref = (frm < 0) | (frm >= n_nodes) | (to < 0) | (to >= n_nodes)
    ok = ~bad_ref
    frm_ok, to_ok = frm[ok], to[ok]

    used = np.zeros(n_nodes, dtype=bool)
    used[frm_ok] = True
    used[to_ok] = True

    # two-way roads: arc (a,b) whose reverse (b,a) also exists
    pairs = set(zip(frm_ok.tolist(), to_ok.tolist()))
    two_way = sum((b, a) in pairs for a, b in pairs)

    # arc lengths in metres
    lon, lat = nodes["lon"].to_numpy(), nodes["lat"].to_numpy()
    length = projected_distance_m(lon[frm_ok], lat[frm_ok], lon[to_ok], lat[to_ok])

    # connectivity
    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))
    G.add_edges_from(zip(frm_ok.tolist(), to_ok.tolist()))
    wcc = [len(c) for c in nx.weakly_connected_components(G)]
    scc = [len(c) for c in nx.strongly_connected_components(G)]
    out_deg = np.array([d for _, d in G.out_degree()])
    in_deg = np.array([d for _, d in G.in_degree()])

    return {
        "n_nodes": n_nodes,
        "n_arcs": n_arcs,
        "n_invalid_node_refs": int(bad_ref.sum()),
        "n_isolated_nodes": int((~used).sum()),
        "n_self_loops": int((frm_ok == to_ok).sum()),
        "n_duplicate_arcs": int(len(frm_ok) - len(pairs)),
        "pct_two_way": round(100 * two_way / max(len(pairs), 1), 2),
        "n_weak_components": len(wcc),
        "largest_wcc_frac": round(max(wcc) / n_nodes, 4),
        "n_strong_components": len(scc),
        "largest_scc_frac": round(max(scc) / n_nodes, 4),
        "n_dead_ends": int(((out_deg == 0) & (in_deg > 0)).sum()),
        "mean_out_degree": round(float(out_deg.mean()), 3),
        "arc_len_mean_m": round(float(length.mean()), 2),
        "arc_len_median_m": round(float(np.median(length)), 2),
        "arc_len_max_m": round(float(length.max()), 2),
        "n_zero_len_arcs": int((length < 0.01).sum()),
    }, length


def main():
    os.makedirs("results", exist_ok=True)
    rows, all_len = [], []
    for trip_dir in list_trips(TRAIN_DIR):
        trip = load_trip(trip_dir, load_route=False)
        stats, length = network_stats(trip["nodes"], trip["arcs"])
        rows.append({"trip_id": trip_dir.name, **stats})
        all_len.append(length)
        print(f"{trip_dir.name}: nodes={stats['n_nodes']} arcs={stats['n_arcs']} "
              f"wcc={stats['n_weak_components']} two_way={stats['pct_two_way']}%")

    df = pd.DataFrame(rows)
    df.to_csv("results/eda_network.csv", index=False)
    print("\nSaved results/eda_network.csv")
    print(df.describe().T)

    # 1. Network size per trip
    plt.figure(figsize=(10, 4))
    plt.bar(df["trip_id"], df["n_arcs"])
    plt.yscale("log"); plt.xticks(rotation=90, fontsize=6)
    plt.ylabel("Arcs (log)"); plt.title("Road segments per trip (train)")
    plt.tight_layout(); plt.savefig("results/eda_network_size.png", dpi=150); plt.close()

    # 2. Arc length distribution
    L = np.concatenate(all_len)
    plt.figure(figsize=(7, 4))
    plt.hist(np.clip(L, 0, 300), bins=100)
    plt.xlabel("Arc length (m, clipped at 300)"); plt.ylabel("Arcs")
    plt.title("Road segment lengths (train)")
    plt.tight_layout(); plt.savefig("results/eda_network_arc_len.png", dpi=150); plt.close()

    # 3. Connectivity
    plt.figure(figsize=(7, 4))
    plt.scatter(df["n_nodes"], df["largest_scc_frac"], s=12)
    plt.xscale("log"); plt.xlabel("Nodes (log)"); plt.ylabel("Largest strongly connected component (fraction)")
    plt.title("Network connectivity per trip (train)")
    plt.tight_layout(); plt.savefig("results/eda_network_connectivity.png", dpi=150); plt.close()


if __name__ == "__main__":
    main()