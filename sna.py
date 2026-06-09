#!/usr/bin/env python3
"""
Social Network Analysis on Instagram comment data.

Builds a directed graph from:
  - Comment interactions (user -> post)
  - Reply edges (replier -> original commenter)
  - @mention edges (mentioner -> mentioned user)

Computes centrality metrics, community detection, and exports to GEXF for Gephi.

Usage
-----
python sna.py
python sna.py --input data/comments_reels_2026-06-06_03-27-47-199.json
python sna.py --input data/comments.json --output results/
"""

import json
import re
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import networkx as nx
import community as community_louvain
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_FILE = "data/comments_reels_2026-06-06_03-27-47-199.json"
OUTPUT_DIR = Path("results")
POST_OWNER = "__POST__"  # virtual node representing the Instagram post


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(data: list) -> nx.DiGraph:
    G = nx.DiGraph()

    # All commenters connect to the post node
    for c in data:
        u = c.get("ownerUsername", "")
        if u:
            G.add_node(u, type="user")
            if G.has_edge(u, POST_OWNER):
                G[u][POST_OWNER]["weight"] += 1
            else:
                G.add_edge(u, POST_OWNER, weight=1, type="comment")

    G.add_node(POST_OWNER, type="post")

    # Reply edges: replier -> original commenter
    for c in data:
        dst = c.get("ownerUsername", "")
        for r in c.get("replies", []):
            src = r.get("ownerUsername", "")
            if src and dst and src != dst:
                if G.has_edge(src, dst):
                    G[src][dst]["weight"] += 1
                else:
                    G.add_edge(src, dst, weight=1, type="reply")

    # @mention edges from comment and reply text
    all_comments = data + [r for c in data for r in c.get("replies", [])]
    for c in all_comments:
        src = c.get("ownerUsername", "")
        for mention in re.findall(r"@([\w.]+)", c.get("text", "")):
            if mention != src and G.has_node(mention):
                if G.has_edge(src, mention):
                    G[src][mention]["weight"] += 1
                else:
                    G.add_edge(src, mention, weight=1, type="mention")

    return G


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse(G: nx.DiGraph) -> dict:
    U = G.to_undirected()
    components = list(nx.connected_components(U))
    largest_nodes = max(components, key=len)
    largest = U.subgraph(largest_nodes).copy()

    degree_cent   = nx.degree_centrality(largest)
    between_cent  = nx.betweenness_centrality(largest, normalized=True)
    in_deg        = dict(G.in_degree())
    out_deg       = dict(G.out_degree())

    partition     = community_louvain.best_partition(largest)
    communities   = Counter(partition.values())

    # Group members per community
    comm_members = defaultdict(list)
    for node, cid in partition.items():
        if node != POST_OWNER:
            comm_members[cid].append((node, degree_cent.get(node, 0)))

    return {
        "graph": G,
        "undirected": U,
        "largest": largest,
        "components": components,
        "degree_cent": degree_cent,
        "between_cent": between_cent,
        "in_deg": in_deg,
        "out_deg": out_deg,
        "partition": partition,
        "communities": communities,
        "comm_members": comm_members,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(r: dict):
    G          = r["graph"]
    largest    = r["largest"]
    components = r["components"]
    dc         = r["degree_cent"]
    bc         = r["between_cent"]
    in_d       = r["in_deg"]
    out_d      = r["out_deg"]
    communities = r["communities"]
    comm_members = r["comm_members"]

    print("=" * 55)
    print("  SOCIAL NETWORK ANALYSIS — INSTAGRAM COMMENTS")
    print("=" * 55)

    print("\n--- GRAPH OVERVIEW ---")
    print(f"  Nodes (users)       : {G.number_of_nodes() - 1}")  # exclude POST node
    print(f"  Edges (interactions): {G.number_of_edges()}")
    print(f"  Density             : {nx.density(G):.6f}")
    print(f"  Connected components: {len(components)}")
    print(f"  Largest component   : {len(max(components, key=len))} nodes")

    def top(d, n=10, exclude=POST_OWNER):
        return [(k, v) for k, v in sorted(d.items(), key=lambda x: -x[1]) if k != exclude][:n]

    print("\n--- TOP 10 BY DEGREE CENTRALITY (most connected) ---")
    for u, s in top(dc):
        print(f"  {u:<35} {s:.4f}")

    print("\n--- TOP 10 BY BETWEENNESS (bridges between groups) ---")
    for u, s in top(bc):
        print(f"  {u:<35} {s:.4f}")

    print("\n--- TOP 10 BY IN-DEGREE (most replied / mentioned) ---")
    for u, s in top({n: in_d.get(n, 0) for n in largest.nodes}):
        print(f"  {u:<35} {s}")

    print("\n--- TOP 10 BY OUT-DEGREE (most active repliers) ---")
    for u, s in top({n: out_d.get(n, 0) for n in largest.nodes}):
        print(f"  {u:<35} {s}")

    print(f"\n--- COMMUNITIES (Louvain) ---")
    print(f"  Number of communities : {len(communities)}")
    print(f"  Sizes                 : {sorted(communities.values(), reverse=True)[:15]}")

    print("\n  Top 5 communities and their key members:")
    for cid, size in sorted(communities.items(), key=lambda x: -x[1])[:5]:
        members = [m[0] for m in sorted(comm_members[cid], key=lambda x: -x[1])[:5]]
        print(f"  Community {cid:>2} ({size:>4} members): {members}")

    print()


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_graph(r: dict, output_dir: Path):
    largest   = r["largest"]
    partition = r["partition"]
    dc        = r["degree_cent"]

    # Only show interaction subgraph (exclude isolated comment-only nodes)
    interaction_nodes = [
        n for n in largest.nodes
        if largest.degree(n) > 1 or n == POST_OWNER
    ]
    sub = largest.subgraph(interaction_nodes).copy()

    # Color by community
    unique_comms = list(set(partition.get(n, -1) for n in sub.nodes))
    cmap = plt.cm.get_cmap("tab20", len(unique_comms))
    color_map = {c: cmap(i) for i, c in enumerate(unique_comms)}
    node_colors = [color_map[partition.get(n, -1)] for n in sub.nodes]

    # Size by degree centrality
    node_sizes = [max(50, dc.get(n, 0) * 5000) for n in sub.nodes]

    fig, ax = plt.subplots(figsize=(16, 12))
    pos = nx.spring_layout(sub, seed=42, k=0.4)

    nx.draw_networkx_edges(sub, pos, alpha=0.2, edge_color="#aaaaaa", ax=ax)
    nx.draw_networkx_nodes(sub, pos, node_color=node_colors,
                           node_size=node_sizes, alpha=0.85, ax=ax)

    # Label only high-degree nodes
    labels = {n: n for n in sub.nodes if dc.get(n, 0) > 0.003}
    nx.draw_networkx_labels(sub, pos, labels, font_size=7, ax=ax)

    ax.set_title("Instagram Comment Interaction Network\n(node size = degree centrality, colour = community)",
                 fontsize=13)
    ax.axis("off")

    out = output_dir / "network_graph.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Graph image saved : {out}")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_gexf(G: nx.DiGraph, partition: dict, dc: dict, bc: dict, output_dir: Path):
    """Export to GEXF format for Gephi."""
    E = G.copy()
    for node in E.nodes:
        E.nodes[node]["community"]   = partition.get(node, -1)
        E.nodes[node]["degree_cent"] = round(dc.get(node, 0), 6)
        E.nodes[node]["betweenness"] = round(bc.get(node, 0), 6)
        E.nodes[node]["label"]       = node
    out = output_dir / "network.gexf"
    nx.write_gexf(E, str(out))
    print(f"  GEXF exported     : {out}")


def export_csv(G: nx.DiGraph, dc: dict, bc: dict, in_d: dict, out_d: dict,
               partition: dict, output_dir: Path):
    """Export node metrics and edge list as CSV."""
    import csv

    # Nodes
    nodes_out = output_dir / "nodes.csv"
    with open(nodes_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["username", "community", "degree_centrality",
                    "betweenness_centrality", "in_degree", "out_degree"])
        for node in sorted(G.nodes):
            if node == POST_OWNER:
                continue
            w.writerow([
                node,
                partition.get(node, -1),
                round(dc.get(node, 0), 6),
                round(bc.get(node, 0), 6),
                in_d.get(node, 0),
                out_d.get(node, 0),
            ])
    print(f"  Nodes CSV saved   : {nodes_out}")

    # Edges
    edges_out = output_dir / "edges.csv"
    with open(edges_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "target", "weight", "type"])
        for src, dst, attrs in G.edges(data=True):
            if src == POST_OWNER or dst == POST_OWNER:
                continue
            w.writerow([src, dst, attrs.get("weight", 1), attrs.get("type", "")])
    print(f"  Edges CSV saved   : {edges_out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SNA on Instagram comment JSON.")
    parser.add_argument("--input", "-i", default=DATA_FILE,
                        help="Path to comments JSON file")
    parser.add_argument("--output", "-o", type=Path, default=OUTPUT_DIR,
                        help="Output directory for results")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading {args.input} …")
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))

    print("Building graph …")
    G = build_graph(data)

    print("Running analysis …\n")
    results = analyse(G)

    print_report(results)

    print("Exporting results …")
    export_gexf(G, results["partition"], results["degree_cent"],
                results["between_cent"], args.output)
    export_csv(G, results["degree_cent"], results["between_cent"],
               results["in_deg"], results["out_deg"],
               results["partition"], args.output)

    print("Generating graph image …")
    plot_graph(results, args.output)

    print("\nDone. Results saved to:", args.output)


if __name__ == "__main__":
    main()
