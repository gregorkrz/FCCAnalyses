#!/usr/bin/env python3
"""
Parse FCCAnalyses-style log lines like:
  ... Particle idx 101  direct daughters: [151, 152, 153]
and plot the full decay graph/tree.

Usage:
  python plot_decay_tree.py input.txt --out decay_tree.png
"""

import argparse
import re
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt

def parse_decay_file(path):
    """
    Returns:
      G: networkx.DiGraph with edges parent->daughter
      seen_nodes: set of all node IDs observed (parents and daughters)
    """
    pat = re.compile(r"Particle idx\s+(\d+)\s+direct daughters:\s*\[(.*?)\]")
    edges = set()
    seen_nodes = set()

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            parent = int(m.group(1))
            seen_nodes.add(parent)

            raw = m.group(2).strip()
            if raw == "" or raw == " ":
                continue  # Shouldn't happen, but guard anyway

            if raw == "":  # double-guard
                daughters = []
            elif raw == "[]":
                daughters = []
            else:
                # raw might be like "151, 152, 153" or empty
                # Strip brackets already removed by regex; handle empties
                parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
                daughters = [int(p) for p in parts] if parts else []

            for d in daughters:
                seen_nodes.add(d)
                edges.add((parent, d))

    G = nx.DiGraph()
    G.add_nodes_from(seen_nodes)
    G.add_edges_from(edges)
    return G, seen_nodes

def find_roots(G):
    """Nodes that never appear as daughters (in-degree 0)."""
    return [n for n in G.nodes if G.in_degree(n) == 0]

def compute_layout(G):
    """
    Try hierarchical layout via Graphviz 'dot'.
    Falls back to networkx spring layout if Graphviz not available.
    """
    # Try pygraphviz first
    try:
        from networkx.drawing.nx_agraph import graphviz_layout
        return graphviz_layout(G, prog="dot")  # top-down hierarchical
    except Exception:
        pass

    # Try pydot
    try:
        from networkx.drawing.nx_pydot import graphviz_layout
        return graphviz_layout(G, prog="dot")
    except Exception:
        pass

    # Fallback to spring layout
    return nx.spring_layout(G, seed=42, k=1 / (1 + len(G))**0.5)

def annotate_roots_and_leaves(G):
    roots = find_roots(G)
    leaves = [n for n in G.nodes if G.out_degree(n) == 0]
    return set(roots), set(leaves)

def plot_graph(G, out_path=None, title=None, figsize=(10, 8), dpi=150):
    pos = compute_layout(G)
    roots, leaves = annotate_roots_and_leaves(G)

    plt.figure(figsize=figsize, dpi=dpi)

    # Node styling by role
    node_colors = []
    node_sizes = []
    for n in G.nodes:
        if n in roots:
            node_colors.append("#4CAF50")  # green for roots
            node_sizes.append(500)
        elif n in leaves:
            node_colors.append("#E91E63")  # pink for leaves
            node_sizes.append(350)
        else:
            node_colors.append("#2196F3")  # blue for internal nodes
            node_sizes.append(400)

    # Draw edges and nodes
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=12, width=1.2, alpha=0.7)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, linewidths=0.8, edgecolors="black")
    nx.draw_networkx_labels(G, pos, labels={n: str(n) for n in G.nodes}, font_size=8)

    # Build a subtitle with quick stats
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    n_roots = len(roots)
    n_leaves = len(leaves)
    stats = f"nodes={n_nodes}, edges={n_edges}, roots={n_roots}, leaves={n_leaves}"

    plt.title(title or f"Decay Graph ({stats})")
    plt.axis("off")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
        print(f"Saved graph to: {out_path}")
    else:
        plt.show()
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Plot decay graph from FCCAnalyses debug log.")
    ap.add_argument("input", help="Path to input .txt log")
    ap.add_argument("--out", help="Output image path (e.g. decay_tree.png or .svg). If omitted, shows an interactive window.")
    ap.add_argument("--title", help="Custom plot title")
    ap.add_argument("--prune", type=int, default=None,
                    help="OPTIONAL: keep only the component containing this particle index (e.g. 101).")
    args = ap.parse_args()

    G, _ = parse_decay_file(args.input)

    if args.prune is not None:
        # Keep only the weakly connected component that contains the requested node
        if args.prune not in G:
            raise SystemExit(f"Particle {args.prune} not found in parsed graph.")
        # Weakly connected because edges are directed but we want component by ignoring direction
        comp_nodes = None
        for comp in nx.weakly_connected_components(G):
            if args.prune in comp:
                comp_nodes = comp
                break
        G = G.subgraph(comp_nodes).copy()

    # If multiple roots exist, this will draw a forest. That's okay.
    plot_graph(G, out_path=args.out, title=args.title)

if __name__ == "__main__":
    main()

