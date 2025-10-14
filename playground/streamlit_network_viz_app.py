import re
import io
import json
import networkx as nx
import streamlit as st
from pyvis.network import Network

PATTERN = re.compile(r"Particle idx\s+(\d+)\s+direct daughters:\s*\[(.*?)\]")

def parse_bytes(content: bytes):
    edges, seen = set(), set()
    for line in io.StringIO(content.decode("utf-8", errors="ignore")):
        m = PATTERN.search(line)
        if not m:
            continue
        parent = int(m.group(1)); seen.add(parent)
        raw = (m.group(2) or "").strip()
        if raw == "":
            daughters = []
        else:
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            daughters = [int(p) for p in parts] if parts else []
        for d in daughters:
            seen.add(d)
            edges.add((parent, d))
    G = nx.DiGraph()
    G.add_nodes_from(seen)
    G.add_edges_from(edges)
    return G

def roots_and_leaves(G):
    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    leaves = [n for n in G.nodes if G.out_degree(n) == 0]
    return set(roots), set(leaves)

def component_with(G, node):
    for comp in nx.weakly_connected_components(G):
        if node in comp:
            return G.subgraph(comp).copy()
    return None

st.set_page_config(page_title="Decay Graph Explorer", layout="wide")
st.title("Decay Graph Explorer")

uploaded = st.file_uploader("Upload your log .txt", type=["txt"])
col1, col2, col3 = st.columns(3)
hier = col1.toggle("Hierarchical layout (top→down)", value=True)
physics = col2.toggle("Physics", value=True)
prune_id = col3.text_input("Prune to component containing particle #", value="")

if uploaded:
    G = parse_bytes(uploaded.read())

    if prune_id.strip().isdigit():
        pid = int(prune_id)
        if pid in G:
            G = component_with(G, pid) or G

    roots, leaves = roots_and_leaves(G)
    st.caption(f"Nodes: {G.number_of_nodes()} • Edges: {G.number_of_edges()} • Roots: {len(roots)} • Leaves: {len(leaves)}")

    net = Network(height="800px", width="100%", directed=True, bgcolor="#ffffff", font_color="#222")
    for n in G.nodes:
        role = "root" if n in roots else ("leaf" if n in leaves else "internal")
        color = "#4CAF50" if role == "root" else ("#E91E63" if role == "leaf" else "#2196F3")
        hint = f"#{n} • {role}\nin={G.in_degree(n)}, out={G.out_degree(n)}"
        size = 18 if role == "root" else (12 if role == "leaf" else 14)
        net.add_node(n, label=str(n), title=hint, color=color, size=size)
    for u, v in G.edges:
        net.add_edge(u, v, arrows="to")

    options = {
        "physics": {"enabled": physics, "solver": "forceAtlas2Based", "stabilization": {"iterations": 200}},
        "layout": {"hierarchical": {"enabled": hier, "direction": "UD", "sortMethod": "directed",
                                    "nodeSpacing": 120, "treeSpacing": 220, "levelSeparation": 160}},
        "interaction": {"hover": True, "navigationButtons": True, "keyboard": True, "selectConnectedEdges": True},
        "edges": {"arrows": {"to": {"enabled": True, "scaleFactor": 0.8}}},
        "nodes": {"shape": "dot"}
    }
    net.set_options(json.dumps(options))
    net.show("graph.html", notebook=False)
    with open("graph.html", "r", encoding="utf-8") as f:
        html = f.read()
    st.components.v1.html(html, height=820, scrolling=True)
else:
    st.info("Upload a log file to visualize the decay graph.")
