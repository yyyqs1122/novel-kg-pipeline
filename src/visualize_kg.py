#!/usr/bin/env python3
"""把知识图谱画成静态 PNG：主连通图 + 孤立节点侧栏。"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties

from utils import ROOT, canonical

COLORS = {"PER": "#E05A4F", "LOC": "#3F8F68", "ORG": "#D19A2A", "TIME": "#4A7FD4", "EVENT": "#8E62B8"}
NAMES = {"PER": "人物", "LOC": "地点", "ORG": "组织机构", "TIME": "时间", "EVENT": "事件/制度"}
FONT = FontProperties(family="Noto Sans CJK SC")


def load(path):
    data = json.loads((ROOT / path).read_text(encoding="utf-8"))
    nodes = [{"name": canonical(n.get("name") or n.get("id")), "type": n["type"]} for n in data["nodes"]]
    edges = [{"head": canonical(e["head"]), "relation": e["relation"], "tail": canonical(e["tail"])}
             for e in data["edges"]]
    return nodes, edges


def draw(path, out, title):
    nodes, edges = load(path)
    G = nx.Graph()
    for n in nodes:
        G.add_node(n["name"], type=n["type"])
    for e in edges:
        G.add_edge(e["head"], e["tail"], label=e["relation"])

    isolated = [n for n in G.nodes if G.degree[n] == 0]
    active = G.subgraph([n for n in G.nodes if G.degree[n] > 0]).copy()

    pos = nx.kamada_kawai_layout(active)
    # 归一化到 [-1, 1]，留出标签空间
    xs = [p[0] for p in pos.values()]; ys = [p[1] for p in pos.values()]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    for n, (x, y) in pos.items():
        pos[n] = ((x - xmin) / max(xmax - xmin, 1e-9) * 2 - 1,
                  (y - ymin) / max(ymax - ymin, 1e-9) * 2 - 1)

    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.25, 1.05], wspace=0.02)
    ax = fig.add_subplot(gs[0, 0])
    panel = fig.add_subplot(gs[0, 1])

    node_colors = [COLORS.get(active.nodes[n].get("type"), "#999999") for n in active.nodes]
    degrees = dict(active.degree)
    sizes = [850 + 150 * degrees[n] for n in active.nodes]

    nx.draw_networkx_edges(active, pos, ax=ax, alpha=0.28, width=1.6, edge_color="#444444")
    nx.draw_networkx_nodes(active, pos, ax=ax, node_color=node_colors, node_size=sizes,
                           edgecolors="#222222", linewidths=0.8, alpha=0.95)
    nx.draw_networkx_labels(active, pos, ax=ax, font_size=10, font_family="Noto Sans CJK SC",
                            font_color="#111111")
    nx.draw_networkx_edge_labels(
        active, pos, ax=ax, edge_labels=nx.get_edge_attributes(active, "label"),
        font_size=7, font_family="Noto Sans CJK SC",
        bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.78),
        label_pos=0.5, rotate=False,
    )

    ax.set_title(f"{title}\n{len(nodes)} 个实体节点 · {len(edges)} 条关系边 · 主连通图 {len(active)} 节点",
                 fontsize=15, pad=10, fontproperties=FONT)
    ax.axis("off")
    ax.margins(0.10)

    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS[t],
                      markersize=11, label=NAMES[t]) for t in COLORS]
    ax.legend(handles=handles, loc="lower left", fontsize=9, frameon=True, ncol=5)

    # 侧栏：孤立节点按类型分组
    panel.set_title(f"孤立节点（{len(isolated)}）", fontsize=13, pad=8, fontproperties=FONT)
    panel.axis("off")
    y = 0.95
    for t in COLORS:
        group = sorted([n for n in isolated if G.nodes[n].get("type") == t])
        if not group:
            continue
        panel.text(0.02, y, f"{NAMES[t]}（{len(group)}）", fontsize=11, color=COLORS[t],
                   fontproperties=FONT, fontweight="bold")
        y -= 0.055
        for name in group:
            panel.text(0.06, y, f"● {name}", fontsize=9.5, color=COLORS[t], fontproperties=FONT)
            y -= 0.042
        y -= 0.012
    panel.text(0.02, 0.02, "这些节点没有实体—实体关系边；\n主图仅显示有关系的节点。",
               fontsize=9, color="#666666", fontproperties=FONT, va="bottom")

    fig.savefig(ROOT / out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", ROOT / out)


if __name__ == "__main__":
    draw("outputs/kg_pipeline.json", "outputs/kg_pipeline.png", "流水线图谱（词典 + LLM/spaCy 融合）")
    draw("outputs/kg_llm.json", "outputs/kg_llm.png", "LLM 端到端图谱")
