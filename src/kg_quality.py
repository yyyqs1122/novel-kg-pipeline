"""知识图谱质量与结构统计（实体级/关系级，统一按规范名）。"""
from __future__ import annotations

from collections import Counter, defaultdict

from schema import ALIASES, RELATION_TYPES
from utils import canonical


def gold_global(gold):
    ents = set()
    for doc in gold["docs"].values():
        for name in doc["entities"]:
            etype = next(s["type"] for s in doc["spans"] if s["canonical"] == name)
            ents.add((canonical(name), etype))
    rels = {triple_key(t["head"], t["relation"], t["tail"]) for t in gold["triples"]}
    return ents, rels


def triple_key(head, rel, tail):
    h, t = canonical(head), canonical(tail)
    if RELATION_TYPES.get(rel, {}).get("symmetric") and h > t:
        h, t = t, h
    return (h, rel, t)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "tp": tp, "fp": fp, "fn": fn}


def entity_quality(nodes, gold):
    gold_ents, _ = gold_global(gold)
    name_of = lambda n: canonical(n.get("name") or n.get("id"))
    pred = {(name_of(n), n["type"]) for n in nodes}
    tp, fp, fn = pred & gold_ents, pred - gold_ents, gold_ents - pred
    return prf(len(tp), len(fp), len(fn)) | {
        "gold": len(gold_ents), "pred": len(pred),
        "extra": sorted(f"{n}({t})" for n, t in fp),
        "missing": sorted(f"{n}({t})" for n, t in fn),
    }


def relation_quality(edges, gold):
    _, gold_rels = gold_global(gold)
    pred = {triple_key(e["head"], e["relation"], e["tail"]) for e in edges if e.get("relation") in RELATION_TYPES}
    tp, fp, fn = pred & gold_rels, pred - gold_rels, gold_rels - pred
    return prf(len(tp), len(fp), len(fn)) | {
        "gold": len(gold_rels), "pred": len(pred),
        "extra": sorted(f"{h}-{r}-{t}" for h, r, t in fp),
        "missing": sorted(f"{h}-{r}-{t}" for h, r, t in fn),
    }


def graph_stats(nodes, edges):
    name_of = lambda n: canonical(n.get("name") or n.get("id"))
    names = {name_of(n) for n in nodes}
    deg = Counter()
    rel_counts = Counter()
    for e in edges:
        h, t = canonical(e["head"]), canonical(e["tail"])
        if h in names and t in names:
            deg[h] += 1; deg[t] += 1
            rel_counts[e["relation"]] += 1
    n = len(names)
    possible = n * (n - 1) if n > 1 else 1
    return {
        "nodes": n,
        "edges": sum(rel_counts.values()),
        "density": round(sum(rel_counts.values()) / possible, 4),
        "avg_degree": round(sum(deg.values()) / max(n, 1), 3),
        "isolated_nodes": sorted(nm for nm in names if deg[nm] == 0),
        "isolated_count": sum(1 for nm in names if deg[nm] == 0),
        "type_counts": dict(Counter(x["type"] for x in nodes)),
        "relation_counts": dict(rel_counts),
    }
