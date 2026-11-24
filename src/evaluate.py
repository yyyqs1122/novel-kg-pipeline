"""统一评估：实体（文档级实体集合）与关系（三元组集合）。

评测口径：
- 实体：同一文档内，规范名与类型同时正确才算 TP；别名（少平→孙少平）自动归一。
  这是面向知识图谱的“实体级”评估，不惩罚同一实体的不同别名写法。
- 关系：在同一文档内按 (head, relation, tail) 精确匹配；对称关系允许反向。
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from utils import canonical, load_gold, load_pred_entities, load_pred_triples, relation_is_symmetric, write_json
from schema import ENTITY_TYPES, RELATION_TYPES


def prf(tp: int, fp: int, fn: int):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "tp": tp, "fp": fp, "fn": fn}


def evaluate_entities(pred_docs, gold=None):
    gold = gold or load_gold()
    per_doc, errors = {}, {}
    type_counts = defaultdict(Counter)
    for doc_id, doc in gold["docs"].items():
        gold_set = {(canonical(e), _type(e, gold)) for e in doc["entities"]}
        pred_raw = pred_docs.get(doc_id, [])
        pred_set = set()
        for item in pred_raw:
            name = canonical(item.get("text", ""))
            etype = item.get("type", "")
            if name and etype in ENTITY_TYPES:
                pred_set.add((name, etype))
        tp_set = gold_set & pred_set
        fp_set = pred_set - gold_set
        fn_set = gold_set - pred_set
        per_doc[doc_id] = prf(len(tp_set), len(fp_set), len(fn_set))
        errors[doc_id] = {
            "missing": [f"{n}({t})" for n, t in sorted(fn_set)],
            "extra": [f"{n}({t})" for n, t in sorted(fp_set)],
        }
        for n, t in gold_set:
            type_counts[t]["gold"] += 1
        for n, t in pred_set:
            type_counts[t]["pred"] += 1
        for n, t in tp_set:
            type_counts[t]["tp"] += 1
        for n, t in fp_set:
            type_counts[t]["fp"] += 1
        for n, t in fn_set:
            type_counts[t]["fn"] += 1

    total = {"tp": sum(v["tp"] for v in per_doc.values()),
             "fp": sum(v["fp"] for v in per_doc.values()),
             "fn": sum(v["fn"] for v in per_doc.values())}
    overall = prf(total["tp"], total["fp"], total["fn"])
    per_type = {}
    for t in ENTITY_TYPES:
        c = type_counts[t]
        per_type[t] = prf(c["tp"], c["fp"], c["fn"]) | {"gold": c["gold"], "pred": c["pred"]}
    macro = round(sum(per_type[t]["f1"] for t in ENTITY_TYPES) / len(ENTITY_TYPES), 4)
    return {"overall": overall, "macro_f1": macro, "per_type": per_type,
            "per_doc": per_doc, "errors": errors,
            "pred_entity_mentions": sum(len(v) for v in pred_docs.values())}


def _type(entity_name, gold):
    # gold["entities"] 只有名字；类型从 spans 反查
    for doc in gold["docs"].values():
        for s in doc["spans"]:
            if canonical(s["canonical"]) == canonical(entity_name):
                return s["type"]
    return None


def _triple_key(doc, head, rel, tail, symmetric=False):
    h, t = canonical(head), canonical(tail)
    if symmetric and h > t:
        h, t = t, h
    return (doc, h, rel, t)


def evaluate_triples(pred_triples, gold=None):
    gold = gold or load_gold()
    gold_by_doc = defaultdict(set)
    for tr in gold["triples"]:
        gold_by_doc[tr["doc"]].add(_triple_key(tr["doc"], tr["head"], tr["relation"], tr["tail"],
                                               relation_is_symmetric(tr["relation"])))
    pred_by_doc = defaultdict(set)
    for tr in pred_triples:
        doc = tr.get("doc")
        rel = tr.get("relation", "")
        if doc not in gold["docs"] or rel not in RELATION_TYPES:
            continue
        pred_by_doc[doc].add(_triple_key(doc, tr.get("head", ""), rel, tr.get("tail", ""),
                                         relation_is_symmetric(rel)))

    per_doc, errors = {}, {}
    rel_counts = defaultdict(Counter)
    for doc_id in gold["docs"]:
        g, p = gold_by_doc[doc_id], pred_by_doc[doc_id]
        per_doc[doc_id] = prf(len(g & p), len(p - g), len(g - p))
        errors[doc_id] = {
            "missing": [f"{h}-{r}-{t}" for _, h, r, t in sorted(g - p)],
            "extra": [f"{h}-{r}-{t}" for _, h, r, t in sorted(p - g)],
        }
        for key in g:
            rel_counts[key[2]]["gold"] += 1
        for key in p:
            rel_counts[key[2]]["pred"] += 1
        for key in g & p:
            rel_counts[key[2]]["tp"] += 1
        for key in p - g:
            rel_counts[key[2]]["fp"] += 1
        for key in g - p:
            rel_counts[key[2]]["fn"] += 1

    total = {"tp": sum(v["tp"] for v in per_doc.values()),
             "fp": sum(v["fp"] for v in per_doc.values()),
             "fn": sum(v["fn"] for v in per_doc.values())}
    overall = prf(total["tp"], total["fp"], total["fn"])
    per_relation = {}
    for r in RELATION_TYPES:
        c = rel_counts[r]
        per_relation[r] = prf(c["tp"], c["fp"], c["fn"]) | {"gold": c["gold"], "pred": c["pred"]}
    supported = [r for r in RELATION_TYPES if per_relation[r]["gold"] > 0]
    macro = round(sum(per_relation[r]["f1"] for r in supported) / len(supported), 4) if supported else 0.0
    return {"overall": overall, "macro_f1": macro, "per_relation": per_relation,
            "per_doc": per_doc, "errors": errors,
            "pred_triples": sum(len(v) for v in pred_by_doc.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--kind", choices=["entity", "triple"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.kind == "entity":
        metrics = evaluate_entities(load_pred_entities(args.pred))
    else:
        metrics = evaluate_triples(load_pred_triples(args.pred))
    write_json(args.out, metrics)
    print(f"{args.kind}: P={metrics['overall']['precision']} R={metrics['overall']['recall']} F1={metrics['overall']['f1']}")


if __name__ == "__main__":
    main()
