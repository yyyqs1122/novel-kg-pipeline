"""由人工标注的 mention 列表生成金标准数据与实体词典。

产物：
- data/annotated/gold.json   包含每篇文档的字符级 span、句子边界、实体集合和关系三元组
- data/annotated/entity_dictionary.json / .csv
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from schema import (
    ALIASES,
    DOC_MENTIONS,
    DOCS,
    ENTITY_INFO,
    ENTITY_TYPES,
    GOLD_TRIPLES,
    RELATION_TYPES,
)

ROOT = Path(__file__).resolve().parents[1]
GOLD_PATH = ROOT / "data" / "annotated" / "gold.json"
DICT_JSON = ROOT / "data" / "annotated" / "entity_dictionary.json"
DICT_CSV = ROOT / "data" / "annotated" / "entity_dictionary.csv"


def canonical(name: str) -> str:
    return ALIASES.get(name, name)


def build_matcher(mentions):
    """按首字索引，匹配时优先取最长串，保证 span 不嵌套。"""
    index = defaultdict(list)
    for surface, etype in mentions:
        index[surface[0]].append((surface, etype))
    for key in index:
        index[key].sort(key=lambda item: len(item[0]), reverse=True)
    return index


def find_spans(text: str, mentions):
    index = build_matcher(mentions)
    spans, i = [], 0
    while i < len(text):
        matched = None
        for surface, etype in index.get(text[i], []):
            if text.startswith(surface, i):
                matched = (surface, etype)
                break
        if matched:
            surface, etype = matched
            spans.append({
                "start": i,
                "end": i + len(surface),
                "text": surface,
                "type": etype,
                "canonical": canonical(surface),
            })
            i += len(surface)
        else:
            i += 1
    return spans


def split_sentences(text: str):
    sentences, start = [], 0
    for m in re.finditer(r"[。！？；\n]", text):
        end = m.end()
        piece = text[start:end]
        if len(piece.strip()) >= 5:
            lead = len(piece) - len(piece.lstrip())
            trail = len(piece) - len(piece.rstrip())
            sentences.append({"start": start + lead, "end": end - trail, "text": piece.strip()})
        start = end
    if start < len(text) and len(text[start:].strip()) >= 5:
        sentences.append({"start": start, "end": len(text), "text": text[start:].strip()})
    return sentences


def main():
    gold = {"docs": {}, "triples": [], "meta": {
        "entity_types": ENTITY_TYPES,
        "relation_types": {k: v["desc"] for k, v in RELATION_TYPES.items()},
        "annotation_note": "人工标注；实体按文档级规范名计分，关系按 (head, relation, tail) 精确匹配。",
    }}
    mention_counter = Counter()
    surface_counter = Counter()
    doc_counts = defaultdict(set)

    for doc_id, meta in DOCS.items():
        text = (ROOT / meta["path"]).read_text(encoding="utf-8")
        spans = find_spans(text, DOC_MENTIONS[doc_id])
        entities = sorted({s["canonical"] for s in spans})
        missing_info = [e for e in entities if e not in ENTITY_INFO]
        if missing_info:
            raise ValueError(f"{doc_id} 未登记释义的实体: {missing_info}")
        # 类型一致性校验
        for s in spans:
            info_type = ENTITY_INFO.get(s["canonical"], (None,))[0]
            if info_type and info_type != s["type"]:
                raise ValueError(f"{doc_id} 实体类型冲突: {s['text']} {s['type']} vs {info_type}")
        for s in spans:
            mention_counter[s["canonical"]] += 1
            surface_counter[s["text"]] += 1
            doc_counts[s["canonical"]].add(doc_id)
        gold["docs"][doc_id] = {
            "title": meta["title"],
            "text": text,
            "spans": spans,
            "entities": entities,
            "sentences": split_sentences(text),
        }

    # 关系校验
    for triple in GOLD_TRIPLES:
        doc = gold["docs"][triple["doc"]]
        for key in ("head", "tail"):
            if triple[key] not in doc["entities"]:
                raise ValueError(f"三元组实体不在文档实体集合中: {triple}")
        if triple["relation"] not in RELATION_TYPES:
            raise ValueError(f"未知关系: {triple['relation']}")
        gold["triples"].append(dict(triple))

    # 词典
    entries = []
    for name, (etype, desc) in ENTITY_INFO.items():
        if name not in mention_counter:
            continue
        surfaces = sorted({s["text"] for d in gold["docs"].values() for s in d["spans"] if s["canonical"] == name})
        aliases = sorted(s for s in surfaces if s != name)
        entries.append({
            "canonical": name,
            "type": etype,
            "type_cn": ENTITY_TYPES[etype],
            "aliases": aliases,
            "frequency": mention_counter[name],
            "docs": sorted(doc_counts[name]),
            "description": desc,
        })
    entries.sort(key=lambda e: (list(ENTITY_TYPES).index(e["type"]), -e["frequency"], e["canonical"]))

    dictionary = {
        "schema": ENTITY_TYPES,
        "entries": entries,
        "stats": {
            "entity_count": len(entries),
            "mention_count": sum(e["frequency"] for e in entries),
            "type_counts": dict(Counter(e["type"] for e in entries)),
        },
    }

    GOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLD_PATH.write_text(json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8")
    DICT_JSON.write_text(json.dumps(dictionary, ensure_ascii=False, indent=2), encoding="utf-8")
    with DICT_CSV.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["规范名", "类型", "类型中文", "别名", "出现次数", "所属文档", "释义"])
        for e in entries:
            writer.writerow([
                e["canonical"], e["type"], e["type_cn"], "|".join(e["aliases"]),
                e["frequency"], "|".join(e["docs"]), e["description"],
            ])

    print(f"docs={len(gold['docs'])} entities={len(entries)} mentions={sum(e['frequency'] for e in entries)} triples={len(gold['triples'])}")
    print("type_counts=", dictionary["stats"]["type_counts"])
    print("sentences=", {k: len(v["sentences"]) for k, v in gold["docs"].items()})


if __name__ == "__main__":
    main()
