"""Neo4j 知识图谱构建、Cypher 导出、查询基准测试。

图谱来源：实体词典（39 个规范实体）+ LLM/spaCy 三元组融合。
同时评估 LLM 端到端图谱的质量与查询性能。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

from neo4j import GraphDatabase

from kg_quality import entity_quality, graph_stats, relation_quality, triple_key
from schema import ENTITY_TYPES
from utils import ROOT, canonical, load_gold, load_pred_triples, write_json

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "kgNeo4j2026")


def load_entities():
    dictionary = json.loads((ROOT / "data/annotated/entity_dictionary.json").read_text(encoding="utf-8"))
    return dictionary["entries"]


def fuse_triples():
    llm = load_pred_triples(ROOT / "data/annotated/llm_triples.json")
    spacy = load_pred_triples(ROOT / "outputs/pred_triples_spacy.json")
    merged = {}
    for source, rows in (("LLM", llm), ("spaCy", spacy)):
        for tr in rows:
            h, r, t = canonical(tr["head"]), tr["relation"], canonical(tr["tail"])
            if h == t:
                continue
            key = triple_key(h, r, t)
            if key not in merged:
                merged[key] = {"head": h, "relation": r, "tail": t, "sources": [source],
                               "evidence": tr.get("evidence", ""), "confidence": 0.9 if source == "LLM" else 1.0}
            else:
                if source not in merged[key]["sources"]:
                    merged[key]["sources"].append(source)
                merged[key]["confidence"] = max(merged[key]["confidence"], 0.9 if source == "LLM" else 1.0)
    return list(merged.values())


def mentions(gold):
    rows = []
    for doc_id, doc in gold["docs"].items():
        counts = Counter(canonical(s["canonical"]) for s in doc["spans"])
        for name, c in counts.items():
            rows.append({"doc": doc_id, "name": name, "count": c})
    return rows


def build_neo4j(nodes, edges, docs, mention_rows):
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    t0 = time.perf_counter()
    with driver.session() as s:
        s.run("MATCH (n) DETACH DELETE n").consume()
        s.run("CREATE CONSTRAINT entity_name IF NOT EXISTS ON (n:Entity) ASSERT n.name IS UNIQUE").consume()
        s.run("CREATE INDEX entity_type IF NOT EXISTS FOR (n:Entity) ON (n.type)").consume()
        s.run("CREATE CONSTRAINT document_id IF NOT EXISTS ON (d:Document) ASSERT d.id IS UNIQUE").consume()

        for etype in ENTITY_TYPES:
            rows = [n for n in nodes if n["type"] == etype]
            if not rows:
                continue
            s.run(
                f"UNWIND $rows AS row MERGE (n:Entity:`{etype}` {{name: row.name}}) "
                "SET n.type=row.type, n.type_cn=row.type_cn, n.aliases=row.aliases, "
                "n.description=row.description, n.frequency=row.frequency, n.docs=row.docs",
                rows=rows,
            ).consume()

        s.run("UNWIND $rows AS row MERGE (d:Document {id: row.id}) SET d.title=row.title", rows=docs).consume()
        s.run(
            "UNWIND $rows AS row MATCH (d:Document {id: row.doc}) MATCH (n:Entity {name: row.name}) "
            "MERGE (d)-[m:MENTIONS]->(n) SET m.count=row.count",
            rows=mention_rows,
        ).consume()

        by_rel = defaultdict(list)
        for e in edges:
            by_rel[e["relation"]].append(e)
        for rel, rows in by_rel.items():
            s.run(
                f"UNWIND $rows AS row MATCH (h:Entity {{name: row.head}}) MATCH (t:Entity {{name: row.tail}}) "
                f"MERGE (h)-[r:`{rel}`]->(t) SET r.sources=row.sources, r.evidence=row.evidence, "
                "r.confidence=row.confidence",
                rows=rows,
            ).consume()

        node_count = s.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
        rel_count = s.run("MATCH ()-[r]->() WHERE type(r) <> 'MENTIONS' RETURN count(r) AS c").single()["c"]
        mention_count = s.run("MATCH ()-[r:MENTIONS]->() RETURN count(r) AS c").single()["c"]
    build_ms = round((time.perf_counter() - t0) * 1000, 1)
    driver.close()
    return {"build_ms": build_ms, "node_count": node_count, "rel_count": rel_count,
            "mention_count": mention_count}


def neo4j_benchmark():
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    queries = {
        "q1_neighbors": ("MATCH (n:Entity {name:$name})-[r]-(m:Entity) RETURN type(r) AS relation, m.name AS other "
                         "ORDER BY other LIMIT 50", {"name": "孙少平"}),
        "q2_children": ("MATCH (p:Entity {name:$name})-[:父子|父女]->(c:Entity) RETURN c.name AS child "
                        "ORDER BY child", {"name": "田福堂"}),
        "q3_event_participants": ("MATCH (p:Entity:PER)-[r:参与]->(e:Entity:EVENT) RETURN p.name AS person, "
                                  "e.name AS event ORDER BY person LIMIT 50", {}),
        "q4_two_hop": ("MATCH path=(a:Entity {name:$name})-[*1..2]-(b:Entity) WHERE b.name <> $name "
                       "RETURN DISTINCT b.name AS other ORDER BY other LIMIT 50", {"name": "孙少平"}),
        "q5_relation_counts": ("MATCH ()-[r]->() WHERE type(r) <> 'MENTIONS' RETURN type(r) AS relation, "
                               "count(*) AS count ORDER BY count DESC", {}),
        "q6_shortest_path": ("MATCH (a:Entity {name:$a}),(b:Entity {name:$b}), p=shortestPath((a)-[*..6]-(b)) "
                             "RETURN length(p) AS hops", {"a": "孙少平", "b": "田福军"}),
    }
    out = {}
    with driver.session() as s:
        for name, (query, params) in queries.items():
            for _ in range(3):
                s.run(query, params).data()
            times = []
            rows = None
            for _ in range(20):
                t0 = time.perf_counter()
                rows = s.run(query, params).data()
                times.append((time.perf_counter() - t0) * 1000)
            times.sort()
            out[name] = {"avg_ms": round(statistics.mean(times), 3),
                         "p95_ms": round(times[int(0.95 * len(times)) - 1], 3),
                         "rows": len(rows or []),
                         "sample": (rows or [])[:2]}
    driver.close()
    return out


# ---------- JSON 图查询（模拟 LLM 直接产出的 JSON KG） ----------
class JsonGraph:
    def __init__(self, nodes, edges):
        self.nodes = {canonical(n.get("name") or n.get("id")): n for n in nodes}
        self.edges = [(canonical(e["head"]), e["relation"], canonical(e["tail"])) for e in edges]
        self.adj = defaultdict(list)
        for i, (h, r, t) in enumerate(self.edges):
            self.adj[h].append((t, r, i)); self.adj[t].append((h, r, i))

    def q1(self, name):
        return [(r, o) for o, r, _ in self.adj.get(name, [])]

    def q2(self, name):
        return [t for h, r, t in self.edges if h == name and r in ("父子", "父女")]

    def q3(self):
        return [(h, t) for h, r, t in self.edges if r == "参与"]

    def q4(self, name):
        seen = {name}; frontier = {name}
        for _ in range(2):
            nxt = set()
            for x in frontier:
                nxt |= {o for o, _, _ in self.adj.get(x, [])}
            seen |= nxt; frontier = nxt
        return sorted(seen - {name})

    def q5(self):
        return Counter(r for _, r, _ in self.edges)

    def q6(self, a, b):
        q = deque([(a, 0)]); seen = {a}
        while q:
            x, d = q.popleft()
            if x == b:
                return d
            for o, _, _ in self.adj.get(x, []):
                if o not in seen:
                    seen.add(o); q.append((o, d + 1))
        return -1


def json_benchmark(path):
    kg = json.loads((ROOT / path).read_text(encoding="utf-8"))
    g = JsonGraph(kg["nodes"], kg["edges"])
    calls = {
        "q1_neighbors": lambda: g.q1("孙少平"),
        "q2_children": lambda: g.q2("田福堂"),
        "q3_event_participants": lambda: g.q3(),
        "q4_two_hop": lambda: g.q4("孙少平"),
        "q5_relation_counts": lambda: g.q5(),
        "q6_shortest_path": lambda: g.q6("孙少平", "田福军"),
    }
    out = {}
    for name, fn in calls.items():
        for _ in range(3):
            fn()
        times = []
        rows = None
        for _ in range(100):
            t0 = time.perf_counter()
            rows = fn()
            times.append((time.perf_counter() - t0) * 1000)
        times.sort()
        out[name] = {"avg_ms": round(statistics.mean(times), 4),
                     "p95_ms": round(times[int(0.95 * len(times)) - 1], 4),
                     "rows": len(rows) if hasattr(rows, "__len__") else 1}
    return out


def export_cypher(nodes, edges, docs, mention_rows, path):
    def q(s):
        return "'" + str(s).replace("\\", "\\\\").replace("'", "\\'") + "'"
    lines = ["// 平凡的世界（三段梗概）知识图谱", "// 自动生成，可直接在 Neo4j Browser 中执行",
             "CREATE CONSTRAINT entity_name IF NOT EXISTS ON (n:Entity) ASSERT n.name IS UNIQUE;",
             "CREATE INDEX entity_type IF NOT EXISTS FOR (n:Entity) ON (n.type);", ""]
    for n in nodes:
        lines.append(
            f"MERGE (n:Entity:{n['type']} {{name:{q(n['name'])}}}) SET n.type={q(n['type'])}, "
            f"n.type_cn={q(n['type_cn'])}, n.description={q(n['description'])}, "
            f"n.aliases={q(n['aliases'])}, n.frequency={n['frequency']}, n.docs={q(n['docs'])};"
        )
    lines.append("")
    for d in docs:
        lines.append(f"MERGE (d:Document {{id:{q(d['id'])}}}) SET d.title={q(d['title'])};")
    for m in mention_rows:
        lines.append(f"MATCH (d:Document {{id:{q(m['doc'])}}}),(n:Entity {{name:{q(m['name'])}}}) "
                     f"MERGE (d)-[r:MENTIONS]->(n) SET r.count={m['count']};")
    lines.append("")
    for e in edges:
        lines.append(f"MATCH (h:Entity {{name:{q(e['head'])}}}),(t:Entity {{name:{q(e['tail'])}}}) "
                     f"MERGE (h)-[r:`{e['relation']}`]->(t) SET r.sources={q(e['sources'])}, "
                     f"r.confidence={e['confidence']}, r.evidence={q(e['evidence'])};")
    (ROOT / path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def db_size_bytes():
    try:
        out = subprocess.check_output(["sudo", "du", "-sb", "/var/lib/neo4j/data/databases/neo4j"],
                                      stderr=subprocess.DEVNULL).decode().split()[0]
        return int(out)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-json", default=str(ROOT / "outputs/kg_pipeline.json"))
    ap.add_argument("--export-cypher", default=str(ROOT / "outputs/kg_pipeline.cypher"))
    ap.add_argument("--metrics", default=str(ROOT / "outputs/kg_metrics.json"))
    args = ap.parse_args()

    gold = load_gold()
    nodes = [{"name": e["canonical"], "type": e["type"], "type_cn": e["type_cn"],
              "aliases": e["aliases"], "description": e["description"],
              "frequency": e["frequency"], "docs": e["docs"]} for e in load_entities()]
    edges = fuse_triples()
    docs = [{"id": k, "title": v["title"]} for k, v in gold["docs"].items()]
    mentions_rows = mentions(gold)

    write_json(args.export_json, {"method": "dictionary + LLM/spaCy fusion", "nodes": nodes, "edges": edges})
    export_cypher(nodes, edges, docs, mentions_rows, Path(args.export_cypher).relative_to(ROOT))

    build = build_neo4j(nodes, edges, docs, mentions_rows)
    build["db_size_bytes"] = db_size_bytes()
    print("neo4j build", build)

    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    with driver.session() as s:
        label_counts = {t: s.run(f"MATCH (n:`{t}`) RETURN count(n) AS c").single()["c"] for t in ENTITY_TYPES}
        rel_counts = {r["relation"]: r["count"] for r in s.run(
            "MATCH ()-[r]->() WHERE type(r) <> 'MENTIONS' RETURN type(r) AS relation, count(*) AS count").data()}
        isolated = [r["name"] for r in s.run(
            "MATCH (n:Entity) OPTIONAL MATCH (n)-[r]-(:Entity) WITH n, count(r) AS deg "
            "WHERE deg = 0 RETURN n.name AS name ORDER BY name").data()]
    driver.close()

    neo = neo4j_benchmark()
    json_pipe = json_benchmark("outputs/kg_pipeline.json")
    json_llm = json_benchmark("outputs/kg_llm.json")

    kg_llm = json.loads((ROOT / "outputs/kg_llm.json").read_text(encoding="utf-8"))
    llm_elapsed_ms = int((ROOT / "outputs/llm_kg_elapsed.txt").read_text().split("=")[1])
    result = {
        "pipeline": {
            "build": build,
            "label_counts": label_counts,
            "relation_counts": rel_counts,
            "isolated_entities": isolated,
            "quality_entities": entity_quality(nodes, gold),
            "quality_relations": relation_quality(edges, gold),
            "structure": graph_stats(nodes, edges),
            "neo4j_queries": neo,
            "json_queries": json_pipe,
        },
        "llm": {
            "generation_ms": llm_elapsed_ms,
            "node_count": len(kg_llm["nodes"]),
            "edge_count": len(kg_llm["edges"]),
            "quality_entities": entity_quality(kg_llm["nodes"], gold),
            "quality_relations": relation_quality(kg_llm["edges"], gold),
            "structure": graph_stats(kg_llm["nodes"], kg_llm["edges"]),
            "json_queries": json_llm,
        },
    }
    write_json(args.metrics, result)
    print("saved", args.metrics)
    print("pipeline quality", result["pipeline"]["quality_entities"], result["pipeline"]["quality_relations"])
    print("llm quality", result["llm"]["quality_entities"], result["llm"]["quality_relations"])


if __name__ == "__main__":
    main()
