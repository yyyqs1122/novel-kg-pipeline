#!/usr/bin/env python3
"""一条命令跑通：语料 → 词典/金标准 → 实体抽取 → 关系抽取 → 知识图谱。

用法：
    python pipeline.py all                 # 全量（BERT 关系约 40 min）
    python pipeline.py prepare             # 只生成 gold + 词典
    python pipeline.py entities            # BERT/BiLSTM 实体 + 评估（含 LLM 缓存）
    python pipeline.py relations           # spaCy/BERT 关系 + 评估（含 LLM 缓存）
    python pipeline.py kg                  # 构建 Neo4j 图谱并压测
    python pipeline.py all --fast          # 2 折/少轮数冒烟
    python pipeline.py all --skip-bert     # 跳过 BERT，用已有输出/缓存
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PY = sys.executable


def run(script: str, *args: str) -> None:
    cmd = [PY, str(SRC / script), *args]
    print(f"\n$ {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    env = dict(os.environ, PYTHONPATH=str(SRC))
    subprocess.run(cmd, cwd=SRC, check=True, env=env)
    print(f"  ✓ {script} 完成，用时 {time.perf_counter() - t0:.1f}s", flush=True)


def evaluate(pred: str, kind: str, out: str) -> None:
    run("evaluate.py", "--pred", str(ROOT / pred), "--kind", kind, "--out", str(ROOT / out))


def stage_prepare() -> None:
    run("make_gold.py")


def stage_entities(fast: bool, skip_bert: bool) -> None:
    if not skip_bert:
        run("bert_ner.py", "--epochs", "2" if fast else "6",
            "--folds", "2" if fast else "5", "--out", "../outputs/pred_entities_bert.json")
        evaluate("outputs/pred_entities_bert.json", "entity", "outputs/metrics_entities_bert.json")
    run("bilstm_crf.py", "--epochs", "10" if fast else "60",
        "--folds", "2" if fast else "5", "--out", "../outputs/pred_entities_bilstm.json")
    evaluate("outputs/pred_entities_bilstm.json", "entity", "outputs/metrics_entities_bilstm.json")
    evaluate("data/annotated/llm_entities.json", "entity", "outputs/metrics_entities_llm.json")


def stage_relations(fast: bool, skip_bert: bool) -> None:
    run("spacy_re.py")
    evaluate("outputs/pred_triples_spacy.json", "triple", "outputs/metrics_triples_spacy.json")
    if not skip_bert:
        run("bert_re.py", "--epochs", "2" if fast else "3",
            "--folds", "2" if fast else "5", "--out", "../outputs/pred_triples_bert.json")
        evaluate("outputs/pred_triples_bert.json", "triple", "outputs/metrics_triples_bert.json")
    evaluate("data/annotated/llm_triples.json", "triple", "outputs/metrics_triples_llm.json")


def stage_kg() -> None:
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "kgNeo4j2026")),
        )
        with driver.session() as s:
            s.run("RETURN 1").consume()
        driver.close()
    except Exception as exc:
        raise SystemExit(
            "Neo4j 未就绪。请先启动：\n"
            "  sudo -u neo4j env NEO4J_CONF=/etc/neo4j /usr/share/neo4j/bin/neo4j start\n"
            f"原始错误：{exc}"
        )
    run("kg_neo4j.py")
    run("visualize_kg.py")


STAGES = {
    "prepare": lambda a: stage_prepare(),
    "entities": lambda a: stage_entities(a.fast, a.skip_bert),
    "relations": lambda a: stage_relations(a.fast, a.skip_bert),
    "kg": lambda a: stage_kg(),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="《平凡的世界》信息抽取与知识图谱流水线")
    ap.add_argument("stage", choices=["all", "prepare", "entities", "relations", "kg"])
    ap.add_argument("--fast", action="store_true", help="2 折/少轮数冒烟，不用于最终指标")
    ap.add_argument("--skip-bert", action="store_true", help="跳过 BERT NER/RE（可用已有输出）")
    ap.add_argument("--dry-run", action="store_true", help="只打印阶段，不执行")
    args = ap.parse_args()

    order = ["prepare", "entities", "relations", "kg"] if args.stage == "all" else [args.stage]
    print(f"项目根目录：{ROOT}")
    print(f"执行阶段：{order}")
    if args.dry_run:
        return
    t0 = time.perf_counter()
    for name in order:
        print(f"\n{'=' * 60}\n[{name}]")
        STAGES[name](args)
    print(f"\n全部完成，总用时 {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
