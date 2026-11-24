#!/usr/bin/env python3
"""把 docs/01-05 合并成一份可直接提交的 REPORT.md。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = [
    "01_实体词典.md",
    "02_实体抽取对比.md",
    "03_关系三元组.md",
    "04_知识图谱对比.md",
    "05_流水线复现.md",
]

HEAD = """# 《平凡的世界》中文信息抽取与知识图谱实验总报告

> 语料：用户上传的三张情节梗概截图（第一部 / 第二部概要 / 第三部）。  
> 任务：实体词典 → 实体抽取（BERT / LLM / BiLSTM-CRF）→ 关系三元组（spaCy / BERT / LLM）→ 知识图谱（Neo4j / LLM）→ 流水线与 GitHub。

## 核心结论

| 任务 | 最佳方法 | F1 |
|---|---|---:|
| 实体抽取 | LLM 零样本 | **0.9515** |
| 关系抽取 | LLM | **0.9589** |
| 图谱（精度） | Neo4j + 词典/融合 | **关系 0.9697** |
| 图谱（召回/背景覆盖） | LLM 端到端 | **关系 1.0000（含 5 条背景边）** |

- 监督模型受 24 句 / 53 个文档级实体 / 37 条正例限制，BERT 实体 F1 0.6813、BiLSTM-CRF 0.4250。
- spaCy 规则关系精确率 1.0、召回 0.8378，漏项集中在跨句与代词回指。
- Neo4j 路线 39 节点 / 33 边，导入 3.97 s；LLM 图谱 44 节点 / 38 边，生成 53.96 s。
- 小图查询 JSON 更快（µs 级），Neo4j 的 4–9 ms 主要是 Bolt/事务开销；规模上 Neo4j 的索引与 Cypher 更有优势。

## 目录

1. [实体词典](docs/01_实体词典.md)
2. [实体抽取对比](docs/02_实体抽取对比.md)
3. [关系三元组](docs/03_关系三元组.md)
4. [知识图谱对比](docs/04_知识图谱对比.md)
5. [流水线复现](docs/05_流水线复现.md)

---
"""


def main():
    parts = [HEAD]
    for name in DOCS:
        text = (ROOT / "docs" / name).read_text(encoding="utf-8")
        text = text.replace("](../outputs/", "](outputs/")
        parts.append(text.rstrip() + "\n\n---\n")
    (ROOT / "REPORT.md").write_text("\n".join(parts), encoding="utf-8")
    print("saved REPORT.md")


if __name__ == "__main__":
    main()
