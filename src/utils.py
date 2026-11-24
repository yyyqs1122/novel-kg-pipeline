"""公共工具：路径、数据加载、别名规范化、预测结果读写。"""
from __future__ import annotations

import json
from pathlib import Path

from schema import ALIASES, ENTITY_INFO, RELATION_TYPES

ROOT = Path(__file__).resolve().parents[1]
GOLD_PATH = ROOT / "data" / "annotated" / "gold.json"
DICT_PATH = ROOT / "data" / "annotated" / "entity_dictionary.json"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_gold():
    return json.loads(GOLD_PATH.read_text(encoding="utf-8"))


def load_dictionary():
    return json.loads(DICT_PATH.read_text(encoding="utf-8"))


def canonical(name: str) -> str:
    name = (name or "").strip()
    return ALIASES.get(name, name)


def entity_type(name: str) -> str | None:
    info = ENTITY_INFO.get(canonical(name))
    return info[0] if info else None


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_pred_entities(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("docs", data)


def load_pred_triples(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("triples", data)


def relation_is_symmetric(rel: str) -> bool:
    return RELATION_TYPES.get(rel, {}).get("symmetric", False)
