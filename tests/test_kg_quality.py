"""轻量单元测试：不依赖 torch / spacy / neo4j，可在 CI 中直接跑。"""
from kg_quality import entity_quality, graph_stats, relation_quality, triple_key
from utils import canonical, load_gold


def test_alias_normalization():
    assert canonical("少平") == "孙少平"
    assert canonical("晓霞") == "田晓霞"
    assert canonical("孙少平") == "孙少平"
    assert canonical("未登录词") == "未登录词"


def test_symmetric_key_order():
    assert triple_key("田润叶", "夫妻", "李向前") == triple_key("李向前", "夫妻", "田润叶")
    assert triple_key("孙少平", "同学", "侯玉英") == triple_key("侯玉英", "同学", "孙少平")


def test_directed_key_not_reversed():
    assert triple_key("田福堂", "父女", "田润叶") != triple_key("田润叶", "父女", "田福堂")
    assert triple_key("孙少平", "暗恋", "郝红梅") != triple_key("郝红梅", "暗恋", "孙少平")


def test_perfect_entity_quality():
    gold = load_gold()
    nodes = [{"name": "孙少平", "type": "PER"}, {"name": "田晓霞", "type": "PER"}]
    # 只拿金标准中真实存在的两个实体做子集校验：子集预测会拉低召回，不应为 1.0
    metrics = entity_quality(nodes, gold)
    assert metrics["tp"] == 2
    assert metrics["fp"] == 0
    assert metrics["precision"] == 1.0


def test_relation_quality_extra_and_missing():
    gold = load_gold()
    # 空预测：recall 为 0，且 missing 非空
    empty = relation_quality([], gold)
    assert empty["precision"] == 0.0
    assert empty["recall"] == 0.0
    assert empty["fn"] == 33


def test_graph_stats_isolated():
    nodes = [{"name": "孙少平", "type": "PER"}, {"name": "田晓霞", "type": "PER"}, {"name": "旱灾", "type": "EVENT"}]
    edges = [{"head": "孙少平", "relation": "恋人", "tail": "田晓霞"}]
    stats = graph_stats(nodes, edges)
    assert stats["nodes"] == 3
    assert stats["edges"] == 1
    assert stats["isolated_count"] == 1
    assert stats["isolated_nodes"] == ["旱灾"]
