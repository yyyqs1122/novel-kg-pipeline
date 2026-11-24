"""spaCy 关系抽取：spaCy 分词/句界/依存树 + 领域规则模板。

- 实体使用金标准 span（oracle NER），保证与 BERT/LLM 的关系抽取可比。
- 规则模板在“实体占位符”上匹配，避免人名/地名长度干扰。
- 输出三元组并附带触发句式与依存距离，便于错误分析。
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict

import jieba
import spacy
from spacy.tokens import Doc

from schema import GENDER
from utils import ROOT, canonical, load_gold, write_json

P = r"【P\d+】"
O = r"【O\d+】"
E = r"【E\d+】"


def gap(n, exclude=None):
    """允许跨过其他实体占位符的间隙；exclude='P' 时不允许跨过人物占位符。"""
    if exclude:
        return rf"(?:(?!【{exclude})(?:[^【]|【[^】]*】)){{0,{n}}}"
    return rf"(?:[^【]|【[^】]*】){{0,{n}}}"


def gp(n):
    return gap(n, "P")


def ge(n):
    return gap(n, "E")


SPECS = [
    # 夫妻
    ("夫妻", rf"({P}){gp(8)}(?:与|和){gp(30)}({P}){gp(8)}(?:结.{{0,2}}婚|结为夫妻|成婚)", 1, 2),
    ("夫妻", rf"({P})(?:的)?(?:妻子|丈夫)({P})", 1, 2),
    ("夫妻", rf"({P})(?:与|和)({P})(?:的)?(?:婚事)", 1, 2),
    # 恋人
    ("恋人", rf"({P}){gp(20)}(?:与|和){gp(20)}({P}){gp(12)}(?:恋爱|相恋)", 1, 2),
    ("恋人", rf"({P}){gp(80)}(?:的女友|的男友|女友|男友)({P})", 1, 2),
    ("恋人", rf"({P}){gp(20)}(?:与|和)({P}){gp(12)}(?:产生.{{0,6}}感情|感情)", 1, 2),
    ("恋人", rf"({P})(?:和|与)({P}){gp(10)}(?:互有爱慕之心)", 1, 2),
    # 暗恋
    ("暗恋", rf"({P}){gp(30)}(?:对){gp(16)}({P}){gp(10)}(?:产生.{{0,5}}情愫|有爱慕之情|钟情|暗恋)", 1, 2),
    ("暗恋", rf"({P}){gp(45)}(?:对她|对他){gp(10)}(?:有爱慕之情|钟情){gp(10)}({P})", 2, 1),
    ("暗恋", rf"({P}){gp(100)}(?:遇|遇见|遇到){gp(10)}({P})(?:表白)", 2, 1),
    # 朋友 / 同学
    ("朋友", rf"({P}){gp(20)}(?:与|和)({P}){gp(15)}(?:建立.{{0,8}}友情|成为朋友|友情)", 1, 2),
    ("同学", rf"({P}){gap(100)}(?:同班同学|同学)({P})", 1, 2),
    # 父女 / 父子 / 兄弟姐妹
    ("父女", rf"({P})(?:的)?(?:女儿)(?:——|是|为)?{gp(12)}({P})", 1, 2),
    ("父子", rf"({P})(?:的)?(?:儿子)(?:——|是|为)?{gp(12)}({P})", 1, 2),
    ("兄弟姐妹", rf"({P})(?:的)?(?:哥哥|弟弟|姐姐|妹妹)(?:——|是)?{gp(10)}({P})", 1, 2),
    # 任职于
    ("任职于", rf"({P}){gp(12)}(?:任|担任|当上|成为){gp(12)}({O})", 1, 2),
    ("任职于", rf"({O}){gp(10)}(?:副主任|主任|书记|记者|教师|队长){gp(10)}({P})", 2, 1),
    ("任职于", rf"({P}){gp(30)}(?:到|在|进入){gap(4)}({O}){gp(18)}(?:工作|当|任|记者|上班)", 1, 2),
    ("任职于", rf"({P}){gp(10)}(?:领导)({O})", 1, 2),
    ("任职于", rf"({P}){gp(30)}(?:到|在){gap(4)}({O})", 1, 2),
    # 就读于
    ("就读于", rf"({P}){gp(25)}(?:到|在|从){gap(4)}({O}){gp(18)}(?:读书|就读|毕业|上学)", 1, 2),
    # 参与
    ("参与", rf"({P}){gp(30)}(?:召开|组织|参加){ge(6)}({E})", 1, 2),
    ("参与", rf"({P}){gp(30)}(?:实行){ge(20)}({E})", 1, 2),
    ("参与", rf"({P}){gp(80)}(?:为了){ge(10)}({E})", 1, 2),
    ("参与", rf"({P}){gp(50)}(?:在){ge(12)}({E})", 1, 2),
    # 反对
    ("反对", rf"({P}){gp(30)}(?:反对|抵制|抵触|阻止){ge(10)}({E})", 1, 2),
    ("反对", rf"({P}){gp(25)}({E}){gp(10)}(?:抵触|反对)", 1, 2),
    # 拥有 / 资助
    ("拥有", rf"({P})(?:的)({O})", 1, 2),
    ("资助", rf"({P})(?:的)?(?:妻子|丈夫)({P}){gap(30)}(?:出资|资助){gap(12)}({O})", 1, 3),
]


class JiebaTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = jieba.lcut(text)
        return Doc(self.vocab, words=words, spaces=[False] * len(words))


def load_nlp():
    nlp = spacy.load("zh_core_web_sm")
    nlp.disable_pipe("ner")
    nlp.tokenizer = JiebaTokenizer(nlp.vocab)
    return nlp


def mask_sentence(text, spans):
    """把实体 span 替换为【类型+序号】占位符，返回掩码文本与映射。"""
    spans = sorted(spans, key=lambda s: s["start"])
    out, mapping, cursor = [], {}, 0
    for i, s in enumerate(spans):
        rel_s = s["start"] - 0
        if rel_s < cursor:
            continue
        out.append(text[cursor:rel_s])
        abbr = {"PER": "P", "LOC": "L", "ORG": "O", "TIME": "T", "EVENT": "E"}
        ph = f"【{abbr[s['type']]}{len(mapping)}】"
        out.append(ph)
        mapping[len(mapping)] = (s, ph)
        cursor = s["end"]
    out.append(text[cursor:])
    return "".join(out), mapping


def dep_distance(doc, sent, h, t):
    """实体 token 在依存树上的最短路径长度（无向），用于错误分析。"""
    def token_span(start, end):
        toks = [x for x in sent if x.idx < end and x.idx + len(x.text) > start]
        return toks

    ht, tt = token_span(h["start"], h["end"]), token_span(t["start"], t["end"])
    if not ht or not tt:
        return -1
    start_i, end_i = ht[0].i, tt[0].i
    adj = defaultdict(set)
    for tok in sent:
        adj[tok.i].add(tok.head.i)
        adj[tok.head.i].add(tok.i)
    from collections import deque
    q, dist = deque([start_i]), {start_i: 0}
    while q:
        cur = q.popleft()
        if cur == end_i:
            return dist[cur]
        for nb in adj[cur]:
            if nb not in dist:
                dist[nb] = dist[cur] + 1
                q.append(nb)
    return -1


def extract_doc(doc_id, doc_gold, nlp):
    text = doc_gold["text"]
    spans = doc_gold["spans"]
    doc = nlp(text)
    results, logs = [], []
    for sent in doc.sents:
        sent_spans = [s for s in spans if s["start"] >= sent.start_char and s["end"] <= sent.end_char]
        if not sent_spans:
            continue
        sent_text = text[sent.start_char:sent.end_char]
        masked, mapping = mask_sentence(sent_text, [
            {**s, "start": s["start"] - sent.start_char, "end": s["end"] - sent.start_char}
            for s in sent_spans
        ])
        # mapping 中的 span 是相对句首的，转回绝对偏移
        for idx, (s, ph) in list(mapping.items()):
            mapping[idx] = ({**s, "start": s["start"] + sent.start_char,
                             "end": s["end"] + sent.start_char}, ph)
        for rel, pattern, hg, tg in SPECS:
            for m in re.finditer(rf"(?=({pattern}))", masked):
                try:
                    hph, tph = m.group(hg + 1), m.group(tg + 1)
                except IndexError:
                    continue
                h = resolve_ph(hph, mapping)
                t = resolve_ph(tph, mapping)
                if not h or not t or h["canonical"] == t["canonical"]:
                    continue
                if not type_ok(rel, h, t):
                    continue
                dd = dep_distance(doc, sent, h, t)
                results.append({
                    "doc": doc_id, "head": h["canonical"], "relation": rel, "tail": t["canonical"],
                    "evidence": sent_text, "rule": pattern, "dep_distance": dd,
                    "head_surface": h["text"], "tail_surface": t["text"],
                })
                logs.append((rel, h["canonical"], t["canonical"]))
    # 冲突消解：同一对实体既有“任职于”又有“就读于”时保留后者
    pairs_read = {(canonical(r["head"]), canonical(r["tail"])) for r in results if r["relation"] == "就读于"}
    results = [r for r in results if not (r["relation"] == "任职于" and
                                          (canonical(r["head"]), canonical(r["tail"])) in pairs_read)]
    # 去重（对称关系归一）
    seen, unique = set(), []
    for r in results:
        h, t = canonical(r["head"]), canonical(r["tail"])
        if r["relation"] in ("夫妻", "恋人", "同学", "朋友", "兄弟姐妹") and h > t:
            h, t = t, h
        key = (r["doc"], h, r["relation"], t)
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    return unique, logs


def resolve_ph(ph, mapping):
    m = re.match(r"【([POLET])(\d+)】", ph or "")
    if not m:
        return None
    idx = int(m.group(2))
    return mapping.get(idx, (None, None))[0]


def type_ok(rel, h, t):
    if rel in ("夫妻", "恋人", "暗恋", "同学", "朋友", "兄弟姐妹"):
        return h["type"] == "PER" and t["type"] == "PER"
    if rel == "父女":
        return h["type"] == "PER" and t["type"] == "PER" and GENDER.get(h["canonical"]) == "M" and GENDER.get(t["canonical"]) == "F"
    if rel == "父子":
        return h["type"] == "PER" and t["type"] == "PER" and GENDER.get(h["canonical"]) == "M" and GENDER.get(t["canonical"]) == "M"
    if rel in ("任职于", "就读于", "拥有", "资助"):
        return h["type"] == "PER" and t["type"] == "ORG"
    if rel in ("参与", "反对"):
        return h["type"] == "PER" and t["type"] == "EVENT"
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs" / "pred_triples_spacy.json"))
    args = ap.parse_args()
    gold = load_gold()
    nlp = load_nlp()
    all_triples, logs = [], []
    for doc_id, doc in gold["docs"].items():
        triples, lg = extract_doc(doc_id, doc, nlp)
        all_triples.extend(triples)
        logs.extend(lg)
        print(f"{doc_id}: {len(triples)} triples")
    result = {"method": "spaCy (jieba tokenizer + dependency tree + rule templates)",
              "triples": all_triples, "count": len(all_triples)}
    write_json(args.out, result)
    print(f"saved {args.out} total={len(all_triples)}")


if __name__ == "__main__":
    main()
