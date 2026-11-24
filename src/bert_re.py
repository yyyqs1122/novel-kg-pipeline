"""BERT 关系分类：句内实体对 + [CLS] 句子 [SEP] 头实体 [SEP] 尾实体 [SEP]。

- 与实体抽取使用同一套 5 折句子切分（seed=42）。
- 负类 NA 下采样加权，缓解正负样本不平衡。
- 关系方向遵循 schema：对称关系训练时同时加入反向样本，评估时允许反向匹配。
"""
from __future__ import annotations

import argparse
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from schema import RELATION_TYPES
from utils import ROOT, canonical, load_gold, write_json

RELS = list(RELATION_TYPES)
LABELS = ["NA"] + RELS
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
NA_ID = 0
MAX_LEN = 256

CUES = {
    "夫妻": ["结婚", "结了婚", "结为夫妻", "妻子", "丈夫", "婚事", "娶", "嫁"],
    "父子": ["儿子"],
    "父女": ["女儿"],
    "兄弟姐妹": ["哥哥", "弟弟", "姐姐", "妹妹"],
    "恋人": ["恋爱", "女友", "男友", "感情", "相恋", "爱慕之心"],
    "暗恋": ["情愫", "爱慕", "钟情", "暗恋", "表白"],
    "同学": ["同学"],
    "朋友": ["友情", "朋友"],
    "任职于": ["任", "担任", "工作", "记者", "教师", "领导", "副主任", "书记"],
    "就读于": ["读书", "就读", "毕业", "上学"],
    "参与": ["召开", "组织", "实行", "参加", "牺牲", "受伤", "事故", "为了"],
    "反对": ["反对", "抵制", "抵触", "阻止"],
    "拥有": ["的砖窑", "的工厂", "的店"],
    "资助": ["出资", "资助", "捐资"],
}


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def build_sent_items(gold):
    items = []
    for doc_id, doc in gold["docs"].items():
        for si, sent in enumerate(doc["sentences"]):
            if len(sent["text"]) < 3:
                continue
            items.append({"doc": doc_id, "sent_idx": si, "text": sent["text"],
                          "start": sent["start"], "end": sent["end"]})
    return items


def spans_in_sentence(doc, sent):
    return [s for s in doc["spans"] if s["start"] >= sent["start"] and s["end"] <= sent["end"]]


def assign_triples_to_sentences(gold):
    """把金标准三元组挂到最能表达该关系的一句话上。"""
    assigned = defaultdict(set)  # (doc, sent_idx) -> {(head, rel, tail)}
    unassigned = []
    for tr in gold["triples"]:
        doc = gold["docs"][tr["doc"]]
        best = None
        for si, sent in enumerate(doc["sentences"]):
            hs = [s for s in spans_in_sentence(doc, sent) if canonical(s["canonical"]) == tr["head"]]
            ts = [s for s in spans_in_sentence(doc, sent) if canonical(s["canonical"]) == tr["tail"]]
            if not hs or not ts:
                continue
            cue_score = sum(1 for c in CUES.get(tr["relation"], []) if c in sent["text"])
            dist = min(abs(h["start"] - t["start"]) for h in hs for t in ts)
            key = (cue_score, -dist, -si)
            if best is None or key > best[0]:
                best = (key, si)
        if best is None:
            unassigned.append(tr)
            continue
        si = best[1]
        assigned[(tr["doc"], si)].add((canonical(tr["head"]), tr["relation"], canonical(tr["tail"])))
        if RELATION_TYPES.get(tr["relation"], {}).get("symmetric"):
            assigned[(tr["doc"], si)].add((canonical(tr["tail"]), tr["relation"], canonical(tr["head"])))
    return assigned, unassigned


def candidate_pairs(doc, sent):
    spans = spans_in_sentence(doc, sent)
    # 同一规范名只保留首次出现，避免重复实体对
    uniq = {}
    for s in spans:
        uniq.setdefault(canonical(s["canonical"]), s)
    names = sorted(uniq, key=lambda n: uniq[n]["start"])
    pairs = []
    for i, h in enumerate(names):
        for j, t in enumerate(names):
            if i == j:
                continue
            pairs.append((h, t, uniq[h]["text"], uniq[t]["text"]))
    return pairs


def build_rows(items, gold, assigned, tokenizer):
    rows = []
    for it in items:
        doc = gold["docs"][it["doc"]]
        sent = doc["sentences"][it["sent_idx"]]
        labels = assigned.get((it["doc"], it["sent_idx"]), set())
        for h, t, hs, ts in candidate_pairs(doc, sent):
            rel = next((r for hh, r, tt in labels if hh == h and tt == t), None)
            label = LABEL2ID[rel] if rel else NA_ID
            rows.append({**it, "head": h, "tail": t, "head_surface": hs, "tail_surface": ts,
                         "label": label})
    return rows


def encode(tokenizer, sent, head, tail):
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    sent_ids = tokenizer(sent, add_special_tokens=False, truncation=True, max_length=MAX_LEN - 40)["input_ids"]
    h_ids = tokenizer(head, add_special_tokens=False, max_length=16)["input_ids"]
    t_ids = tokenizer(tail, add_special_tokens=False, max_length=16)["input_ids"]
    ids = [cls_id] + sent_ids + [sep_id] + h_ids + [sep_id] + t_ids + [sep_id]
    ids = ids[:MAX_LEN]
    return ids, [1] * len(ids)


class REDataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.rows, self.tokenizer = rows, tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        ids, mask = encode(self.tokenizer, r["text"], r["head_surface"], r["tail_surface"])
        return {"ids": torch.tensor(ids), "mask": torch.tensor(mask),
                "label": torch.tensor(r["label"]), "row": r}


def collate(batch, pad_id=0):
    n = max(len(b["ids"]) for b in batch)
    ids = torch.stack([torch.cat([b["ids"], torch.full((n - len(b["ids"]),), pad_id, dtype=torch.long)]) for b in batch])
    mask = torch.stack([torch.cat([b["mask"], torch.zeros(n - len(b["mask"]), dtype=torch.long)]) for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    return {"ids": ids, "mask": mask, "labels": labels, "rows": [b["row"] for b in batch]}


def train_fold(train_rows, test_rows, tokenizer, model_name, epochs, device, seed):
    train_loader = DataLoader(REDataset(train_rows, tokenizer), batch_size=8, shuffle=True,
                              collate_fn=lambda b: collate(b, tokenizer.pad_token_id))
    test_loader = DataLoader(REDataset(test_rows, tokenizer), batch_size=16, shuffle=False,
                             collate_fn=lambda b: collate(b, tokenizer.pad_token_id))
    cnt = Counter(r["label"] for r in train_rows)
    weights = [0.10 if i == NA_ID else min(4.0, max(1.5, len(train_rows) / (len(LABELS) * max(cnt[i], 1)) * 1.5)) for i in range(len(LABELS))]
    weight_tensor = torch.tensor(weights, dtype=torch.float, device=device)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
    losses = []
    for epoch in range(epochs):
        model.train(); total, steps = 0.0, 0
        for b in train_loader:
            opt.zero_grad()
            logits = model(input_ids=b["ids"].to(device), attention_mask=b["mask"].to(device)).logits
            loss = loss_fn(logits, b["labels"].to(device))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += float(loss.item()); steps += 1
        losses.append(round(total / max(steps, 1), 4))

    model.eval(); preds, probs = [], []
    with torch.no_grad():
        for b in test_loader:
            logits = model(input_ids=b["ids"].to(device), attention_mask=b["mask"].to(device)).logits
            p = torch.softmax(logits, -1)
            for row, prob, pred in zip(b["rows"], p.cpu(), logits.argmax(-1).cpu()):
                if int(pred) != NA_ID:
                    preds.append({"doc": row["doc"], "head": row["head"], "tail": row["tail"],
                                  "relation": ID2LABEL[int(pred)], "confidence": round(float(prob[pred]), 4),
                                  "evidence": row["text"]})
    return preds, losses, len(train_rows), len(test_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base-chinese")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "pred_triples_bert.json"))
    args = ap.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)
    gold = load_gold()
    assigned, unassigned = assign_triples_to_sentences(gold)
    items = build_sent_items(gold)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    all_preds, logs = [], []
    t0 = time.time()
    for fold, (tr, te) in enumerate(kf.split(items), 1):
        train_items = [items[i] for i in tr]
        test_items = [items[i] for i in te]
        train_rows = build_rows(train_items, gold, assigned, tokenizer)
        test_rows = build_rows(test_items, gold, assigned, tokenizer)
        preds, losses, n_train, n_test = train_fold(train_rows, test_rows, tokenizer, args.model, args.epochs, device, args.seed + fold)
        all_preds.extend(preds)
        logs.append({"fold": fold, "train_pairs": n_train, "test_pairs": n_test,
                     "loss_curve": losses, "pred_triples": len(preds)})
        print(f"[fold {fold}] train_pairs={n_train} test_pairs={n_test} loss={losses[-1]} preds={len(preds)}")

    result = {"method": "BERT (bert-base-chinese, sentence-pair relation classifier, 5-fold CV)",
              "model": args.model, "epochs": args.epochs, "folds": logs,
              "elapsed_sec": round(time.time() - t0, 2),
              "unassigned_gold_triples": len(unassigned),
              "triples": all_preds}
    write_json(args.out, result)
    print(f"saved {args.out} gold_unassigned={len(unassigned)} elapsed={result['elapsed_sec']}s")


if __name__ == "__main__":
    main()
