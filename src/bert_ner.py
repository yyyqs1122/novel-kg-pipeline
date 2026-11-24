"""BERT 实体抽取：bert-base-chinese + BIO 序列标注 + 5 折句子级交叉验证。

设计说明：
- 语料很小（24 句、77 个 mention），采用句子级 5 折交叉验证，保证每条句子只出现在
  验证集一次，避免句子内泄漏。
- O 标签占比极高，使用类别加权交叉熵（O 降权、实体标签升权），防止模型退化为全 O。
- 输出统一格式，供 evaluate.py 与报告生成器使用。
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer

from schema import ENTITY_TYPES
from utils import ROOT, load_gold, write_json

LABELS = ["O"] + [f"{p}-{t}" for t in ENTITY_TYPES for p in ("B", "I")]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
MAX_LEN = 256


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_items(gold):
    items = []
    for doc_id, doc in gold["docs"].items():
        for si, sent in enumerate(doc["sentences"]):
            chars = list(sent["text"])
            if len(chars) < 3:
                continue
            labels = ["O"] * len(chars)
            for span in doc["spans"]:
                if span["start"] >= sent["start"] and span["end"] <= sent["end"]:
                    off = span["start"] - sent["start"]
                    n = len(span["text"])
                    if 0 <= off < len(chars) and off + n <= len(chars):
                        labels[off] = f"B-{span['type']}"
                        for j in range(off + 1, off + n):
                            labels[j] = f"I-{span['type']}"
            items.append({"doc": doc_id, "sent_idx": si, "start": sent["start"],
                          "chars": chars, "labels": labels})
    return items


class TokenDS(Dataset):
    def __init__(self, items, tokenizer):
        self.rows = []
        for it in items:
            enc = tokenizer(it["chars"], is_split_into_words=True, truncation=True,
                            max_length=MAX_LEN, add_special_tokens=True)
            word_ids = enc.word_ids()
            labels = []
            prev = None
            char_logits_index = []
            for pos, wi in enumerate(word_ids):
                if wi is None:
                    labels.append(-100)
                elif wi != prev:
                    labels.append(LABEL2ID[it["labels"][wi]])
                    char_logits_index.append((wi, pos))
                    prev = wi
                else:
                    labels.append(-100)
            self.rows.append({
                "doc": it["doc"], "sent_idx": it["sent_idx"], "start": it["start"],
                "chars": it["chars"],
                "input_ids": torch.tensor(enc["input_ids"]),
                "attention_mask": torch.tensor(enc["attention_mask"]),
                "labels": torch.tensor(labels),
                "char_pos": char_logits_index,
            })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def collate(batch, pad_id=0):
    max_len = max(len(r["input_ids"]) for r in batch)
    ids, masks, labels = [], [], []
    for r in batch:
        n = len(r["input_ids"])
        ids.append(torch.cat([r["input_ids"], torch.full((max_len - n,), pad_id, dtype=torch.long)]))
        masks.append(torch.cat([r["attention_mask"], torch.zeros(max_len - n, dtype=torch.long)]))
        labels.append(torch.cat([r["labels"], torch.full((max_len - n,), -100, dtype=torch.long)]))
    return {"input_ids": torch.stack(ids), "attention_mask": torch.stack(masks),
            "labels": torch.stack(labels), "rows": batch}


def decode(chars, pred_ids):
    spans, cur = [], None
    for i, (ch, pid) in enumerate(zip(chars, pred_ids)):
        lab = ID2LABEL[int(pid)]
        if lab.startswith("B-"):
            if cur:
                spans.append(cur)
            cur = {"start": i, "end": i + 1, "type": lab[2:], "text": ch}
        elif lab.startswith("I-") and cur and lab[2:] == cur["type"]:
            cur["text"] += ch
            cur["end"] = i + 1
        else:
            if cur:
                spans.append(cur)
                cur = None
    if cur:
        spans.append(cur)
    return spans


def predict_model(model, loader, device):
    model.eval()
    out = {}
    with torch.no_grad():
        for batch in loader:
            logits = model(input_ids=batch["input_ids"].to(device),
                           attention_mask=batch["attention_mask"].to(device)).logits
            for row, logits_row in zip(batch["rows"], logits):
                char_preds = [None] * len(row["chars"])
                for wi, pos in row["char_pos"]:
                    if wi < len(char_preds):
                        char_preds[wi] = int(torch.argmax(logits_row[pos]).item())
                char_preds = [p if p is not None else 0 for p in char_preds]
                spans = decode(row["chars"], char_preds)
                # 单字碎片（如“少”“安”“介”）一律丢弃：本语料没有单字实体
                out.setdefault(row["doc"], []).extend(
                    {"text": s["text"], "type": s["type"]} for s in spans if len(s["text"]) >= 2
                )
    return out


def train_fold(train_items, test_items, tokenizer, model_name, epochs, device, seed):
    ds_train = TokenDS(train_items, tokenizer)
    ds_test = TokenDS(test_items, tokenizer)
    train_loader = DataLoader(ds_train, batch_size=4, shuffle=True,
                              collate_fn=lambda b: collate(b, tokenizer.pad_token_id))
    test_loader = DataLoader(ds_test, batch_size=16, shuffle=False,
                             collate_fn=lambda b: collate(b, tokenizer.pad_token_id))

    label_counter = Counter()
    for it in train_items:
        for lab in it["labels"]:
            label_counter[lab] += 1
    total = sum(label_counter.values())
    weights = []
    for lab in LABELS:
        c = label_counter.get(lab, 0)
        if lab == "O":
            weights.append(0.5)
        elif c == 0:
            weights.append(1.5)
        else:
            # 稀有实体类型适度加权，但避免把 O 压得过低导致过预测
            weights.append(min(2.5, 1.0 + 0.8 * (total / len(LABELS)) / c))
    weight_tensor = torch.tensor(weights, dtype=torch.float, device=device)

    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=-100)
    losses = []
    for epoch in range(epochs):
        model.train()
        epoch_loss, steps = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(input_ids=batch["input_ids"].to(device),
                           attention_mask=batch["attention_mask"].to(device)).logits
            loss = loss_fn(logits.view(-1, len(LABELS)), batch["labels"].to(device).view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.item())
            steps += 1
        losses.append(round(epoch_loss / max(steps, 1), 4))
    pred = predict_model(model, test_loader, device)
    return pred, losses, len(train_items), len(test_items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base-chinese")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "pred_entities_bert.json"))
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)
    gold = load_gold()
    items = build_items(gold)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    all_pred, fold_logs = {}, []
    t0 = time.time()
    for fold, (train_idx, test_idx) in enumerate(kf.split(items), 1):
        train_items = [items[i] for i in train_idx]
        test_items = [items[i] for i in test_idx]
        pred, losses, n_train, n_test = train_fold(
            train_items, test_items, tokenizer, args.model, args.epochs, device, args.seed + fold
        )
        for doc, rows in pred.items():
            all_pred.setdefault(doc, []).extend(rows)
        fold_logs.append({"fold": fold, "train_sentences": n_train, "test_sentences": n_test,
                          "loss_curve": losses, "pred_mentions": sum(len(v) for v in pred.values())})
        print(f"[fold {fold}] train={n_train} test={n_test} loss={losses[-1]} preds={fold_logs[-1]['pred_mentions']}")

    result = {
        "method": "BERT (bert-base-chinese, BIO, 5-fold sentence CV)",
        "model": args.model,
        "epochs": args.epochs,
        "folds": fold_logs,
        "elapsed_sec": round(time.time() - t0, 2),
        "docs": {doc: rows for doc, rows in all_pred.items()},
    }
    write_json(args.out, result)
    print(f"saved {args.out} elapsed={result['elapsed_sec']}s")


if __name__ == "__main__":
    main()
