"""BiLSTM-CRF 实体抽取（PyTorch，字符级）。

特征：字符 embedding + jieba 词边界(B/M/E/S) + 字符类别 + 训练折词典特征。
不使用预训练语言模型，代表“传统深度序列标注”路线。5 折句子级交叉验证与 BERT 完全一致。
"""
from __future__ import annotations

import argparse
import random
import time
from collections import Counter
from pathlib import Path

import jieba
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

from schema import ENTITY_TYPES
from utils import ROOT, load_gold, write_json

LABELS = ["O"] + [f"{p}-{t}" for t in ENTITY_TYPES for p in ("B", "I")]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
TYPE_IDX = {t: i for i, t in enumerate(ENTITY_TYPES)}
EMB_DIM = 128
HID_DIM = 128
BOUNDARY_DIM = 4
CAT_DIM = 6
GAZ_DIM = len(ENTITY_TYPES)


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


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
            items.append({"doc": doc_id, "chars": chars, "labels": labels})
    return items


def build_vocab(items):
    chars = sorted({c for it in items for c in it["chars"]})
    return {"<PAD>": 0, "<UNK>": 1, **{c: i + 2 for i, c in enumerate(chars)}}


def char_category(ch):
    if ch.isdigit():
        return 0
    if ("a" <= ch.lower() <= "z"):
        return 1
    if "\u4e00" <= ch <= "\u9fff":
        return 2
    if ch in "，。！？；：、“”‘’（）《》—…·,.!?;:":
        return 3
    if ch.isspace():
        return 4
    return 5


def boundary_tags(text):
    tags = []
    for token in jieba.lcut(text):
        n = len(token)
        if n == 0:
            continue
        for k in range(n):
            tags.append(0 if n == 1 else (1 if k == 0 else (3 if k == n - 1 else 2)))
    while len(tags) < len(text):
        tags.append(0)
    return tags[:len(text)]


def train_entities(items):
    ents = set()
    for it in items:
        i = 0
        while i < len(it["labels"]):
            lab = it["labels"][i]
            if lab.startswith("B-"):
                t = lab[2:]; j = i + 1
                while j < len(it["labels"]) and it["labels"][j] == f"I-{t}":
                    j += 1
                ents.add(("".join(it["chars"][i:j]), t))
                i = j
            else:
                i += 1
    return ents


def gaz_features(chars, ents):
    gaz = [[0] * GAZ_DIM for _ in chars]
    text = "".join(chars)
    for surface, etype in ents:
        if not surface:
            continue
        start = text.find(surface)
        while start >= 0:
            for p in range(start, min(start + len(surface), len(chars))):
                gaz[p][TYPE_IDX[etype]] = 1
            start = text.find(surface, start + 1)
    return gaz


def make_rows(items, vocab, ents):
    rows = []
    for it in items:
        chars = it["chars"]
        feats = []
        btags = boundary_tags("".join(chars))
        gaz = gaz_features(chars, ents)
        for i, ch in enumerate(chars):
            b = [0] * BOUNDARY_DIM
            b[btags[i]] = 1
            c = [0] * CAT_DIM
            c[char_category(ch)] = 1
            feats.append([vocab.get(ch, 1)] + b + c + gaz[i])
        tags = [LABEL2ID[l] for l in it["labels"]]
        rows.append({"doc": it["doc"], "chars": chars, "feats": feats, "tags": tags})
    return rows


class SeqDS(Dataset):
    def __init__(self, rows, vocab_size):
        self.rows = rows
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return {
            "doc": r["doc"],
            "chars": r["chars"],
            "char_ids": torch.tensor([f[0] for f in r["feats"]], dtype=torch.long),
            "extra": torch.tensor([f[1:] for f in r["feats"]], dtype=torch.long),
            "tags": torch.tensor(r["tags"], dtype=torch.long),
        }


def collate(batch):
    max_len = max(len(r["char_ids"]) for r in batch)
    out = {"doc": [], "chars": [], "char_ids": [], "extra": [], "tags": [], "mask": []}
    for r in batch:
        n = len(r["char_ids"])
        out["doc"].append(r["doc"])
        out["chars"].append(r["chars"])
        out["char_ids"].append(torch.cat([r["char_ids"], torch.zeros(max_len - n, dtype=torch.long)]))
        out["extra"].append(torch.cat([r["extra"], torch.zeros(max_len - n, r["extra"].shape[1], dtype=torch.long)]))
        out["tags"].append(torch.cat([r["tags"], torch.zeros(max_len - n, dtype=torch.long)]))
        out["mask"].append(torch.cat([torch.ones(n), torch.zeros(max_len - n)]))
    for k in ("char_ids", "extra", "tags", "mask"):
        out[k] = torch.stack(out[k])
    return out


class CRF(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags) * 0.1)
        self.start_transitions = nn.Parameter(torch.randn(num_tags) * 0.1)
        self.end_transitions = nn.Parameter(torch.randn(num_tags) * 0.1)
        # BIO 合法性约束：I-X 只能从 B-X / I-X 转移而来，且不能作为序列起点
        trans_mask = torch.zeros(num_tags, num_tags)
        start_mask = torch.zeros(num_tags)
        for i, lab in enumerate(LABELS):
            if lab.startswith("I-"):
                same_b = LABEL2ID["B-" + lab[2:]]
                for j in range(num_tags):
                    if j not in (same_b, i):
                        trans_mask[j, i] = -1e4
                start_mask[i] = -1e4
        self.register_buffer("trans_mask", trans_mask)
        self.register_buffer("start_mask", start_mask)

    def _normalizer(self, emissions, mask):
        alpha = self.start_transitions + self.start_mask + emissions[:, 0]
        for i in range(1, emissions.shape[1]):
            nxt = torch.logsumexp(alpha.unsqueeze(2) + (self.transitions + self.trans_mask).unsqueeze(0) + emissions[:, i].unsqueeze(1), dim=1)
            alpha = torch.where(mask[:, i].unsqueeze(1).bool(), nxt, alpha)
        return torch.logsumexp(alpha + self.end_transitions, dim=1)

    def forward(self, emissions, tags, mask):
        batch, seq, _ = emissions.shape
        score = (self.start_transitions + self.start_mask)[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)
        for i in range(1, seq):
            emit = emissions[:, i].gather(1, tags[:, i].unsqueeze(1)).squeeze(1)
            trans = (self.transitions + self.trans_mask)[tags[:, i - 1], tags[:, i]]
            score = score + mask[:, i] * (emit + trans)
        last = mask.sum(1).long() - 1
        score = score + self.end_transitions[tags[torch.arange(batch), last]]
        return self._normalizer(emissions, mask) - score

    def viterbi(self, emissions, mask):
        batch, seq, _ = emissions.shape
        alpha = self.start_transitions + self.start_mask + emissions[:, 0]
        back = []
        for i in range(1, seq):
            nxt = alpha.unsqueeze(2) + (self.transitions + self.trans_mask).unsqueeze(0)
            best, idx = nxt.max(dim=1)
            alpha = torch.where(mask[:, i].unsqueeze(1).bool(), best + emissions[:, i], alpha)
            back.append(idx)
        final = alpha + self.end_transitions
        best_last = final.argmax(1)
        lengths = mask.sum(1).long().tolist()
        paths = []
        for b in range(batch):
            tag = int(best_last[b].item())
            path = [tag]
            for i in range(seq - 1, 0, -1):
                if i >= lengths[b]:
                    continue
                tag = int(back[i - 1][b, tag].item())
                path.append(tag)
            paths.append(list(reversed(path)))
        return paths


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, num_tags):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMB_DIM, padding_idx=0)
        self.lstm = nn.LSTM(EMB_DIM + BOUNDARY_DIM + CAT_DIM + GAZ_DIM, HID_DIM,
                            num_layers=1, bidirectional=True, batch_first=True)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(HID_DIM * 2, num_tags)
        self.crf = CRF(num_tags)

    def emissions(self, char_ids, extra):
        x = torch.cat([self.emb(char_ids), extra.float()], dim=-1)
        h, _ = self.lstm(x)
        return self.fc(self.drop(h))


def decode_spans(chars, tag_ids):
    spans, cur = [], None
    for i, (ch, tid) in enumerate(zip(chars, tag_ids)):
        lab = ID2LABEL[int(tid)]
        if lab.startswith("B-"):
            if cur: spans.append(cur)
            cur = {"text": ch, "type": lab[2:], "start": i, "end": i + 1}
        elif lab.startswith("I-") and cur and lab[2:] == cur["type"]:
            cur["text"] += ch; cur["end"] = i + 1
        else:
            if cur: spans.append(cur); cur = None
    if cur: spans.append(cur)
    return spans


def run_fold(train_items, test_items, vocab, epochs, device, seed):
    ents = train_entities(train_items)
    rows_train = make_rows(train_items, vocab, ents)
    rows_test = make_rows(test_items, vocab, ents)
    train_loader = DataLoader(SeqDS(rows_train, len(vocab)), batch_size=8, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(SeqDS(rows_test, len(vocab)), batch_size=16, shuffle=False, collate_fn=collate)

    model = BiLSTMCRF(len(vocab), len(LABELS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    ce_weights = torch.tensor([0.4] + [1.6] * (len(LABELS) - 1), device=device)
    ce_loss = nn.CrossEntropyLoss(weight=ce_weights)
    losses = []
    for epoch in range(epochs):
        model.train()
        total, steps = 0.0, 0
        for b in train_loader:
            opt.zero_grad()
            emissions = model.emissions(b["char_ids"].to(device), b["extra"].to(device))
            mask = b["mask"].to(device)
            # CRF.forward 返回的是正 NLL（logZ - gold_score），直接最小化
            loss = model.crf(emissions, b["tags"].to(device), mask).mean()
            flat_em = emissions[mask.bool()]
            flat_tags = b["tags"].to(device)[mask.bool()]
            loss = loss + 0.8 * ce_loss(flat_em, flat_tags)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()); steps += 1
        losses.append(round(total / max(steps, 1), 4))

    model.eval()
    pred = {}
    with torch.no_grad():
        for b in test_loader:
            emissions = model.emissions(b["char_ids"].to(device), b["extra"].to(device))
            paths = model.crf.viterbi(emissions, b["mask"].to(device))
            for doc, chars, path in zip(b["doc"], b["chars"], paths):
                for s in decode_spans(chars, path):
                    if len(s["text"]) >= 2:
                        pred.setdefault(doc, []).append({"text": s["text"], "type": s["type"]})
    return pred, losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "pred_entities_bilstm.json"))
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)
    gold = load_gold()
    items = build_items(gold)
    vocab = build_vocab(items)
    kf = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    all_pred, logs = {}, []
    t0 = time.time()
    for fold, (tr, te) in enumerate(kf.split(items), 1):
        pred, losses = run_fold([items[i] for i in tr], [items[i] for i in te], vocab, args.epochs, device, args.seed + fold)
        for doc, rows in pred.items():
            all_pred.setdefault(doc, []).extend(rows)
        logs.append({"fold": fold, "train_sentences": len(tr), "test_sentences": len(te),
                     "loss_curve": losses, "pred_mentions": sum(len(v) for v in pred.values())})
        print(f"[fold {fold}] loss={losses[-1]} preds={logs[-1]['pred_mentions']}")

    result = {"method": "BiLSTM-CRF (char features, 5-fold sentence CV)", "folds": logs,
              "elapsed_sec": round(time.time() - t0, 2), "docs": all_pred}
    write_json(args.out, result)
    print(f"saved {args.out} elapsed={result['elapsed_sec']}s")


if __name__ == "__main__":
    main()
