#!/usr/bin/env python3
"""Parse FAQ-Endo.txt → Q/A cards → embed questions → append to Qdrant.

Adds to the existing Clinic2 collection (does NOT wipe it).
Card ids: endo_<num>, payload.source = "endo".

Endo PDF text mixes three header styles:
  - `- 2 question؟` at line start
  - bare `27` line then questions
  - mid-line `… -11 دندانی) چند کانال دارد؟` (section id stuck inside Q variants)
"""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

ROOT = Path(__file__).resolve().parents[2]
RAG = Path(__file__).resolve().parents[1]
SRC = ROOT / "faqs" / "sources" / "endo.txt"
OUT = RAG / "data" / "endo"
COLLECTION = os.environ.get("QDRANT_COLLECTION", "clinic2_faq_qa")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
MODEL_ID = os.environ.get("EMBED_MODEL", "jinaai/jina-embeddings-v5-text-small")
EMBED_DIM = 1024

FIXES = {
    "اصطالحات": "اصطلاحات",
    "اصطالحا": "اصطلاحا",
    "تالش": "تلاش",
    "قبالً": "قبلاً",
    "قبال": "قبلا",
    "الزم": "لازم",
    "الیهای": "لایه‌ای",
    "الیه": "لایه",
    "پالک": "پلاک",
    "مالحظات": "ملاحظات",
    "مالحظاتی": "ملاحظاتی",
    "اطالع": "اطلاع",
    "طوالنی": "طولانی",
    "معموالً": "معمولاً",
    "معموال": "معمولا",
    "باال": "بالا",
    "مثالً": "مثلاً",
    "عالئم": "علائم",
    "ریفالکس": "ریفلاکس",
    "دندنپزشک": "دندانپزشک",
    "دالیل": "دلایل",
    "مشکالت": "مشکلات",
}


def clean_text(text: str) -> str:
    text = re.sub(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]", "", text)
    for bad, good in FIXES.items():
        text = text.replace(bad, good)
    # drop English citation / journal lines mixed into the PDF extract
    text = re.sub(
        r"[A-Za-z][A-Za-z0-9 ,\.\-–—'\":;/()]{15,}(?:\d{4})?[A-Za-z0-9 ,\.\-–—'\":;/()]{0,80}",
        " ",
        text,
    )
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


def _trailing_question_block(left: str) -> tuple[int, str]:
    """Start index + text of trailing short Persian FAQ questions in `left`."""
    marks = list(re.finditer(r"[؟?]", left))
    if not marks:
        return len(left), ""
    segs: list[tuple[int, str]] = []
    for i, mm in enumerate(marks):
        end = mm.end()
        prev_end = marks[i - 1].end() if i else max(0, mm.start() - 90)
        start = prev_end
        # prefer newline start if present
        nl = left.rfind("\n", prev_end, mm.start())
        if nl >= prev_end:
            start = nl + 1
        seg = left[start:end].strip()
        if len(seg) < 5 or len(seg) > 140:
            continue
        ascii_ratio = sum(1 for c in seg if ord(c) < 128) / max(len(seg), 1)
        if ascii_ratio > 0.45:
            continue
        segs.append((start, seg))
    if not segs:
        return len(left), ""
    # keep only a trailing cluster near the marker
    cluster = [segs[-1]]
    for s in reversed(segs[:-1]):
        if cluster[0][0] - (s[0] + len(s[1])) > 40:
            break
        cluster.insert(0, s)
        if len(cluster) >= 6:
            break
    return cluster[0][0], " ".join(s for _, s in cluster)


def normalize_inline_section_ids(text: str) -> str:
    """Move mid-line '-11'/'-16' markers to a proper card header."""
    matches = list(re.finditer(r"(?<![-\d\n])-(\d{1,3})(?=\s)", text))
    for m in reversed(matches):
        n = int(m.group(1))
        left = text[max(0, m.start() - 280) : m.start()]
        right = text[m.end() : m.end() + 160]
        if n <= 4 and "؟" not in left[-80:] and "؟" not in right[:80]:
            continue
        left_marks = list(re.finditer(r"[؟?]", left))
        gap_after_q = (len(left) - left_marks[-1].end()) if left_marks else 999
        # If a long answer sits between the last ؟ and "-N", questions for this
        # card continue AFTER the marker (e.g. care FAQ "-25").
        if gap_after_q > 80:
            begin_in_left, qblock = len(left), ""
        else:
            begin_in_left, qblock = _trailing_question_block(left)
        cont = ""
        rm = re.match(r"^(\s*[^؟?\n]{0,100}[؟?])", right)
        if rm:
            frag = rm.group(1).strip()
            ascii_ratio = sum(1 for c in frag if ord(c) < 128) / max(len(frag), 1)
            if ascii_ratio <= 0.45 and len(frag) <= 120:
                if qblock and not qblock.endswith(("؟", "?")):
                    cont = frag
                    qblock = (qblock + " " + cont).strip()
                elif not qblock:
                    cont = frag
                    qblock = cont
        if "؟" not in qblock and "?" not in qblock:
            continue
        abs_start = max(0, m.start() - 280) + begin_in_left
        while abs_start < m.start() and text[abs_start].isspace():
            abs_start += 1
        consume_end = m.end() + (len(rm.group(0)) if cont and rm else 0)
        header = f"\n- {n} {qblock} "
        text = text[:abs_start] + header + text[consume_end:]
    return text


def _has_question(peek: str) -> bool:
    return "؟" in peek or "?" in peek


def find_card_starts(text: str) -> list[tuple[int, int]]:
    """Return (char_offset, section_num) for real FAQ cards."""
    cands: list[tuple[int, int]] = []

    # A) lone number line (later Endo sections)
    for m in re.finditer(r"(?m)^(?:-\s*)?(\d{1,3})\s*$", text):
        if _has_question(text[m.end() : m.end() + 220]):
            cands.append((m.start(), int(m.group(1))))

    # B) line-start "- N …"
    for m in re.finditer(r"(?m)^-\s*(\d{1,3})\s+\S", text):
        if _has_question(text[m.start() : m.start() + 220]):
            cands.append((m.start(), int(m.group(1))))

    cands.sort(key=lambda x: (x[0], x[1]))
    uniq: list[tuple[int, int]] = []
    seen_pos = set()
    for pos, n in cands:
        if pos in seen_pos:
            continue
        seen_pos.add(pos)
        uniq.append((pos, n))

    kept: list[tuple[int, int]] = []
    last = 0
    for pos, n in uniq:
        if not kept:
            kept.append((pos, n))
            last = n
            continue
        if n <= 4 and last >= 9:
            continue
        if n > last or (n == last and n >= 10):
            kept.append((pos, n))
            last = n
    return kept


def parse_card(body: str) -> tuple[list[str], str]:
    body = " ".join(body.split())
    questions: list[str] = []
    while True:
        m = re.match(r"^(.{3,160}?[؟?])\s*", body)
        if not m:
            break
        q = m.group(1).strip()
        ascii_ratio = sum(1 for c in q if ord(c) < 128) / max(len(q), 1)
        # drop citations / answer-shaped "questions"
        if (
            len(q) >= 5
            and len(q) <= 140
            and ascii_ratio <= 0.45
            and not re.match(r"^[A-Za-z0-9,\.\-–—\s]+$", q)
        ):
            questions.append(q)
        body = body[m.end() :]
    answer = re.sub(r"^[-–—]\s*", "", body).strip()
    return questions, answer


CARD1 = {
    "id": "endo_1",
    "section": 1,
    "source": "endo",
    "questions": [
        "عصب کشی درسته یا عصب کُشی؟",
        "عصب کشی یا عصب کُشی؟",
        "نام درست عصب کشی چیه؟",
        "نام درست درمان عصب‌کشی چیست؟",
    ],
    "answer": (
        'هر دو اشتباه است و "درمان ریشه" یا روت کانال تراپی نام درست این درمان است.'
    ),
    "n_questions": 4,
    "answer_len": 0,
}


def _card_id(section: int, used: dict[int, int]) -> str:
    used[section] = used.get(section, 0) + 1
    k = used[section]
    return f"endo_{section}" if k == 1 else f"endo_{section}b"


def build_records() -> list[dict]:
    raw = SRC.read_text(encoding="utf-8")
    text = clean_text(raw)
    text = re.sub(r"(?m)^FAQ-Endo\s*$", "", text)
    text = normalize_inline_section_ids(text)
    starts = find_card_starts(text)

    records = [dict(CARD1, answer_len=len(CARD1["answer"]))]
    used_ids: dict[int, int] = {1: 1}

    for idx, (pos, num) in enumerate(starts):
        if num == 1 and idx == 0:
            # may match garbled header; skip in favor of CARD1
            continue
        end = starts[idx + 1][0] if idx + 1 < len(starts) else len(text)
        chunk = text[pos:end]
        # strip header marker
        chunk = re.sub(r"^(?:-\s*)?\d{1,3}\s*", "", chunk, count=1)
        chunk = re.sub(r"^-\d{1,3}\s*", "", chunk, count=1)
        questions, answer = parse_card(chunk)
        if num == 1:
            continue
        cid = _card_id(num, used_ids)
        records.append(
            {
                "id": cid,
                "section": num,
                "source": "endo",
                "questions": questions,
                "answer": answer,
                "n_questions": len(questions),
                "answer_len": len(answer),
            }
        )
    return records


def print_review(records: list[dict]) -> None:
    print(f"Parsed {len(records)} Endo cards")
    nums = [r["section"] for r in records]
    print(f"Sections: {nums[:20]} ... {nums[-5:]}")
    bad = [r for r in records if r["n_questions"] == 0 or r["answer_len"] < 20]
    print(f"Needs review (0 Q or short A): {len(bad)}")
    for r in bad[:20]:
        print(f"  - {r['id']}: qs={r['n_questions']} ans_len={r['answer_len']}")
    # eval-critical cards
    for key in (
        "کانال",
        "ضایعه",
        "آبسه",
        "اپیکو",
        "کلسیفیه",
        "رابردم",
        "فایل چیه",
        "فایل یعنی",
        "عفونت پای",
    ):
        hits = [
            r["id"]
            for r in records
            if any(key in q for q in r["questions"])
            or key in (r.get("answer") or "")[:200]
        ]
        print(f"  keyword {key!r} -> {hits[:5]}")


def delete_existing_endo(client: QdrantClient) -> None:
    if not client.collection_exists(COLLECTION):
        return
    try:
        client.delete(
            collection_name=COLLECTION,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="source", match=qm.MatchValue(value="endo")
                        )
                    ]
                )
            ),
        )
        print("Removed previous source=endo points (if any)")
    except Exception as e:
        print(f"Note: could not filter-delete endo points ({e}); continuing")


def index_questions(records: list[dict]) -> int:
    import torch
    from sentence_transformers import SentenceTransformer

    device = os.environ.get(
        "EMBED_DEVICE",
        "cuda" if torch.cuda.is_available() else "cpu",
    )
    # if llama holds the GPU, fall back to CPU without failing
    if device.startswith("cuda"):
        try:
            free, _ = torch.cuda.mem_get_info()
            if free < 2 * 1024**3:
                print(f"GPU free {free/1024**2:.0f}MiB < 2GiB → using cpu")
                device = "cpu"
        except Exception:
            device = "cpu"

    print(f"Loading {MODEL_ID} on {device} (local cache only)...")
    kwargs = {"trust_remote_code": True, "device": device, "local_files_only": True}
    if device.startswith("cuda"):
        kwargs["model_kwargs"] = {"torch_dtype": torch.bfloat16}
    model = SentenceTransformer(MODEL_ID, **kwargs)

    client = QdrantClient(url=QDRANT_URL, check_compatibility=False)
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            COLLECTION,
            vectors_config=qm.VectorParams(size=EMBED_DIM, distance=qm.Distance.COSINE),
        )
        print(f"Created collection `{COLLECTION}`")
    else:
        n = client.get_collection(COLLECTION).points_count or 0
        print(f"Appending to `{COLLECTION}` ({n} existing points)")
        delete_existing_endo(client)

    points = []
    usable = [r for r in records if r["questions"] and r["answer_len"] >= 10]
    print(f"Embedding {len(usable)} cards on {device}...")
    for rec in usable:
        vecs = model.encode(
            rec["questions"],
            task="retrieval",
            prompt_name="document",
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        for q, v in zip(rec["questions"], vecs):
            points.append(
                qm.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=v.tolist(),
                    payload={
                        "id": rec["id"],
                        "section": rec["section"],
                        "source": "endo",
                        "question": q,
                        "answer": rec["answer"],
                        "all_questions": rec["questions"],
                    },
                )
            )
    client.upload_points(COLLECTION, points, batch_size=64, wait=True)
    total = client.get_collection(COLLECTION).points_count or 0
    print(f"Indexed {len(points)} Endo question vectors → `{COLLECTION}` (now {total} pts)")
    return len(points)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = build_records()
    print_review(records)
    (OUT / "records.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    n = index_questions(records)
    print(f"Done. records → {OUT / 'records.json'} ({n} vectors)")


if __name__ == "__main__":
    main()
