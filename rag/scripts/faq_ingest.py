#!/usr/bin/env python3
"""Generic FAQ → Qdrant ingestion for Tita-Arshia RAG.

Usage (CLI):
  python scripts/faq_ingest.py prosthesis
  python scripts/faq_ingest.py 1745874242733_FAQ-Prosthesis
  python scripts/faq_ingest.py /path/to/file.txt --source prosthesis
  EMBED_DEVICE=cuda python scripts/faq_ingest.py restorative

Programmatic:
  from faq_ingest import ingest_faq
  stats = ingest_faq("prosthesis")
  stats = ingest_faq("1745874242733_FAQ-Prosthesis")
"""

from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

ROOT = Path(__file__).resolve().parents[2]
RAG = Path(__file__).resolve().parents[1]
SOURCES_DIR = ROOT / "faqs" / "sources"
DATA_DIR = RAG / "data"

COLLECTION = os.environ.get("QDRANT_COLLECTION", "clinic2_faq_qa")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
MODEL_ID = os.environ.get("EMBED_MODEL", "jinaai/jina-embeddings-v5-text-small")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "1024"))

# timestamped PDF names → clean source id
LEGACY_NAME_MAP = {
    "1745874239755_FAQ-Endo": "endo",
    "1745874241233_FAQ-Diagnostics": "diagnostics",
    "1745874242733_FAQ-Prosthesis": "prosthesis",
    "1745874244336_FAQ-Restorative": "restorative",
    "FAQ-Clinic2": "clinic2",
    "FAQ-General Anesthesia": "anesthesia",
    "FAQ-Implant": "implant",
    "FAQ-Ortho(2)": "ortho",
    "FAQ-Pediatrics": "pediatrics",
    "FAQ-Perio": "perio",
    "FAQ-Radio": "radio",
    "FAQ-Surgery(2)": "surgery",
}

# short id prefix in Qdrant (keep readable; clinic2 stays clinic2_)
PREFIX_MAP = {
    "diagnostics": "diag",
    "prosthesis": "pros",
    "restorative": "rest",
    "anesthesia": "anes",
    "pediatrics": "ped",
    "surgery": "surg",
    "implant": "impl",
    "clinic2": "clinic2",
    "endo": "endo",
    "perio": "perio",
    "ortho": "ortho",
    "radio": "radio",
}

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
    "سالمت": "سلامت",
}


# ---------------------------------------------------------------------------
# path / name resolution
# ---------------------------------------------------------------------------

def resolve_source(spec: str | Path) -> tuple[Path, str]:
    """Return (txt_path, source_id) from a short name, legacy name, or file path."""
    raw = str(spec).strip()
    p = Path(raw).expanduser()

    if p.is_file():
        stem = p.stem
        source = LEGACY_NAME_MAP.get(stem) or _stem_to_source(stem)
        return p.resolve(), source

    # strip extension if user passed foo.txt
    key = raw
    if key.endswith(".txt") or key.endswith(".html"):
        key = Path(key).stem

    if key in LEGACY_NAME_MAP:
        source = LEGACY_NAME_MAP[key]
    else:
        source = _stem_to_source(key)

    # FAQ-Prosthesis → prosthesis
    source = source.lower().replace(" ", "_")

    candidate = SOURCES_DIR / f"{source}.txt"
    if candidate.is_file():
        return candidate, source

    # try original name under sources (shouldn't exist after rename)
    alt = SOURCES_DIR / f"{key}.txt"
    if alt.is_file():
        return alt, source

    raise FileNotFoundError(
        f"FAQ source not found for {spec!r}. Tried {candidate}. "
        f"Available: {sorted(x.stem for x in SOURCES_DIR.glob('*.txt'))}"
    )


def _stem_to_source(stem: str) -> str:
    m = re.search(r"FAQ[-_]?(.+)$", stem, flags=re.I)
    if m:
        name = m.group(1)
    else:
        name = stem
    name = re.sub(r"\(\d+\)$", "", name).strip()
    name = name.replace(" ", "_")
    return name.lower()


def id_prefix(source: str) -> str:
    return PREFIX_MAP.get(source, source[:6])


# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    text = re.sub(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]", "", text)
    for bad, good in FIXES.items():
        text = text.replace(bad, good)
    text = re.sub(
        r"[A-Za-z][A-Za-z0-9 ,\.\-–—'\":;/()]{15,}(?:\d{4})?[A-Za-z0-9 ,\.\-–—'\":;/()]{0,80}",
        " ",
        text,
    )
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


def _trailing_question_block(left: str) -> tuple[int, str]:
    marks = list(re.finditer(r"[؟?]", left))
    if not marks:
        return len(left), ""
    segs: list[tuple[int, str]] = []
    for i, mm in enumerate(marks):
        end = mm.end()
        prev_end = marks[i - 1].end() if i else max(0, mm.start() - 90)
        start = prev_end
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
    cluster = [segs[-1]]
    for s in reversed(segs[:-1]):
        if cluster[0][0] - (s[0] + len(s[1])) > 40:
            break
        cluster.insert(0, s)
        if len(cluster) >= 6:
            break
    return cluster[0][0], " ".join(s for _, s in cluster)


def normalize_inline_section_ids(text: str) -> str:
    matches = list(re.finditer(r"(?<![-\d\n])-(\d{1,3})(?=\s)", text))
    for m in reversed(matches):
        n = int(m.group(1))
        left = text[max(0, m.start() - 280) : m.start()]
        right = text[m.end() : m.end() + 160]
        if n <= 4 and "؟" not in left[-80:] and "؟" not in right[:80]:
            continue
        left_marks = list(re.finditer(r"[؟?]", left))
        gap_after_q = (len(left) - left_marks[-1].end()) if left_marks else 999
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
    cands: list[tuple[int, int]] = []
    for m in re.finditer(r"(?m)^(?:-\s*)?(\d{1,3})\s*$", text):
        if _has_question(text[m.end() : m.end() + 220]):
            cands.append((m.start(), int(m.group(1))))
    for m in re.finditer(r"(?m)^-\s*(\d{1,3})\s+\S", text):
        if _has_question(text[m.start() : m.start() + 220]):
            cands.append((m.start(), int(m.group(1))))
    cands.sort(key=lambda x: (x[0], x[1]))
    uniq: list[tuple[int, int]] = []
    seen = set()
    for pos, n in cands:
        if pos in seen:
            continue
        seen.add(pos)
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


def build_records(txt_path: Path, source: str, prefix: str | None = None) -> list[dict]:
    prefix = prefix or id_prefix(source)
    raw = txt_path.read_text(encoding="utf-8")
    text = clean_text(raw)
    text = re.sub(r"(?m)^FAQ-[^\n]*\s*$", "", text)
    text = normalize_inline_section_ids(text)
    starts = find_card_starts(text)

    records: list[dict] = []
    used: dict[int, int] = {}
    for idx, (pos, num) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else len(text)
        chunk = text[pos:end]
        chunk = re.sub(r"^(?:-\s*)?\d{1,3}\s*", "", chunk, count=1)
        chunk = re.sub(r"^-\d{1,3}\s*", "", chunk, count=1)
        questions, answer = parse_card(chunk)
        used[num] = used.get(num, 0) + 1
        cid = f"{prefix}_{num}" if used[num] == 1 else f"{prefix}_{num}b"
        records.append(
            {
                "id": cid,
                "section": num,
                "source": source,
                "questions": questions,
                "answer": answer,
                "n_questions": len(questions),
                "answer_len": len(answer),
            }
        )
    return records


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------

def delete_source_points(client: QdrantClient, source: str, collection: str) -> None:
    if not client.collection_exists(collection):
        return
    try:
        client.delete(
            collection_name=collection,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="source", match=qm.MatchValue(value=source)
                        )
                    ]
                )
            ),
        )
        print(f"Removed previous source={source!r} points (if any)")
    except Exception as e:
        print(f"Note: could not filter-delete {source} points ({e}); continuing")


def index_records(
    records: list[dict],
    *,
    source: str,
    collection: str = COLLECTION,
    qdrant_url: str = QDRANT_URL,
    embed_device: str | None = None,
) -> int:
    import torch
    from sentence_transformers import SentenceTransformer

    device = embed_device or os.environ.get(
        "EMBED_DEVICE",
        "cuda" if torch.cuda.is_available() else "cpu",
    )
    if device.startswith("cuda"):
        try:
            free, _ = torch.cuda.mem_get_info()
            if free < 2 * 1024**3:
                print(f"GPU free {free/1024**2:.0f}MiB < 2GiB → using cpu")
                device = "cpu"
        except Exception:
            device = "cpu"

    offline = os.environ.get("HF_HUB_OFFLINE", "0") in ("1", "true", "True")
    print(f"Loading {MODEL_ID} on {device}...")
    kwargs: dict = {
        "trust_remote_code": True,
        "device": device,
        "local_files_only": offline,
    }
    if device.startswith("cuda"):
        kwargs["model_kwargs"] = {"torch_dtype": torch.bfloat16}
    model = SentenceTransformer(MODEL_ID, **kwargs)

    client = QdrantClient(url=qdrant_url, check_compatibility=False)
    if not client.collection_exists(collection):
        client.create_collection(
            collection,
            vectors_config=qm.VectorParams(size=EMBED_DIM, distance=qm.Distance.COSINE),
        )
        print(f"Created collection `{collection}`")
    else:
        n = client.get_collection(collection).points_count or 0
        print(f"Appending to `{collection}` ({n} existing points)")
        delete_source_points(client, source, collection)

    usable = [r for r in records if r["questions"] and r["answer_len"] >= 10]
    print(f"Embedding {len(usable)}/{len(records)} cards on {device}...")
    points = []
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
                        "source": source,
                        "question": q,
                        "answer": rec["answer"],
                        "all_questions": rec["questions"],
                    },
                )
            )
    client.upload_points(collection, points, batch_size=64, wait=True)
    total = client.get_collection(collection).points_count or 0
    print(f"Indexed {len(points)} vectors → `{collection}` (now {total} pts)")
    return len(points)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def ingest_faq(
    spec: str | Path,
    *,
    source: str | None = None,
    prefix: str | None = None,
    collection: str = COLLECTION,
    qdrant_url: str = QDRANT_URL,
    embed_device: str | None = None,
    save_records: bool = True,
) -> dict:
    """Parse an FAQ .txt and upsert question vectors into Qdrant.

    Parameters
    ----------
    spec:
        Short name (`prosthesis`), legacy name (`1745874242733_FAQ-Prosthesis`),
        or path to a `.txt` file.
    source:
        Override Qdrant `source` / payload tag (default derived from `spec`).
    prefix:
        Card id prefix (default from PREFIX_MAP), e.g. `pros_1`.
    """
    txt_path, resolved = resolve_source(spec)
    source = source or resolved
    prefix = prefix or id_prefix(source)

    print(f"Ingest FAQ  source={source!r}  file={txt_path}")
    records = build_records(txt_path, source=source, prefix=prefix)
    bad = [r for r in records if r["n_questions"] == 0 or r["answer_len"] < 20]
    print(f"Parsed {len(records)} cards  (needs_review={len(bad)})")
    if records:
        print(f"  sections: {[r['section'] for r in records[:12]]} ...")
        r0 = records[0]
        print(f"  sample {r0['id']}: Q={r0['questions'][:2]} A_len={r0['answer_len']}")

    out_dir = DATA_DIR / source
    if save_records:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "records.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Wrote {out_dir / 'records.json'}")

    n_vec = index_records(
        records,
        source=source,
        collection=collection,
        qdrant_url=qdrant_url,
        embed_device=embed_device,
    )
    stats = {
        "source": source,
        "file": str(txt_path),
        "cards": len(records),
        "vectors": n_vec,
        "needs_review": len(bad),
        "collection": collection,
        "records_path": str(out_dir / "records.json") if save_records else None,
    }
    print(f"Done: {stats}")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest a FAQ .txt into Qdrant RAG")
    ap.add_argument(
        "spec",
        nargs="+",
        help="Source name (prosthesis|anesthesia), legacy name, or .txt path "
        "(quote names with spaces, or pass words unquoted — they are joined)",
    )
    ap.add_argument("--source", default=None, help="Override source tag")
    ap.add_argument("--prefix", default=None, help="Override card id prefix")
    ap.add_argument("--collection", default=COLLECTION)
    ap.add_argument("--device", default=None, help="cuda|cpu (default EMBED_DEVICE/auto)")
    ap.add_argument("--no-save", action="store_true", help="Skip writing records.json")
    args = ap.parse_args()
    spec = " ".join(args.spec) if isinstance(args.spec, list) else args.spec
    ingest_faq(
        spec,
        source=args.source,
        prefix=args.prefix,
        collection=args.collection,
        embed_device=args.device,
        save_records=not args.no_save,
    )


if __name__ == "__main__":
    main()
