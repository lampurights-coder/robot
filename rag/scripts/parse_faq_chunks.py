#!/usr/bin/env python3
"""Step 1a: Parse FAQ .txt into structured Q&A chunk candidates (no LLM).

Supports two layouts found in the corpus:

A) Perio-style
     1
     - question?
     - variant?
     answer paragraph...

B) Endo-style
     - 2 question text that may wrap
       across lines?
     answer paragraph...
     - 3 next question...

Writes:
  data/chunks_raw.jsonl
  data/chunk_map.md
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAG = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "faqs" / "sources"
OUT = RAG / "data"
BIDI = re.compile(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
SECTION_LONE = re.compile(r"^\d+$")
SECTION_DASH = re.compile(r"^-\s*(\d+)\s+(.*)$")


def clean(s: str) -> str:
    s = BIDI.sub("", s)
    s = unicodedata.normalize("NFC", s)
    return s.strip()


def flush_chunk(
    chunks: list[dict],
    *,
    topic: str,
    source: str,
    section_id: int,
    start_line: int,
    questions: list[str],
    answer_lines: list[str],
) -> None:
    answer = re.sub(r"\s+", " ", " ".join(answer_lines)).strip()
    questions = [re.sub(r"\s+", " ", q).strip() for q in questions if q.strip()]
    if not questions and not answer:
        return
    chunk_id = f"{topic}::sec-{section_id}"
    # disambiguate duplicate section ids
    n_dup = sum(1 for c in chunks if c["chunk_id"] == chunk_id or c["chunk_id"].startswith(chunk_id + "#"))
    if n_dup:
        chunk_id = f"{chunk_id}#{n_dup+1}"
    chunks.append(
        {
            "chunk_id": chunk_id,
            "topic": topic,
            "source_file": source,
            "section_id": section_id,
            "start_line": start_line,
            "questions": questions,
            "answer": answer,
            "chunk_text": "\n".join(
                [f"Q: {q}" for q in questions] + ([f"A: {answer}"] if answer else [])
            ),
            "boundary_rule": "section_id_or_dash_number",
        }
    )


def parse_file(path: Path) -> list[dict]:
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    lines = [clean(x) for x in raw_lines]
    topic = re.sub(r"^\d+_", "", path.stem)
    chunks: list[dict] = []

    section_id: int | None = None
    start_line = 0
    questions: list[str] = []
    answer_lines: list[str] = []
    mode: str | None = None  # 'q' collecting questions, 'a' answer

    def start_new(sid: int, line_no: int, first_q: str | None = None) -> None:
        nonlocal section_id, start_line, questions, answer_lines, mode
        if section_id is not None:
            flush_chunk(
                chunks,
                topic=topic,
                source=path.name,
                section_id=section_id,
                start_line=start_line,
                questions=questions,
                answer_lines=answer_lines,
            )
        section_id = sid
        start_line = line_no
        questions = [first_q] if first_q else []
        answer_lines = []
        mode = "q"

    for idx, line in enumerate(lines, start=1):
        if not line:
            continue

        # Layout A: lone integer
        if SECTION_LONE.fullmatch(line):
            start_new(int(line), idx, None)
            continue

        # Layout B: "- N question..."
        m = SECTION_DASH.match(line)
        if m:
            start_new(int(m.group(1)), idx, m.group(2).strip() or None)
            continue

        if section_id is None:
            continue

        if line.startswith("-"):
            q = line.lstrip("-").strip()
            # skip if it's another "- N ..." already handled
            if re.match(r"^\d+\s+", q):
                continue
            if q in {"?", "؟"} and questions:
                questions[-1] = questions[-1].rstrip("؟?") + "؟"
            elif q:
                questions.append(q)
            mode = "q"
            continue

        # continuation / answer
        if mode == "q" and questions and not questions[-1].endswith(("؟", "?", "!", "۔")):
            # likely wrapped question
            questions[-1] = (questions[-1] + " " + line).strip()
        else:
            mode = "a"
            answer_lines.append(line)

    if section_id is not None:
        flush_chunk(
            chunks,
            topic=topic,
            source=path.name,
            section_id=section_id,
            start_line=start_line,
            questions=questions,
            answer_lines=answer_lines,
        )
    return chunks


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_chunks: list[dict] = []
    map_lines = [
        "# FAQ chunk map (where chunks start)",
        "",
        "Boundary rules:",
        "1. A lone integer line starts a chunk (Perio-style).",
        "2. A line matching `- N ...` starts a chunk (Endo-style).",
        "3. Other `- ...` lines are question variants; following text is the answer.",
        "",
    ]

    src_dir = SOURCES if SOURCES.is_dir() else ROOT
    for path in sorted(src_dir.glob("*.txt")):
        chunks = parse_file(path)
        all_chunks.extend(chunks)
        map_lines.append(f"## {path.name} — {len(chunks)} chunks")
        for c in chunks[:8]:
            qs = " | ".join(c["questions"][:2]) or "(no Q)"
            map_lines.append(
                f"- line {c['start_line']}: `{c['chunk_id']}` — {qs[:140]}"
            )
        if len(chunks) > 8:
            map_lines.append(f"- … +{len(chunks) - 8} more")
        map_lines.append("")

    raw_path = OUT / "chunks_raw.jsonl"
    with raw_path.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    (OUT / "chunk_map.md").write_text("\n".join(map_lines), encoding="utf-8")
    print(f"wrote {len(all_chunks)} chunks -> {raw_path}")
    print(f"map -> {OUT / 'chunk_map.md'}")


if __name__ == "__main__":
    main()
