# FAQ RAG eval report

- Generated: `2026-10-13 13:45:29 UTC`
- Device: `cuda`
- Reranker: `True`
- Suites: `eval-20-8`

## Combined

| Metric | Score |
|--------|------:|
| n | 20 |
| Retrieval @0.70 | 17/20 (85.0%) |
| Retrieval @0.55 | 19/20 (95.0%) |
| E2E @0.65 | 17/20 (85.0%) |
| Refused | 1/20 (5.0%) |

## Per suite

| Suite | n | Ret@0.70 | Ret@0.55 | E2E@0.65 | Refused |
|-------|--:|--------:|--------:|---------:|--------:|
| `eval-20-8` | 20 | 17/20 (85%) | 19/20 (95%) | 17/20 (85%) | 1/20 |

## Misses / refuses

| Suite | # | Status | Card | Question |
|-------|--:|--------|------|----------|
| `eval-20-8` | 9 | SOFT/E2E✗ | `impl_11` | ایمپلنتم بیرون اومده یا لق شده، چیکار کنم؟ |
| `eval-20-8` | 17 | REFUSE | `endo_97` | استئواینتگریشن یعنی چی؟ |
| `eval-20-8` | 18 | SOFT/E2E✗ | `impl_71b` | به جز ایمپلنت چطوری می‌شه دندون گذاشت؟ |

## Artifacts

- Combined JSON: `/home/ai/ollm-ocr-2/tita-arshia/rag/data/eval/combined_rerank.json`
- Per-suite JSON: `/home/ai/ollm-ocr-2/tita-arshia/rag/data/eval/*_rerank.json`
