# Local weather tool (no remote, no email)

Uses `/home/ai/ollm-ocr-2/tita-arshia/agent/weather_mcp` in-process.

## Agent routing (`fastapi_faq_app.py`)

| Question type | Path |
|---------------|------|
| Weather / city (e.g. هوای تهران) | `weather_mcp.service.get_weather(city)` |
| Clinic FAQ | Jina → Qdrant → BGE → Qwen (unchanged) |

```bash
curl -s http://127.0.0.1:7860/ask -H 'Content-Type: application/json' \
  -d '{"question":"هوای تهران چطوره؟"}'
```

## Cursor MCP (stdio, local)

See `rag/mcp.json` / `~/.cursor/mcp.json`.
