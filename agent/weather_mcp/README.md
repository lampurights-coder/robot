# Weather MCP

Weather-only MCP: search the web once, **cache the best page URL**, then reuse it.

- Returns **Pydantic** `WeatherResult` (Farsi + °C)
- No email
- Cache file: `weather_mcp/.weather_url_cache.json`

## Tools

| Tool | Purpose |
|------|---------|
| `get_weather_tool(city, force_refresh=False, force_search=False)` | Get weather |
| `list_weather_cache()` | List cached cities |
| `clear_weather_cache(city=None)` | Clear one/all cities |

### Cache behavior

1. First call for `Tehran` → LangSearch → pick URL → fetch → structure → **save URL + result**
2. Later same day → return cached **result** (`cache_hit=true`, no search, no fetch)
3. `force_refresh=true` → fetch cached URL again (no search)
4. `force_search=true` → new LangSearch and update URL

## Run / test without Cursor

```bash
cd ~/Documents/agent
python -c "from weather_mcp.service import get_weather; print(get_weather('Tehran').model_dump_json(indent=2))"
```

## Cursor (local stdio)

Project config: `.cursor/mcp.json`  
Reload MCP in Cursor Settings → MCP.

```bash
python weather_mcp/server.py
```

## Remote client / local LLM on another PC (HTTP SSE)

stdio only works on the same machine. For a remote LLM (e.g. on `192.168.7.178`), run the **HTTP SSE** server on the machine that has this repo:

```bash
cd ~/Documents/agent
python weather_mcp/run_http.py
# listens on 0.0.0.0:8765
```

Find this PC’s LAN IP:

```bash
hostname -I
# example: 192.168.7.10
```

On the **remote** machine / LLM client, add MCP config pointing at this host:

```json
{
  "mcpServers": {
    "weather": {
      "url": "http://192.168.7.10:8765/sse"
    }
  }
}
```

Replace `192.168.7.10` with the IP of the PC running `run_http.py`.

Optional env overrides:

```bash
FASTMCP_HOST=0.0.0.0 FASTMCP_PORT=8765 python weather_mcp/run_http.py
```

Firewall (if needed):

```bash
sudo ufw allow 8765/tcp
```

### Notes

- Keep `.env` (LangSearch + GapGPT keys) on the **server** machine; the remote client only needs the URL.
- This SSE endpoint has **no auth** — use only on trusted LAN.
- If the LLM UI wants streamable HTTP instead of SSE, try: `http://<IP>:8765/mcp`
