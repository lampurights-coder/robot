"""Expose weather MCP over the LAN for remote LLM clients.

On THIS machine (where weather_mcp lives), run:
  python weather_mcp/run_http.py

Default: http://0.0.0.0:8765/sse
Remote client (e.g. 192.168.7.178) connects to:
  http://<THIS_MACHINE_LAN_IP>:8765/sse
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env", override=True)

# Bind on all interfaces so LAN clients can reach the server
os.environ.setdefault("FASTMCP_HOST", "0.0.0.0")
os.environ.setdefault("FASTMCP_PORT", "8765")

from weather_mcp.server import mcp  # noqa: E402

if __name__ == "__main__":
    host = os.environ["FASTMCP_HOST"]
    port = os.environ["FASTMCP_PORT"]
    print(f"Weather MCP listening on http://{host}:{port}")
    print(f"  SSE endpoint:              http://{host}:{port}/sse")
    print(f"  Streamable HTTP endpoint:  http://{host}:{port}/mcp")
    print("Point your remote client at THIS machine's LAN IP, e.g.")
    print(f"  http://192.168.7.x:{port}/sse")
    # SSE is widely supported by MCP clients / local LLM UIs
    mcp.run(transport="sse")
