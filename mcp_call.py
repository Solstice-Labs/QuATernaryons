#!/usr/bin/env python3
"""Call the local Scrapling MCP server: `python3 research/mcp_call.py <tool> [json-args]`
Args default to '{}'. Prints result text (or JSON) to stdout."""
import json, sys, urllib.request

URL = "http://127.0.0.1:8765/mcp"
TOKEN = open("/tmp/scrapling-mcp-token").read().strip()
HDRS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}
INIT = {"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {
    "protocolVersion": "2024-11-05", "capabilities": {},
    "clientInfo": {"name": "buffy", "version": "0"}}}

def post(payload, session_id=None, timeout=180):
    h = dict(HDRS)
    if session_id:
        h["Mcp-Session-Id"] = session_id
    req = urllib.request.Request(URL, json.dumps(payload).encode(), h, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        sid = r.headers.get("Mcp-Session-Id")
        body = r.read().decode()
    return sid, body

def parse_sse(body):
    for line in body.splitlines():
        if line.startswith("data: "):
            return json.loads(line[6:])
    return json.loads(body) if body.strip() else None

def main():
    tool = sys.argv[1]
    args = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    sid, _ = post(INIT)
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"name": tool, "arguments": args}}
    _, body = post(payload, sid)
    msg = parse_sse(body)
    if msg is None:
        print("empty response"); return
    if "error" in msg:
        print(json.dumps(msg["error"], indent=2)[:4000]); return
    content = msg["result"].get("content", [])
    for c in content:
        if c.get("type") == "text":
            t = c["text"]
            print(t if len(t) < 60000 else t[:60000] + "\n...[TRUNCATED]")

if __name__ == "__main__":
    main()
