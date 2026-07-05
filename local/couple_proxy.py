"""Local couple-as-one-model proxy for OpenCode (or any OpenAI-compatible client).

Sits on :9000. Qwythos (:8080) silently PLANS once per conversation; Ornith (:8081)
answers every turn with the plan injected — tool calls pass through untouched, so
OpenCode's file/terminal tools work. Start AFTER launch.py has both servers up.

Run:  pip install fastapi uvicorn requests
      python couple_proxy.py
OpenCode baseURL: http://127.0.0.1:9000/v1   model: couple
"""
from __future__ import annotations
import re

import requests
import uvicorn
from fastapi import FastAPI, Request

import os
# Split-brain across machines: set env vars to point at another laptop, e.g.
#   set ORN_URL=http://100.101.5.23:8081   (friend's Tailscale IP running Ornith)
QWY = os.environ.get("QWY_URL", "http://127.0.0.1:8080") + "/v1/chat/completions"  # planner
ORN = os.environ.get("ORN_URL", "http://127.0.0.1:8081") + "/v1/chat/completions"  # doer (tools)

app = FastAPI()
PLANS: dict[int, str] = {}


def strip_think(t: str) -> str:
    return re.sub(r"<think>.*?</think>", "", t or "", flags=re.DOTALL).strip()


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": "couple", "object": "model", "owned_by": "local"}]}


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    msgs = body.get("messages", [])
    first_user = str(next((m.get("content") for m in msgs if m.get("role") == "user"), ""))
    fp = hash(first_user[:2000])
    if fp not in PLANS:
        r = requests.post(QWY, json={"messages": [
            {"role": "system", "content": "You are a senior planning engineer. Write a SHORT "
             "numbered plan (3-6 steps) for the task. Concrete files/steps. No code."},
            {"role": "user", "content": first_user[:6000]}],
            "max_tokens": 600, "temperature": 0.3,
            "chat_template_kwargs": {"enable_thinking": False}}, timeout=600)
        pm = r.json()["choices"][0]["message"]
        # thinking-mode models put text in reasoning_content and leave content empty
        PLANS[fp] = strip_think(pm.get("content") or pm.get("reasoning_content") or "")
        print(f"[plan] {PLANS[fp][:200]}", flush=True)
    fwd = dict(body)
    fwd["messages"] = [{"role": "system",
                        "content": "Follow this plan from your planning partner:\n" + PLANS[fp]}] + msgs
    fwd.pop("model", None)
    fwd["chat_template_kwargs"] = {"enable_thinking": False}  # doer: answer, don't ruminate
    out = requests.post(ORN, json=fwd, timeout=1200).json()
    try:
        m = out["choices"][0]["message"]
        if not m.get("content") and not m.get("tool_calls"):
            m["content"] = m.get("reasoning_content") or ""   # thinking-mode fallback
        if m.get("content"):
            m["content"] = strip_think(m["content"])
    except Exception:
        pass
    out["model"] = "couple"
    return out


if __name__ == "__main__":
    print("couple proxy on http://127.0.0.1:9000/v1  (model: couple)")
    uvicorn.run(app, host="127.0.0.1", port=9000, log_level="warning")
