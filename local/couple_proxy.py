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

QWY = "http://127.0.0.1:8080/v1/chat/completions"   # planner
ORN = "http://127.0.0.1:8081/v1/chat/completions"   # doer (tools)

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
            "max_tokens": 400, "temperature": 0.3}, timeout=600)
        PLANS[fp] = strip_think(r.json()["choices"][0]["message"]["content"])
        print(f"[plan] {PLANS[fp][:200]}", flush=True)
    fwd = dict(body)
    fwd["messages"] = [{"role": "system",
                        "content": "Follow this plan from your planning partner:\n" + PLANS[fp]}] + msgs
    fwd.pop("model", None)
    out = requests.post(ORN, json=fwd, timeout=1200).json()
    try:
        m = out["choices"][0]["message"]
        if m.get("content"):
            m["content"] = strip_think(m["content"])
    except Exception:
        pass
    out["model"] = "couple"
    return out


if __name__ == "__main__":
    print("couple proxy on http://127.0.0.1:9000/v1  (model: couple)")
    uvicorn.run(app, host="127.0.0.1", port=9000, log_level="warning")
