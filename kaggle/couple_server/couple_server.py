# couple-server | Kaggle T4x2 -> ONE OpenAI-compatible endpoint for OpenCode/any agent.
# Qwythos (GPU0) silently PLANS each new conversation; Ornith (GPU1) answers + tool-calls.
# Public URL via cloudflared quick tunnel (printed in log). Point OpenCode at <url>/v1.
import os, subprocess, threading, re, time, json

def sh(c, **k): print("$", c, flush=True); return subprocess.run(c, shell=True, **k)

print("=== install ===", flush=True)
sh("pip -q install huggingface_hub fastapi uvicorn")
sh("pip -q install 'llama-cpp-python[server]' "
   "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124")
sh("wget -q -O /tmp/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/"
   "download/cloudflared-linux-amd64 && chmod +x /tmp/cloudflared")

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

QWY = Llama(model_path=hf_hub_download("empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF",
            "Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf"),
            n_gpu_layers=-1, n_ctx=32768, main_gpu=0, flash_attn=True, verbose=False)
ORN = Llama(model_path=hf_hub_download("deepreinforce-ai/Ornith-1.0-9B-GGUF",
            "ornith-1.0-9b-Q4_K_M.gguf"),
            n_gpu_layers=-1, n_ctx=32768, main_gpu=1, flash_attn=True, verbose=False)
print("=== both models loaded (Qwythos GPU0 planner | Ornith GPU1 doer) ===", flush=True)

def strip_think(t):
    return re.sub(r"<think>.*?</think>", "", t or "", flags=re.DOTALL).strip()

from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()
PLANS = {}  # conversation fingerprint -> plan (plan once per convo, not per turn)

def fingerprint(msgs):
    first_user = next((m for m in msgs if m.get("role") == "user"), {})
    return hash(str(first_user.get("content"))[:2000])

@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": "couple", "object": "model", "owned_by": "ornith-couple"}]}

@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    msgs = body.get("messages", [])
    fp = fingerprint(msgs)
    if fp not in PLANS:  # first turn of a conversation -> Qwythos plans silently
        task = str(next((m.get("content") for m in msgs if m.get("role") == "user"), ""))[:6000]
        p = QWY.create_chat_completion(messages=[
            {"role": "system", "content": "You are a senior planning engineer. Write a SHORT "
             "numbered plan (3-6 steps) for the task. Concrete files/steps. No code."},
            {"role": "user", "content": task}], max_tokens=400, temperature=0.3)
        PLANS[fp] = strip_think(p["choices"][0]["message"]["content"])
        print(f"[plan] {PLANS[fp][:200]}", flush=True)
    # inject plan as extra system msg; Ornith answers (tools pass through untouched)
    aug = [{"role": "system", "content": "Follow this plan from your planning partner:\n" + PLANS[fp]}] + msgs
    kw = dict(messages=aug, max_tokens=min(int(body.get("max_tokens", 4096) or 4096), 8192),
              temperature=body.get("temperature", 0.3))
    if body.get("tools"):
        kw["tools"] = body["tools"]
        if body.get("tool_choice"): kw["tool_choice"] = body["tool_choice"]
    out = ORN.create_chat_completion(**kw)
    m = out["choices"][0]["message"]
    if m.get("content"): m["content"] = strip_think(m["content"])
    out["model"] = "couple"
    return out

def tunnel():
    time.sleep(3)
    p = subprocess.Popen(["/tmp/cloudflared", "tunnel", "--url", "http://127.0.0.1:9000"],
                         stderr=subprocess.PIPE, text=True)
    for line in p.stderr:
        mm = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
        if mm:
            print(f"\n*** PUBLIC ENDPOINT: {mm.group(0)}/v1  (model id: couple) ***\n", flush=True)
    p.wait()

threading.Thread(target=tunnel, daemon=True).start()
print("=== serving on :9000, tunnel starting... stays up until kernel timeout (~9-12h) ===", flush=True)
uvicorn.run(app, host="0.0.0.0", port=9000, log_level="warning")
