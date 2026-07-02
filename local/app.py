"""Gradio UI for the local couple — watch plan/act/swap live, or chat with either model.

Run:  pip install gradio requests
      python launch.py          (starts both llama-servers)
      python app.py             (opens http://127.0.0.1:7860)

Tabs:
  1. Couple Agent — give a task, watch PLAN -> tools -> REVIEW -> SWAP stream live.
  2. Direct Chat  — talk to Qwythos or Ornith individually (sanity-check each brain).
"""
from __future__ import annotations
import threading, queue

import gradio as gr
import requests

import couple_agent as ca


def run_couple_stream(task: str):
    q: "queue.Queue[str|None]" = queue.Queue()
    log_lines: list[str] = []

    def worker():
        try:
            ca.run_couple(task, log=lambda m: q.put(str(m)))
        except Exception as e:
            q.put(f"⚠️ ERROR: {e}")
        q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = q.get()
        if item is None:
            break
        log_lines.append(item)
        yield "\n\n".join(log_lines)


def direct_chat(message, history, model):
    msgs = [{"role": "system", "content": "You are a helpful coding assistant."}]
    for u, a in history:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    msgs.append({"role": "user", "content": message})
    try:
        m = ca.chat(model, msgs, tools=False)
        return ca.strip_think(m.get("content", ""))
    except requests.ConnectionError:
        return f"⚠️ {model} server not running — start with `python launch.py`"


def server_status():
    out = []
    for name, url in ca.ENDPOINTS.items():
        base = url.rsplit("/v1", 1)[0]
        try:
            requests.get(base + "/health", timeout=3)
            out.append(f"🟢 {name} UP ({base})")
        except Exception:
            out.append(f"🔴 {name} DOWN ({base})")
    return "  |  ".join(out)


def main():
    with gr.Blocks(title="ornith-couple local") as demo:
        gr.Markdown("# 🧠🤝🛠️ ornith-couple — Qwythos (planner) × Ornith (doer)")
        status = gr.Markdown(server_status())
        gr.Button("refresh status", size="sm").click(server_status, outputs=status)

        with gr.Tab("Couple Agent"):
            task = gr.Textbox(label="Task", placeholder="e.g. Create fib.py with a memoized fibonacci, add a test, run it",
                              lines=3)
            out = gr.Markdown()
            gr.Button("Run couple", variant="primary").click(run_couple_stream, inputs=task, outputs=out)
            gr.Markdown("Legend: 📋 plan · 🔧 tool call · 💬 doer msg · 🔍 planner review · 🔄 role swap · ✅ done")

        with gr.Tab("Direct Chat"):
            model = gr.Radio(["qwythos", "ornith"], value="qwythos", label="Model")
            gr.ChatInterface(fn=lambda m, h, mod: direct_chat(m, h, mod),
                             additional_inputs=[model])

    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)


if __name__ == "__main__":
    main()
