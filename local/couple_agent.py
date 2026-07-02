"""Agentic couple: Qwythos (planner, :8080) x Ornith (doer, :8081) with REAL tool calls.

Tools the DOER can call: read_file, write_file, list_dir, run_cmd (guarded).
Loop: planner plans -> doer acts with tools until it says DONE -> planner reviews ->
retry with feedback -> SWAP hats on 2nd failure -> bail on 3rd. All via the two
llama-server OpenAI-compatible endpoints (started by launch.py).
"""
from __future__ import annotations
import json, pathlib, re, subprocess

import requests

ENDPOINTS = {"qwythos": "http://127.0.0.1:8080/v1/chat/completions",
             "ornith":  "http://127.0.0.1:8081/v1/chat/completions"}

WORKDIR = pathlib.Path.cwd()          # tools are sandboxed to CWD
MAX_TOOL_TURNS = 12                    # per doer attempt
BLOCKED = ("rm -rf", "format", "del /", "shutdown", "reg ", "mkfs")

TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a text file",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                    "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Write/overwrite a text file",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"},
                    "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "list_dir", "description": "List files in a directory",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                    "required": ["path"]}}},
    {"type": "function", "function": {"name": "run_cmd", "description":
     "Run a shell command (tests, python, git). 60s timeout.",
     "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"]}}},
]


def _safe(path: str) -> pathlib.Path:
    p = (WORKDIR / path).resolve()
    if WORKDIR.resolve() not in p.parents and p != WORKDIR.resolve():
        raise ValueError(f"path escapes workdir: {path}")
    return p


def exec_tool(name: str, args: dict) -> str:
    try:
        if name == "read_file":
            return _safe(args["path"]).read_text(encoding="utf-8", errors="replace")[:8000]
        if name == "write_file":
            p = _safe(args["path"]); p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(args["content"], encoding="utf-8"); return f"wrote {args['path']}"
        if name == "list_dir":
            return "\n".join(x.name + ("/" if x.is_dir() else "") for x in _safe(args["path"]).iterdir())[:4000]
        if name == "run_cmd":
            c = args["cmd"]
            if any(b in c.lower() for b in BLOCKED):
                return "BLOCKED: dangerous command"
            r = subprocess.run(c, shell=True, cwd=WORKDIR, capture_output=True, text=True, timeout=60)
            return (r.stdout + r.stderr)[-4000:] or f"(exit {r.returncode}, no output)"
        return f"unknown tool {name}"
    except Exception as e:
        return f"TOOL ERROR: {e}"


def strip_think(t: str) -> str:
    return re.sub(r"<think>.*?</think>", "", t or "", flags=re.DOTALL).strip()


def chat(model: str, messages: list, tools: bool = False, max_tokens: int = 1200):
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0.3}
    if tools:
        body["tools"] = TOOLS
    r = requests.post(ENDPOINTS[model], json=body, timeout=600)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]


PLANNER_SYS = ("You are the PLANNER of a two-model coding couple working in a real directory. "
               "Write a SHORT numbered plan (2-5 steps) for the task. Name concrete files/commands. No code.")
DOER_SYS = ("You are the DOER of a two-model coding couple. You have tools: read_file, write_file, "
            "list_dir, run_cmd. Execute the plan step by step USING TOOLS. Verify your work with "
            "run_cmd when possible. When the task is fully done and verified, reply with a line "
            "starting 'DONE:' and a summary. If stuck, reply 'STUCK:' and why.")
REVIEW_SYS = ("You are the PLANNER reviewing the DOER's work. Given the task, plan, and the doer's "
              "final message + tool log, reply exactly 'OK' if the task is genuinely complete, "
              "otherwise one line of what is wrong/missing.")


def doer_run(doer: str, task: str, plan: str, hint: str | None, log):
    msgs = [{"role": "system", "content": DOER_SYS},
            {"role": "user", "content": f"Task:\n{task}\n\nPlan:\n{plan}"
             + (f"\n\nPrevious attempt failed: {hint}. Fix it." if hint else "")}]
    tool_log = []
    for _ in range(MAX_TOOL_TURNS):
        m = chat(doer, msgs, tools=True)
        calls = m.get("tool_calls")
        if calls:
            msgs.append(m)
            for tc in calls:
                fn = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"].get("arguments") or "{}")
                except Exception:
                    args = {}
                out = exec_tool(fn, args)
                tool_log.append((fn, args, out[:300]))
                log(f"🔧 {doer} → {fn}({json.dumps(args)[:120]}) → {out[:160]}")
                msgs.append({"role": "tool", "tool_call_id": tc.get("id", "t"), "content": out})
            continue
        text = strip_think(m.get("content", ""))
        log(f"💬 {doer}: {text[:400]}")
        return text, tool_log
    return "STUCK: tool-turn budget exhausted", tool_log


def run_couple(task: str, log=print):
    planner, doer = "qwythos", "ornith"
    fails, hint = 0, None
    plan = strip_think(chat(planner, [{"role": "system", "content": PLANNER_SYS},
                                      {"role": "user", "content": task}])["content"])
    log(f"📋 PLAN ({planner}):\n{plan}")
    while True:
        final, tool_log = doer_run(doer, task, plan, hint, log)
        review = strip_think(chat(planner, [
            {"role": "system", "content": REVIEW_SYS},
            {"role": "user", "content": f"Task:\n{task}\n\nPlan:\n{plan}\n\nDoer said:\n{final}\n\n"
             f"Tool log tail:\n{json.dumps(tool_log[-6:], default=str)[:2000]}"}])["content"])
        log(f"🔍 REVIEW ({planner}): {review[:300]}")
        if final.startswith("DONE:") and review.strip().upper().startswith("OK"):
            log("✅ COUPLE: task complete (doer done + planner approved)")
            return True
        fails += 1
        hint = review if not review.upper().startswith("OK") else final
        if fails == 1:
            log("↻ retry #1 (same hats, with reviewer feedback)"); continue
        if fails == 2:
            planner, doer = doer, planner
            log(f"🔄 SWAP! planner={planner} doer={doer}")
            plan = strip_think(chat(planner, [{"role": "system", "content": PLANNER_SYS},
                                              {"role": "user", "content": task}])["content"])
            log(f"📋 NEW PLAN ({planner}):\n{plan}"); continue
        log("❌ bailed after 3 failed attempts")
        return False


if __name__ == "__main__":
    import sys
    run_couple(" ".join(sys.argv[1:]) or "Create hello.py that prints 'couple works', run it, confirm output.")
