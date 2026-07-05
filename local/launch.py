"""Local couple runtime — downloads Q4_K_M GGUFs, starts two llama-server processes.

Requires llama.cpp's `llama-server` binary on PATH (you already run llama.cpp Vulkan
on the Arc GPU). One model on GPU, one on CPU by default — two 9B Q4 models (~6GB each)
usually can't share one consumer GPU. Flip --both-gpu if you have 16GB+ VRAM.

Usage:  python launch.py            # start both servers
        python launch.py --ui       # start servers + Gradio UI
"""
from __future__ import annotations
import argparse, subprocess, sys, time, urllib.request

from huggingface_hub import hf_hub_download, list_repo_files

QWY_REPO = "empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF"
ORN_REPO = "deepreinforce-ai/Ornith-1.0-9B-GGUF"
# exact filenames (verified on Kaggle runs) — skips the flaky list_repo_files API call
KNOWN = {QWY_REPO: "Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf",
         ORN_REPO: "ornith-1.0-9b-Q4_K_M.gguf"}
QWY_PORT, ORN_PORT = 8080, 8081


def get_gguf(repo: str, prefer: str = "q4_k_m") -> str:
    last = None
    for attempt in range(5):                      # WinError 10054 = transient reset; retry
        try:
            if repo in KNOWN:
                print(f"  {repo} -> {KNOWN[repo]}")
                return hf_hub_download(repo, KNOWN[repo])
            fs = [f for f in list_repo_files(repo) if f.lower().endswith(".gguf")]
            pick = ([f for f in fs if prefer in f.lower()]
                    or [f for f in fs if "q4" in f.lower()] or fs)[0]
            print(f"  {repo} -> {pick}")
            return hf_hub_download(repo, pick)
        except Exception as e:
            last = e
            print(f"  attempt {attempt+1}/5 failed ({str(e)[:80]}) — retrying in {2**attempt}s")
            time.sleep(2 ** attempt)
    raise SystemExit(f"could not download from {repo}: {last}\n"
                     f"Tip: set HF_ENDPOINT=https://hf-mirror.com and retry if HF is blocked/flaky.")


import os
LLAMA_BIN = os.environ.get(
    "LLAMA_SERVER",
    r"C:\Users\Sujit Narrayan M\vibe-thinker\runtime\bin\llama-server.exe")  # Vulkan build


def start_server(model_path: str, port: int, gpu: bool, ctx: int = 16384) -> subprocess.Popen:
    ngl = "99" if gpu else "0"
    cmd = [LLAMA_BIN, "-m", model_path, "--port", str(port), "--ctx-size", str(ctx),
           "-ngl", ngl, "--jinja"]  # --jinja enables chat template + TOOL CALLS
    print("$", " ".join(cmd))
    return subprocess.Popen(cmd)


def wait_up(port: int, tries: int = 120) -> bool:
    for _ in range(tries):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            return True
        except Exception:
            time.sleep(2)
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ui", action="store_true", help="also launch the Gradio interface")
    ap.add_argument("--both-gpu", action="store_true", help="put both models on GPU (16GB+ VRAM)")
    ap.add_argument("--ctx", type=int, default=16384)
    a = ap.parse_args()

    print("=== download Q4_K_M GGUFs (cached after first run) ===")
    qwy, orn = get_gguf(QWY_REPO), get_gguf(ORN_REPO)

    print("=== start servers (Qwythos:8080 planner | Ornith:8081 doer) ===")
    procs = [start_server(qwy, QWY_PORT, gpu=True, ctx=a.ctx),
             start_server(orn, ORN_PORT, gpu=a.both_gpu, ctx=a.ctx)]
    for port, name in [(QWY_PORT, "qwythos"), (ORN_PORT, "ornith")]:
        print(f"  waiting for {name}:{port} ...", "UP" if wait_up(port) else "FAILED")

    if a.ui:
        import app  # noqa: F401  (starts Gradio, blocks)
        app.main()
    else:
        print("servers running. Ctrl+C to stop. Run `python app.py` for the UI.")
        try:
            procs[0].wait()
        except KeyboardInterrupt:
            for p in procs:
                p.terminate()


if __name__ == "__main__":
    main()
