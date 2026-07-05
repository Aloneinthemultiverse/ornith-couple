# friend_qwythos.ps1 — one-shot setup for the couple's PLANNER on friend's laptop (RTX 4050 6GB)
# Downloads Qwythos v3 non-MTP Q4_K_M (fixed tokenizer + OpenCode tool-calling) and serves it.
# Run in PowerShell:  powershell -ExecutionPolicy Bypass -File friend_qwythos.ps1

$MODEL_DIR = "C:\models"
$MODEL = "Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf"   # v3 NON-MTP (the MTP file is broken on new llama.cpp)
$URL = "https://huggingface.co/empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF/resolve/main/$MODEL"

New-Item -ItemType Directory -Force $MODEL_DIR | Out-Null
$dest = Join-Path $MODEL_DIR $MODEL

if (-not (Test-Path $dest) -or (Get-Item $dest).Length -lt 5GB) {
    Write-Host ">> downloading $MODEL (~5.5GB, resumes if interrupted)..."
    curl.exe -L -C - --retry 10 --retry-delay 5 -o $dest $URL
    if ($LASTEXITCODE -ne 0) { Write-Host "!! download failed - rerun this script to resume"; exit 1 }
}
Write-Host ">> model ready: $dest ($([math]::Round((Get-Item $dest).Length/1GB,2)) GB)"

# find llama-server.exe (extracted cuda build)
$server = @("C:\llama\llama-server.exe") + (Get-ChildItem C:\ -Recurse -Depth 3 -Filter llama-server.exe -ErrorAction SilentlyContinue | ForEach-Object FullName) | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $server) { Write-Host "!! llama-server.exe not found - extract the cuda-12.4 zips to C:\llama"; exit 1 }
Write-Host ">> using $server"

# 4050 6GB: partial GPU offload; KV cache spills to system RAM; 16k context
# If it crashes with CUDA out-of-memory: change -ngl 28 to -ngl 22 and rerun.
& $server -m $dest --host 0.0.0.0 --port 8080 -ngl 28 --ctx-size 16384 --jinja
