"""RunPod Serverless handler for Kira Surge 1.

Starts SGLang with --disable-cuda-graph, proxies requests.
Writes debug log to /tmp/kira_debug.log for troubleshooting.
"""
import logging
import os
import subprocess
import sys
import time
import traceback

# Setup logging to both stdout and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("kira")

# Log everything to stdout so RunPod captures it
print("=" * 60, flush=True)
print("  KIRA HANDLER STARTING", flush=True)
print("=" * 60, flush=True)

MODEL_NAME = os.environ.get("MODEL_NAME", "cyganovroman/kira-surge-1")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
CONTEXT_LENGTH = os.environ.get("CONTEXT_LENGTH", "8192")
DTYPE = os.environ.get("DTYPE", "bfloat16")
PORT = 8080

print(f"MODEL_NAME={MODEL_NAME}", flush=True)
print(f"CONTEXT_LENGTH={CONTEXT_LENGTH}", flush=True)
print(f"DTYPE={DTYPE}", flush=True)
print(f"HF_TOKEN={'set' if HF_TOKEN else 'NOT SET'}", flush=True)

# Test imports first
try:
    print("Testing imports...", flush=True)
    import sglang
    print(f"  sglang version: {sglang.__version__}", flush=True)
    import transformers
    print(f"  transformers version: {transformers.__version__}", flush=True)
    import runpod
    print(f"  runpod imported OK", flush=True)
    print("All imports OK!", flush=True)
except Exception as e:
    print(f"IMPORT ERROR: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

import requests as http_requests

def start_sglang():
    print("Starting SGLang server...", flush=True)
    env = os.environ.copy()
    env["HF_TOKEN"] = HF_TOKEN
    env["SGLANG_DISABLE_CUDNN_CHECK"] = "1"

    cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", MODEL_NAME,
        "--tp", "1",
        "--trust-remote-code",
        "--dtype", DTYPE,
        "--context-length", CONTEXT_LENGTH,
        "--port", str(PORT),
        "--host", "0.0.0.0",
        "--disable-cuda-graph",
        "--disable-radix-cache",
    ]
    print(f"CMD: {' '.join(cmd)}", flush=True)

    proc = subprocess.Popen(
        cmd, env=env,
        stdout=sys.stdout,  # Forward SGLang output to stdout
        stderr=sys.stdout,
    )

    for i in range(180):  # 15 min max
        try:
            r = http_requests.get(f"http://localhost:{PORT}/health", timeout=3)
            if r.status_code == 200:
                print(f"SGLang READY after {i*5}s!", flush=True)
                return proc
        except:
            pass
        if proc.poll() is not None:
            print(f"SGLang DIED with code {proc.returncode}", flush=True)
            sys.exit(1)
        time.sleep(5)
        if i % 12 == 0:
            print(f"  Waiting for SGLang... {i*5}s", flush=True)

    print("SGLang TIMEOUT after 15 min", flush=True)
    sys.exit(1)

try:
    sglang_proc = start_sglang()
except Exception as e:
    print(f"FATAL: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

def handler(job):
    try:
        inp = job["input"]
        if "openai_route" in inp:
            route = inp["openai_route"]
            payload = inp.get("openai_input", {})
        elif "messages" in inp:
            route = "/v1/chat/completions"
            payload = {
                "model": MODEL_NAME,
                "messages": inp["messages"],
                "max_tokens": inp.get("max_tokens", 2048),
                "temperature": inp.get("temperature", 0.7),
            }
        else:
            return {"error": "Provide 'messages' or 'openai_route'+'openai_input'"}

        r = http_requests.post(f"http://localhost:{PORT}{route}", json=payload, timeout=300)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"HANDLER ERROR: {e}", flush=True)
        return {"error": str(e)}

print("Registering with RunPod...", flush=True)
runpod.serverless.start({"handler": handler})
