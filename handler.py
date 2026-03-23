"""RunPod Serverless handler for Kira Surge 1.

Starts SGLang with --disable-cuda-graph for fast startup,
then proxies requests via runpod.serverless.start().
"""
import logging
import os
import subprocess
import time

import requests as http_requests
import runpod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kira")

MODEL_NAME = os.environ.get("MODEL_NAME", "cyganovroman/kira-surge-1")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
CONTEXT_LENGTH = os.environ.get("CONTEXT_LENGTH", "8192")
DTYPE = os.environ.get("DTYPE", "bfloat16")
PORT = 8080

def start_sglang():
    logger.info("Starting SGLang for %s...", MODEL_NAME)
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

    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    # Wait up to 15 min for SGLang to be ready
    for i in range(180):
        try:
            r = http_requests.get(f"http://localhost:{PORT}/health", timeout=3)
            if r.status_code == 200:
                logger.info("SGLang READY after %ds", i * 5)
                return proc
        except:
            pass
        if proc.poll() is not None:
            out = proc.stdout.read().decode()[-1000:]
            raise RuntimeError(f"SGLang died: {out}")
        time.sleep(5)

    raise RuntimeError("SGLang timeout after 15 min")

sglang_proc = start_sglang()

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
        logger.error("Error: %s", e)
        return {"error": str(e)}

logger.info("Starting RunPod handler")
runpod.serverless.start({"handler": handler})
