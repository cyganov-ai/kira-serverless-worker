"""RunPod Serverless handler for Kira Surge 1 via SGLang.

Starts SGLang server at container startup, then proxies RunPod
job requests to SGLang's OpenAI-compatible API.
"""

import logging
import os
import subprocess
import time

import requests
import runpod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kira-handler")

# Configuration from environment
MODEL_NAME = os.environ.get("MODEL_NAME", "cyganovroman/kira-surge-1")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
CONTEXT_LENGTH = os.environ.get("CONTEXT_LENGTH", "8192")
DTYPE = os.environ.get("DTYPE", "bfloat16")
SGLANG_PORT = 8080  # Internal port for SGLang

# Start SGLang server in background
def start_sglang():
    """Launch SGLang server and wait until it's ready."""
    logger.info("Starting SGLang server for model: %s", MODEL_NAME)

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
        "--port", str(SGLANG_PORT),
        "--host", "0.0.0.0",
    ]

    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Wait for SGLang to be ready (poll health endpoint)
    max_wait = 900  # 15 minutes max
    start = time.time()
    while time.time() - start < max_wait:
        try:
            resp = requests.get(f"http://localhost:{SGLANG_PORT}/health", timeout=5)
            if resp.status_code == 200:
                logger.info("SGLang server is READY! (took %.0fs)", time.time() - start)
                return process
        except requests.ConnectionError:
            pass

        # Check if process died
        if process.poll() is not None:
            output = process.stdout.read().decode() if process.stdout else ""
            logger.error("SGLang process died! Output: %s", output[-2000:])
            raise RuntimeError(f"SGLang failed to start: {output[-500:]}")

        time.sleep(5)

    raise RuntimeError("SGLang server failed to become ready within 15 minutes")


# Start SGLang at container init
logger.info("=" * 60)
logger.info("  Kira Surge 1 — RunPod Serverless Handler")
logger.info("=" * 60)
sglang_process = start_sglang()


def handler(job):
    """Process inference request by proxying to SGLang."""
    try:
        job_input = job["input"]

        # Support both direct messages and openai_route format
        if "openai_route" in job_input:
            route = job_input["openai_route"]
            payload = job_input.get("openai_input", {})
        elif "messages" in job_input:
            route = "/v1/chat/completions"
            payload = {
                "model": MODEL_NAME,
                "messages": job_input["messages"],
                "max_tokens": job_input.get("max_tokens", 2048),
                "temperature": job_input.get("temperature", 0.7),
                "top_p": job_input.get("top_p", 0.95),
            }
        else:
            return {"error": "Invalid input. Provide 'messages' or 'openai_route' + 'openai_input'"}

        # Forward to SGLang
        resp = requests.post(
            f"http://localhost:{SGLANG_PORT}{route}",
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()

    except Exception as e:
        logger.error("Handler error: %s", str(e))
        return {"error": str(e)}


logger.info("Starting RunPod handler...")
runpod.serverless.start({"handler": handler})
