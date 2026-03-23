# Kira Surge 1 — RunPod Serverless Worker
# =========================================
# worker-sglang base + transformers 5.3 + Python-based patches.
# Keeps native entrypoint for proper RunPod handler registration.

FROM runpod/worker-sglang:2.0.2

# Upgrade transformers for Qwen3.5 support
RUN pip install --no-cache-dir "transformers>=5.3.0" "huggingface_hub>=0.30"

# Apply all SGLang patches via Python script (more reliable than sed)
COPY patch_sglang.py /tmp/patch_sglang.py
RUN python3 /tmp/patch_sglang.py && rm /tmp/patch_sglang.py

ENV SGLANG_DISABLE_CUDNN_CHECK=1

# Keep native worker-sglang entrypoint
