# Kira Surge 1 — RunPod Serverless Worker
# =========================================
# worker-sglang base + transformers 5.3 + SGLang patches
# Disables CUDA graphs for fast startup on serverless

FROM runpod/worker-sglang:2.0.2

# Upgrade transformers for Qwen3.5 support
RUN pip install --no-cache-dir "transformers>=5.3.0" "huggingface_hub>=0.30"

# Apply SGLang patches via Python
COPY patch_sglang.py /tmp/patch_sglang.py
RUN python3 /tmp/patch_sglang.py && rm /tmp/patch_sglang.py

ENV SGLANG_DISABLE_CUDNN_CHECK=1
# Disable CUDA graphs to prevent long startup timeout on serverless
ENV DISABLE_CUDA_GRAPH=true
