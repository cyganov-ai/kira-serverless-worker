# Kira Surge 1 — RunPod Serverless Worker
# =========================================
# Custom handler that starts SGLang + wraps with RunPod handler

FROM runpod/worker-sglang:2.0.2

# Upgrade transformers for Qwen3.5 support
RUN pip install --no-cache-dir "transformers>=5.3.0" "huggingface_hub>=0.30" "runpod>=1.7.0"

# Apply SGLang patches
COPY patch_sglang.py /tmp/patch_sglang.py
RUN python3 /tmp/patch_sglang.py && rm /tmp/patch_sglang.py

# Our custom handler
COPY handler.py /app/handler.py

ENV SGLANG_DISABLE_CUDNN_CHECK=1
ENV DISABLE_CUDA_GRAPH=true

CMD ["python", "/app/handler.py"]
# Build trigger: 1774292515
# Debug build: 1774292894
