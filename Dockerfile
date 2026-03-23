FROM runpod/worker-sglang:2.0.2

# Upgrade transformers to 5.3+ for Qwen3.5 support
RUN pip install --no-cache-dir "transformers>=5.3.0" "huggingface_hub>=0.30"

ENV SGLANG_DISABLE_CUDNN_CHECK=1
