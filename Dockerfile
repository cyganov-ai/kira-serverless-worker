FROM runpod/worker-sglang:2.0.2

# Upgrade transformers for Qwen3.5 support
RUN pip install --no-cache-dir "transformers>=5.3.0" "huggingface_hub>=0.30"

# Fix AutoImageProcessor.register() conflict between transformers 5.3 and SGLang
# The issue: transformers 5.3 changed register() signature, removing exist_ok param
RUN find / -path "*/sglang/srt/configs/utils.py" -exec sed -i 's/, exist_ok=True//g' {} \; 2>/dev/null; \
    find / -path "*/sglang/srt/configs/utils.py" -exec sed -i 's/exist_ok=True//g' {} \; 2>/dev/null; \
    echo "Patched"

ENV SGLANG_DISABLE_CUDNN_CHECK=1
