# Kira Surge 1 — RunPod Serverless Worker
# =========================================
# Uses worker-sglang base image AS-IS (keeps its native entrypoint)
# Only upgrades transformers and patches SGLang for Qwen3.5.
# The base image handles SGLang startup + RunPod handler registration.

FROM runpod/worker-sglang:2.0.2

# Upgrade transformers for Qwen3.5 support
RUN pip install --no-cache-dir "transformers>=5.3.0" "huggingface_hub>=0.30"

# Fix AutoImageProcessor.register() conflict
RUN find / -path "*/sglang/srt/configs/utils.py" -exec sed -i 's/, exist_ok=True//g' {} \; 2>/dev/null; \
    echo "Patched AutoImageProcessor"

# Fix layers_block_type property (read-only in transformers 5.3)
RUN QWEN_FILE=$(find / -path "*/sglang/srt/models/qwen3_5.py" 2>/dev/null | head -1) && \
    if [ -n "$QWEN_FILE" ]; then \
        sed -i "s/config\.layers_block_type = \[\]/object.__setattr__(config, 'layers_block_type', [])/" "$QWEN_FILE" && \
        echo "Patched layers_block_type"; \
    fi

# Fix empty rope_scaling {}
RUN ROPE_FILE=$(find / -path "*/sglang/srt/layers/rotary_embedding.py" 2>/dev/null | head -1) && \
    if [ -n "$ROPE_FILE" ]; then \
        sed -i 's/elif rope_scaling is None:/elif rope_scaling is None or rope_scaling == {}:/' "$ROPE_FILE" && \
        echo "Patched rope_scaling"; \
    fi

ENV SGLANG_DISABLE_CUDNN_CHECK=1

# DO NOT override CMD — let the base image's native entrypoint handle everything
