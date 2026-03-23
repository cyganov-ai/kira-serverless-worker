# Kira Surge 1 — RunPod Serverless Worker (Clean SGLang Build)
# =============================================================
# Clean install of SGLang + transformers 5.3 for Qwen3.5 support.
# Uses runpod handler that wraps SGLang's OpenAI-compatible API.

FROM runpod/base:0.6.2-cuda12.2.0

# Install SGLang with all dependencies
RUN pip install --no-cache-dir \
    "sglang[all]>=0.4.5" \
    "transformers>=5.3.0" \
    "huggingface_hub>=0.30" \
    "runpod>=1.7.0" \
    flashinfer

# Patch SGLang for Qwen3.5 compatibility:
# 1. Fix layers_block_type property (read-only in transformers 5.3)
# 2. Fix empty rope_scaling {} handling
# 3. Fix AutoImageProcessor.register() signature change
RUN QWEN_FILE=$(find / -path "*/sglang/srt/models/qwen3_5.py" 2>/dev/null | head -1) && \
    if [ -n "$QWEN_FILE" ]; then \
        sed -i "s/config\.layers_block_type = \[\]/object.__setattr__(config, 'layers_block_type', [])/" "$QWEN_FILE" && \
        echo "Patched layers_block_type in $QWEN_FILE"; \
    fi

RUN ROPE_FILE=$(find / -path "*/sglang/srt/layers/rotary_embedding.py" 2>/dev/null | head -1) && \
    if [ -n "$ROPE_FILE" ]; then \
        sed -i 's/elif rope_scaling is None:/elif rope_scaling is None or rope_scaling == {}:/' "$ROPE_FILE" && \
        echo "Patched rope_scaling in $ROPE_FILE"; \
    fi

RUN find / -path "*/sglang/srt/configs/utils.py" -exec sed -i 's/, exist_ok=True//g' {} \; 2>/dev/null; \
    find / -path "*/sglang/srt/configs/utils.py" -exec sed -i 's/exist_ok=True//g' {} \; 2>/dev/null; \
    echo "Patched AutoImageProcessor"

# Copy RunPod handler
COPY handler.py /app/handler.py

ENV SGLANG_DISABLE_CUDNN_CHECK=1

CMD ["python", "/app/handler.py"]
