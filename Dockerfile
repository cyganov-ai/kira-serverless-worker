# Kira Surge 1 — RunPod Serverless Worker (SGLang)
# ==================================================
# Based on runpod/worker-sglang with patches for Qwen3.5 support:
# - transformers 5.3+ for Qwen3_5ForConditionalGeneration
# - SGLang patches for layers_block_type property, rope_scaling, AutoImageProcessor
#
# Environment variables (set in RunPod endpoint config):
#   MODEL_NAME: HuggingFace model repo (e.g., cyganovroman/kira-surge-1)
#   HF_TOKEN: HuggingFace API token for private models
#   CONTEXT_LENGTH: Max context length (default: 4096)
#   DTYPE: Model dtype (default: bfloat16)

FROM runpod/worker-sglang:2.0.2

# 1. Upgrade transformers for Qwen3.5 support
RUN pip install --no-cache-dir "transformers>=5.3.0" "huggingface_hub>=0.30"

# 2. Fix AutoImageProcessor.register() conflict (transformers 5.3 changed signature)
RUN find / -path "*/sglang/srt/configs/utils.py" -exec sed -i 's/, exist_ok=True//g' {} \; 2>/dev/null; \
    find / -path "*/sglang/srt/configs/utils.py" -exec sed -i 's/exist_ok=True//g' {} \; 2>/dev/null; \
    echo "Patched AutoImageProcessor"

# 3. Fix layers_block_type property (read-only in transformers 5.3 Qwen3_5TextConfig)
#    SGLang tries: config.layers_block_type = []  which fails because it's a @property
#    Fix: use object.__setattr__ to bypass the property
RUN QWEN_FILE=$(find / -path "*/sglang/srt/models/qwen3_5.py" 2>/dev/null | head -1) && \
    if [ -n "$QWEN_FILE" ]; then \
        python3 -c "
import re
with open('$QWEN_FILE') as f:
    code = f.read()

# Replace the block that tries to set config.layers_block_type
# Find the pattern where it builds layers_block_type
old_pattern = r'if not hasattr\(config.*?layers_block_type.*?\n.*?config\.layers_block_type = \[\]'
replacement = '''# Patched: derive layers_block_type for text-only Qwen3.5 configs
        try:
            _lbt = config.layers_block_type
        except (AttributeError, TypeError):
            _lbt = None
        if not _lbt:
            object.__setattr__(config, '_layers_block_type_override', [])
            # Monkey-patch the property to return our list
            try:
                config.__class__.layers_block_type = property(
                    lambda self: getattr(self, '_layers_block_type_override', None)
                )
            except (TypeError, AttributeError):
                pass
            _lbt_ref = getattr(config, '_layers_block_type_override', [])'''

# Simpler approach: just replace the exact assignment lines
code = code.replace(
    'config.layers_block_type = []',
    'object.__setattr__(config, \"layers_block_type\", [])'
)
code = code.replace(
    'config.layers_block_type.append',
    'config.__dict__.setdefault(\"layers_block_type\", []); config.__dict__[\"layers_block_type\"].append'
)

with open('$QWEN_FILE', 'w') as f:
    f.write(code)
print('Patched layers_block_type in', '$QWEN_FILE')
"; \
    else echo "qwen3_5.py not found, skipping patch"; fi

# 4. Fix empty rope_scaling {} (SGLang doesn't handle empty dict)
RUN ROPE_FILE=$(find / -path "*/sglang/srt/layers/rotary_embedding.py" 2>/dev/null | head -1) && \
    if [ -n "$ROPE_FILE" ]; then \
        sed -i 's/elif rope_scaling is None:/elif rope_scaling is None or rope_scaling == {}:/' "$ROPE_FILE" && \
        echo "Patched rope_scaling in $ROPE_FILE"; \
    else echo "rotary_embedding.py not found, skipping patch"; fi

# 5. Disable CUDNN check (not needed, avoids startup error)
ENV SGLANG_DISABLE_CUDNN_CHECK=1
