FROM runpod/worker-sglang:2.0.2

# Pin transformers that supports Qwen3.5 but doesn't break SGLang's AutoImageProcessor
RUN pip install --no-cache-dir "transformers==5.3.0" "huggingface_hub>=0.30" && \
    pip install --no-cache-dir --force-reinstall "transformers==5.3.0"

# Fix the AutoImageProcessor.register() conflict
RUN python3 -c "
import sglang.srt.configs.utils as u
import inspect
src = inspect.getsource(u.register_image_processor)
if 'exist_ok=True' in src:
    new_src = src.replace('exist_ok=True', '')
    # Monkey-patch not needed at import time, fix the file directly
" 2>/dev/null; \
    sed -i 's/exist_ok=True//' /sgl-workspace/python/sglang/srt/configs/utils.py 2>/dev/null || true; \
    sed -i 's/, exist_ok=True//' /sgl-workspace/python/sglang/srt/configs/utils.py 2>/dev/null || true

ENV SGLANG_DISABLE_CUDNN_CHECK=1
