"""Patch SGLang for Qwen3.5 + transformers 5.3 compatibility."""
import glob

def find_files(pattern):
    """Find ALL files matching glob pattern."""
    return glob.glob(pattern, recursive=True)

# 1. Patch layers_block_type property issue
for qwen_file in find_files("/**/sglang/srt/models/qwen3_5.py"):
    with open(qwen_file) as f:
        code = f.read()
    code = code.replace(
        "config.layers_block_type = []",
        "object.__setattr__(config, 'layers_block_type', [])"
    )
    with open(qwen_file, 'w') as f:
        f.write(code)
    print(f"Patched layers_block_type in {qwen_file}")

# 2. Patch rope_scaling empty dict
for rope_file in find_files("/**/sglang/srt/layers/rotary_embedding.py"):
    with open(rope_file) as f:
        code = f.read()
    code = code.replace(
        "elif rope_scaling is None:",
        "elif rope_scaling is None or rope_scaling == {}:"
    )
    with open(rope_file, 'w') as f:
        f.write(code)
    print(f"Patched rope_scaling in {rope_file}")

# 3. Replace utils.py entirely to fix AutoImageProcessor.register() conflict
UTILS_REPLACEMENT = '''from typing import Type

from transformers import (
    AutoImageProcessor,
    AutoProcessor,
    BaseImageProcessor,
    PretrainedConfig,
    ProcessorMixin,
)


def register_image_processor(
    config: Type[PretrainedConfig], image_processor: Type[BaseImageProcessor]
):
    """Register customized hf image processor."""
    try:
        AutoImageProcessor.register(config, image_processor)
    except TypeError:
        try:
            AutoImageProcessor.register(config, slow_image_processor_class=image_processor)
        except TypeError:
            print(f"Warning: Could not register image processor for {config}")


def register_processor(config: Type[PretrainedConfig], processor: Type[ProcessorMixin]):
    """Register customized hf processor."""
    try:
        AutoProcessor.register(config, processor)
    except TypeError:
        print(f"Warning: Could not register processor for {config}")
'''

for utils_file in find_files("/**/sglang/srt/configs/utils.py"):
    with open(utils_file, 'w') as f:
        f.write(UTILS_REPLACEMENT)
    print(f"Replaced utils.py at {utils_file}")

# Also patch janus_pro.py which calls register with wrong args
for janus_file in find_files("/**/sglang/srt/configs/janus_pro.py"):
    with open(janus_file) as f:
        code = f.read()
    # Remove any exist_ok args from direct register calls
    code = code.replace(", exist_ok=True", "")
    code = code.replace("exist_ok=True", "")
    with open(janus_file, 'w') as f:
        f.write(code)
    print(f"Patched janus_pro.py at {janus_file}")

print("All patches applied!")
