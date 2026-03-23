"""Patch SGLang for Qwen3.5 + transformers 5.3 compatibility.

Run this after pip install to fix all known issues.
"""
import glob
import os
import re

def find_file(pattern):
    """Find a file matching glob pattern."""
    matches = glob.glob(pattern, recursive=True)
    return matches[0] if matches else None

# 1. Patch layers_block_type property issue
qwen_file = find_file("/usr/**/sglang/srt/models/qwen3_5.py")
if qwen_file:
    with open(qwen_file) as f:
        code = f.read()

    # Replace ALL instances of setting config.layers_block_type
    code = code.replace(
        "config.layers_block_type = []",
        "object.__setattr__(config, 'layers_block_type', [])"
    )
    # Also handle append calls after the override
    code = code.replace(
        "config.layers_block_type.append",
        "config.__dict__.get('layers_block_type', config.layers_block_type if hasattr(type(config), 'layers_block_type') else []).append"
    )

    with open(qwen_file, 'w') as f:
        f.write(code)
    print(f"Patched layers_block_type in {qwen_file}")
else:
    print("qwen3_5.py not found, skipping")

# 2. Patch rope_scaling empty dict
rope_file = find_file("/usr/**/sglang/srt/layers/rotary_embedding.py")
if rope_file:
    with open(rope_file) as f:
        code = f.read()

    code = code.replace(
        "elif rope_scaling is None:",
        "elif rope_scaling is None or rope_scaling == {}:"
    )

    with open(rope_file, 'w') as f:
        f.write(code)
    print(f"Patched rope_scaling in {rope_file}")
else:
    print("rotary_embedding.py not found, skipping")

# 3. Patch AutoImageProcessor.register() exist_ok conflict
utils_file = find_file("/usr/**/sglang/srt/configs/utils.py")
if utils_file:
    with open(utils_file) as f:
        code = f.read()

    code = code.replace(", exist_ok=True", "")
    code = code.replace("exist_ok=True", "")

    with open(utils_file, 'w') as f:
        f.write(code)
    print(f"Patched AutoImageProcessor in {utils_file}")
else:
    print("configs/utils.py not found, skipping")

print("All patches applied!")
