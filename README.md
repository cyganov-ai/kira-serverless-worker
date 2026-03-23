# Kira Surge 1 Serverless Worker

Custom RunPod Serverless worker for Kira Surge 1 (Qwen3.5-27B SFT+DPO).

Based on `runpod/worker-sglang:2.0.2` with transformers 5.3+ for Qwen3.5 support.

## Environment Variables

- `MODEL_NAME`: `cyganovroman/kira-surge-1`
- `HF_TOKEN`: Your HuggingFace token
- `CONTEXT_LENGTH`: `4096`
- `DTYPE`: `bfloat16`
