# Checkpoint Notes

This directory is a local workspace for LoRA adapters and training metadata.
Adapter weights and full base models are not tracked.

Final mainline:

```text
Base model: Qwen3-4B-Base
Adapter: checkpoints/sft/sql_sft_v3_qwen3_4b
Pipeline: schema v2 + value-linking v1 + n20 candidates + rerank v15
Result: 450/500 = 90.0% SQLite execution accuracy
```

Expected local assets for full reproduction:

```text
hf_cache/models/Qwen3-4B-Base
checkpoints/sft/sql_sft_v3_qwen3_4b/adapter_model.safetensors
```

These artifacts are not redistributed. Keep model and dataset licenses separate
from the repository code license.
