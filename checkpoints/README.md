# Checkpoint 工作区

这个目录用于本地存放 LoRA adapters 和训练元数据。Adapter 权重和完整 base model 不进入 Git 跟踪范围。

最终主线：

```text
基础模型：Qwen3-4B-Base
Adapter：checkpoints/sft/sql_sft_v3_qwen3_4b
Pipeline：schema v2 + value-linking v1 + n20 candidates + rerank v15
结果：450/500 = 90.0% SQLite 执行准确率
```

完整复现时需要的本地资产：

```text
hf_cache/models/Qwen3-4B-Base
checkpoints/sft/sql_sft_v3_qwen3_4b/adapter_model.safetensors
```

这些资产不随仓库重新分发。模型和数据集许可证需要分别遵循各自上游协议。
