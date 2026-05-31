# ShopMind 模型微调 Agent

从平台数据库提取电商对话数据，使用 MLX LoRA 微调本地模型，提供基础模型 vs 微调后模型的效果对比。

## 快速开始

### 1. 提取训练数据

```bash
curl -X POST http://localhost:8008/invoke \
  -H "Content-Type: application/json" \
  -d '{"action":"prepare_data"}'
```

### 2. 运行 MLX LoRA 微调（M1 Max 本地执行）

```bash
pip install mlx-lm

# 下载基础模型
python -c "from mlx_lm import load; load('Qwen/Qwen2.5-0.5B-Instruct')"

# 开始微调（约20-40分钟）
mlx_lm.lora \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --train \
  --data ./data \
  --iters 500 \
  --steps-per-eval 100 \
  --val-batches 5

# 微调完成后，设置模型路径
echo "FINETUNED_MODEL_PATH=./mlx_models" >> .env
```

### 3. 对比效果

```bash
curl -X POST http://localhost:8008/invoke \
  -H "Content-Type: application/json" \
  -d '{"action":"compare","prompt":"碳板跑鞋有什么优势？"}'
```

## 为什么微调

| 场景 | 基础模型 | 微调后 |
|------|---------|--------|
| 产品推荐 | 通用建议 | 结合 ShopMind 产品库的具体推荐 |
| 客服话术 | 标准客服语气 | ShopMind 品牌风格 |
| 异议处理 | 泛化回答 | 基于真实销售对话的应对策略 |
