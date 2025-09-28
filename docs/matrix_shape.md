| model name | 矩阵尺寸 | 矩阵来源 | 备注 |
|---|---|---|---|
| Llama-3 8B | 4096 × 4096 | Attention Q投影 (`q_proj`) | hidden=4096, heads=32, head_dim=128；GQA: kv_heads=8 |
| Llama-3 8B | 4096 × 1024 | Attention K投影 (`k_proj`, GQA) | 8×128=1024（KV 组维） |
| Llama-3 8B | 4096 × 1024 | Attention V投影 (`v_proj`, GQA) | 同上 |
| Llama-3 8B | 4096 × 4096 | Attention O投影 (`o_proj`) | 输出回 hidden |
| Llama-3 8B | 4096 × 14336 | MLP `gate_proj`/`up_proj` | SwiGLU 上升矩阵 |
| Llama-3 8B | 14336 × 4096 | MLP `down_proj` | 回缩到 hidden |
| Llama-3 70B | 8192 × 8192 | Attention Q/O投影 | hidden=8192, heads=64, kv_heads=8, inter=28672 |
| Llama-3 70B | 8192 × 1024 | Attention K/V投影 (GQA) | 8×128=1024 |
| Llama-3 70B | 8192 × 28672 | MLP `gate/up` | SwiGLU |
| Llama-3 70B | 28672 × 8192 | MLP `down` | — |
| Llama-3.2 3B | 3072 × 3072 | Attention Q/O投影 | hidden=3072, heads=24, head_dim=128, inter=8192 |
| Llama-3.2 3B | 3072 × 1024 | Attention K/V投影 (GQA) | kv_heads=8 → 8×128=1024 |
| Llama-3.2 3B | 3072 × 8192 | MLP `gate/up` | — |
| Llama-3.2 3B | 8192 × 3072 | MLP `down` | — |
| Llama-3.2 1B | 2048 × 2048 | Attention Q/O投影 | hidden=2048, heads=32, head_dim=64, inter=8192 |
| Llama-3.2 1B | 2048 × 512 | Attention K/V投影 (GQA) | kv_heads=8 → 8×64=512 |
| Llama-3.2 1B | 2048 × 8192 | MLP `gate/up` | — |
| Llama-3.2 1B | 8192 × 2048 | MLP `down` | — |
| Qwen3 8B | 4096 × 4096 | Attention Q/O投影 | hidden=4096, inter=22016，层=32 |
| Qwen3 8B | 4096 × 1024 | Attention K/V投影 (GQA) | head_dim=128, kv_heads=8 |
| Qwen3 8B | 4096 × 22016 | MLP `gate/up` | — |
| Qwen3 8B | 22016 × 4096 | MLP `down` | — |
| Phi-2 (2.7B) | 2560 × 2560 | Attention Q/O投影 | hidden=2560, heads=32，inter=10240 |
| Phi-2 (2.7B) | 2560 × 2560 | Attention K/V投影 | KV=Q=32，投影到 hidden |
| Phi-2 (2.7B) | 2560 × 10240 | MLP `gate/up` | GELU 变体 |
| Phi-2 (2.7B) | 10240 × 2560 | MLP `down` | — |
| GPT-OSS 20B（MoE） | 2880 × 4096 | Attention Q投影 | hidden=2880；heads=64, head_dim=64 → 4096 |
| GPT-OSS 20B（MoE） | 2880 × 512 | Attention K/V投影 (GQA) | kv_heads=8 → 8×64=512 |
| GPT-OSS 20B（MoE） | 2880 × 2880 | 每个专家的 `gate/up` | num_local_experts=32，top-k=4 |
| GPT-OSS 20B（MoE） | 2880 × 2880 | 每个专家的 `down` | MoE 层仅激活部分专家 |
| Mixtral 8×7B（MoE） | 4096 × 14336 | 每个专家 `gate/up` | hidden=4096, inter=14336, experts=8, top-k=2 |
| Mixtral 8×7B（MoE） | 14336 × 4096 | 每个专家 `down` | 计算上相当于 12B 激活参数级别 |
| Gemma-2 2B | 2304 × 2304 | Attention Q/O投影 | hidden=2304, heads=8, head_dim=256, inter=9216 |
| Gemma-2 2B | 2304 × 1024 | Attention K/V投影 (GQA) | kv_heads=4 → 4×256=1024 |
| Gemma-2 2B | 2304 × 9216 | MLP `gate/up` | — |
| Gemma-2 2B | 9216 × 2304 | MLP `down` | — |
