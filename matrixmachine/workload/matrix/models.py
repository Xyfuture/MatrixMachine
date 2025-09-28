"""
Matrix configurations for different model workloads.
Based on popular LLM architectures and their matrix dimensions.
"""

from matrixmachine.core.description import MatrixShape


# Llama-3 8B matrices
LLAMA3_8B_Q_PROJ = MatrixShape(rows=4096, cols=4096)
LLAMA3_8B_K_PROJ = MatrixShape(rows=4096, cols=1024)  # GQA: kv_heads=8
LLAMA3_8B_V_PROJ = MatrixShape(rows=4096, cols=1024)
LLAMA3_8B_O_PROJ = MatrixShape(rows=4096, cols=4096)
LLAMA3_8B_GATE_UP = MatrixShape(rows=4096, cols=14336)  # SwiGLU
LLAMA3_8B_DOWN = MatrixShape(rows=14336, cols=4096)


# Llama-3 70B matrices
LLAMA3_70B_Q_PROJ = MatrixShape(rows=8192, cols=8192)
LLAMA3_70B_K_PROJ = MatrixShape(rows=8192, cols=1024)  # GQA
LLAMA3_70B_V_PROJ = MatrixShape(rows=8192, cols=1024)
LLAMA3_70B_O_PROJ = MatrixShape(rows=8192, cols=8192)
LLAMA3_70B_GATE_UP = MatrixShape(rows=8192, cols=28672)
LLAMA3_70B_DOWN = MatrixShape(rows=28672, cols=8192)


# Llama-3.2 3B matrices
LLAMA32_3B_Q_PROJ = MatrixShape(rows=3072, cols=3072)
LLAMA32_3B_K_PROJ = MatrixShape(rows=3072, cols=1024)  # GQA
LLAMA32_3B_V_PROJ = MatrixShape(rows=3072, cols=1024)
LLAMA32_3B_O_PROJ = MatrixShape(rows=3072, cols=3072)
LLAMA32_3B_GATE_UP = MatrixShape(rows=3072, cols=8192)
LLAMA32_3B_DOWN = MatrixShape(rows=8192, cols=3072)


# Llama-3.2 1B matrices
LLAMA32_1B_Q_PROJ = MatrixShape(rows=2048, cols=2048)
LLAMA32_1B_K_PROJ = MatrixShape(rows=2048, cols=512)  # GQA
LLAMA32_1B_V_PROJ = MatrixShape(rows=2048, cols=512)
LLAMA32_1B_O_PROJ = MatrixShape(rows=2048, cols=2048)
LLAMA32_1B_GATE_UP = MatrixShape(rows=2048, cols=8192)
LLAMA32_1B_DOWN = MatrixShape(rows=8192, cols=2048)


# Qwen3 8B matrices
QWEN3_8B_Q_PROJ = MatrixShape(rows=4096, cols=4096)
QWEN3_8B_K_PROJ = MatrixShape(rows=4096, cols=1024)  # GQA
QWEN3_8B_V_PROJ = MatrixShape(rows=4096, cols=1024)
QWEN3_8B_O_PROJ = MatrixShape(rows=4096, cols=4096)
QWEN3_8B_GATE_UP = MatrixShape(rows=4096, cols=22016)
QWEN3_8B_DOWN = MatrixShape(rows=22016, cols=4096)


# Phi-2 (2.7B) matrices
PHI2_Q_PROJ = MatrixShape(rows=2560, cols=2560)
PHI2_K_PROJ = MatrixShape(rows=2560, cols=2560)  # No GQA
PHI2_V_PROJ = MatrixShape(rows=2560, cols=2560)
PHI2_O_PROJ = MatrixShape(rows=2560, cols=2560)
PHI2_GATE_UP = MatrixShape(rows=2560, cols=10240)
PHI2_DOWN = MatrixShape(rows=10240, cols=2560)


# Gemma-2 2B matrices
GEMMA2_2B_Q_PROJ = MatrixShape(rows=2304, cols=2304)
GEMMA2_2B_K_PROJ = MatrixShape(rows=2304, cols=1024)  # GQA
GEMMA2_2B_V_PROJ = MatrixShape(rows=2304, cols=1024)
GEMMA2_2B_O_PROJ = MatrixShape(rows=2304, cols=2304)
GEMMA2_2B_GATE_UP = MatrixShape(rows=2304, cols=9216)
GEMMA2_2B_DOWN = MatrixShape(rows=9216, cols=2304)


# Mixtral 8x7B (MoE) matrices - per expert
MIXTRAL_8X7B_GATE_UP = MatrixShape(rows=4096, cols=14336)
MIXTRAL_8X7B_DOWN = MatrixShape(rows=14336, cols=4096)


# Collections for easy access
LLAMA3_8B_MATRICES = [
    LLAMA3_8B_Q_PROJ, LLAMA3_8B_K_PROJ, LLAMA3_8B_V_PROJ,
    LLAMA3_8B_O_PROJ, LLAMA3_8B_GATE_UP, LLAMA3_8B_DOWN
]

LLAMA3_70B_MATRICES = [
    LLAMA3_70B_Q_PROJ, LLAMA3_70B_K_PROJ, LLAMA3_70B_V_PROJ,
    LLAMA3_70B_O_PROJ, LLAMA3_70B_GATE_UP, LLAMA3_70B_DOWN
]

SMALL_MODEL_MATRICES = [
    LLAMA32_1B_Q_PROJ, LLAMA32_1B_O_PROJ, LLAMA32_1B_GATE_UP,
    GEMMA2_2B_Q_PROJ, GEMMA2_2B_O_PROJ, GEMMA2_2B_GATE_UP,
    PHI2_Q_PROJ, PHI2_O_PROJ, PHI2_GATE_UP
]

LARGE_MODEL_MATRICES = [
    LLAMA3_8B_Q_PROJ, LLAMA3_8B_GATE_UP, LLAMA3_8B_DOWN,
    LLAMA3_70B_Q_PROJ, LLAMA3_70B_GATE_UP, LLAMA3_70B_DOWN,
    QWEN3_8B_Q_PROJ, QWEN3_8B_GATE_UP, QWEN3_8B_DOWN
]

ALL_MATRICES = LLAMA3_8B_MATRICES + LLAMA3_70B_MATRICES + SMALL_MODEL_MATRICES