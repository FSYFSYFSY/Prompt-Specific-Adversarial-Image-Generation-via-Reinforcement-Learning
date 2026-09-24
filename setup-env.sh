#!/bin/bash
# ============================================================
# 复现 sglang-cu130 环境
#   目标平台: NVIDIA RTX PRO 6000 Blackwell (sm_120) + CUDA 13
#   关键版本: sglang 0.5.18 / torch 2.13.0+cu130 / transformers 5.12.1
#   适用模型: Qwen3-VL-8B-Instruct, JailJudge-guard, InternVL3-8B,
#             MiniCPM-V-2_6, LLaVA-OneVision
#   用法    : bash setup-env.sh [安装路径]
#             默认安装到 /root/autodl-tmp/sglang-cu130-v2 （数据盘，快）
# ============================================================
set -euo pipefail

ENV="${1:-/root/autodl-tmp/sglang-cu130-v2}"
LOCK="/autodl-fs/data/requirements-sglang-cu130.lock.txt"
PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
TORCH_INDEX="https://download.pytorch.org/whl/cu130"

echo "=== [1/5] 创建 conda 环境: $ENV"
conda create -p "$ENV" python=3.12 -y

PY="$ENV/bin/python"

# 必须先把 CUDA 13 版 torch 装上：PyPI 上的 torch==2.13.0 是 cu12 构建，
# 一旦先装 sglang 就会被解析成 PyPI 版，导致 Blackwell 无法使用（sm_120 不在编译列表）。
echo "=== [2/5] 安装 CUDA 13 版 PyTorch（顺序不能反）"
"$PY" -m pip install --upgrade pip -i "$PIP_MIRROR"
"$PY" -m pip install --index-url "$TORCH_INDEX" \
    torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0

# 锁文件里的 torch==2.13.0 已被上面的 2.13.0+cu130 满足，pip 会跳过，不会降级成 cu12
echo "=== [3/5] 按锁文件安装其余依赖"
"$PY" -m pip install -r "$LOCK" -i "$PIP_MIRROR"

echo "=== [4/5] 校验"
"$PY" - <<'EOF'
import torch, sglang, transformers
assert torch.version.cuda and torch.version.cuda.startswith("13"), \
    f"torch 不是 CUDA 13 构建: {torch.version.cuda}"
assert "sm_120" in torch.cuda.get_arch_list(), "torch 未编译 sm_120，Blackwell 不可用"
assert transformers.__version__ == "5.12.1", f"transformers 版本不符: {transformers.__version__}"
print(f"  torch        : {torch.__version__}  (cuda {torch.version.cuda})")
print(f"  sglang       : {sglang.__version__}")
print(f"  transformers : {transformers.__version__}")
print(f"  GPU          : {torch.cuda.get_device_name(0)}  cap={torch.cuda.get_device_capability(0)}")
print("  ✅ 环境校验通过")
EOF

echo "=== [5/5] 启动服务前必须导出的环境变量"
cat <<VARS

export CUDA_HOME=$ENV/lib/python3.12/site-packages/nvidia/cu13
export PATH=$ENV/bin:\$CUDA_HOME/bin:\$PATH
export MAX_JOBS=12          # flashinfer 首次 JIT 编译并行度
export OMP_NUM_THREADS=8    # 容器默认为 0，会导致 libgomp 报错并退化为单线程
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
VARS

echo
echo "完成。启动方式可参考 /autodl-fs/data/sglang.sh"
