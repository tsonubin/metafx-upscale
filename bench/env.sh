# Source before running bench scripts: venv + MPS memory cap (torch's default lets
# the process grow past physical RAM and swap; high=1.0 caps it at recommended max).
source "$(dirname "${BASH_SOURCE[0]:-$0}")/../third_party/venv/bin/activate"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.9
export PYTHONWARNINGS=ignore
