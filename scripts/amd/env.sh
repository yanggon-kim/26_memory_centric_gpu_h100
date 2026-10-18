# Source on an AMD HPC Fund compute node: HF eager route on the cluster PyTorch module.
# Records nothing by itself; inventory.py captures versions.
module load pytorch/2.10.0 >/dev/null 2>&1
export WORK=${WORK:-/work1/proj_blaisetine/yanggon}
export HF_HOME=$WORK/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export AMD_VENV=$WORK/venvs/hf_rocm_py312
export PY=$AMD_VENV/bin/python
export ROCM_PATH=${ROCM_PATH:-$(dirname "$(dirname "$(command -v hipcc)")")}
export TORCHINDUCTOR_CACHE_DIR=$WORK/cache/inductor
unset ROCR_VISIBLE_DEVICES
