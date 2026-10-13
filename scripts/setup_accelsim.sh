#!/usr/bin/env bash
# All setup exports are confined to this child shell. Never source this script.
set -euo pipefail
if (( $# < 1 )); then
  echo 'Usage: CUDA_HOME=/existing/toolkit SIM_BUILD_JOBS=4 scripts/setup_accelsim.sh /owned/external/checkout-root'
  echo 'Requires existing CMake>=3.17, C++ compiler, flex/bison/bc, zlib/GL/zstd development libraries and network access.'
  exit 0
fi
: "${CUDA_HOME:?Select an existing matching toolkit explicitly}"
export CUDA_INSTALL_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export ACCELSIM_CONFIG=release
setup_root="$(realpath -m -- "$1")"
package_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
framework="$setup_root/accel-sim-framework"
jobs="${SIM_BUILD_JOBS:-4}"
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] && (( jobs <= 16 )) || { echo 'Bound SIM_BUILD_JOBS to1–16' >&2; exit 2; }
for required_tool in git cmake g++ flex bison bc curl tar sha256sum; do
  command -v "$required_tool" >/dev/null || { echo "Missing required tool: $required_tool; install within permitted user/system scope before retry" >&2; exit 1; }
done
mkdir -p "$setup_root/logs" "$setup_root/downloads"
checkout_pin() {
  local url="$1" destination="$2" revision="$3"
  if [[ ! -d "$destination/.git" ]]; then
    git clone --filter=blob:none --no-checkout "$url" "$destination"
    git -C "$destination" checkout --detach "$revision"
  fi
  [[ "$(git -C "$destination" rev-parse HEAD)" == "$revision" ]] || { echo "Unexpected checkout $destination" >&2; exit 1; }
  if [[ "$destination" == "$framework" ]]; then return; fi
  [[ -z "$(git -C "$destination" diff --name-only)" && -z "$(git -C "$destination" diff --cached --name-only)" ]] || { echo "Tracked source changes at $destination; record/review before build" >&2; exit 1; }
}
checkout_pin https://github.com/accel-sim/accel-sim-framework.git "$framework" 64653015f85fb5664c84a10f48527e8897d289d0
checkout_pin https://github.com/accel-sim/gpgpu-sim_distribution.git "$framework/gpu-simulator/gpgpu-sim" 91880c53383d5a6a6742bfb1be2c5f34e39c7871
checkout_pin https://github.com/pybind/pybind11.git "$framework/gpu-simulator/extern/pybind11" 6254086d96062249c5153c17eab6b4437dd1589f
mapfile -t patch_spec < <(python3 -c 'import json,sys; p=json.load(open(sys.argv[1]))["reviewed_local_patch"]; print(p["path"],p["sha256"],p["source_path"],p["original_source_sha256"],p["patched_source_sha256"],sep="\n")' "$package_dir/configs/simulator_pin.json")
patch_file="$package_dir/${patch_spec[0]}"
source_file="$framework/${patch_spec[2]}"
[[ "$(sha256sum "$patch_file" | cut -d ' ' -f1)" == "${patch_spec[1]}" ]] || { echo 'Reviewed patch hash mismatch' >&2; exit 1; }
source_hash="$(sha256sum "$source_file" | cut -d ' ' -f1)"
if [[ "$source_hash" == "${patch_spec[3]}" ]]; then
  git -C "$framework" apply --check "$patch_file"
  git -C "$framework" apply "$patch_file"
fi
[[ "$(sha256sum "$source_file" | cut -d ' ' -f1)" == "${patch_spec[4]}" ]] || { echo 'Reviewed patched source hash mismatch' >&2; exit 1; }
[[ "$(git -C "$framework" diff --name-only)" == "${patch_spec[2]}" && -z "$(git -C "$framework" diff --cached --name-only)" ]] || { echo 'Unreviewed tracked framework changes' >&2; exit 1; }
cd "$framework"
set +u
source gpu-simulator/setup_environment.sh release > "$setup_root/logs/build_environment.log" 2>&1
set -u
extra=()
if [[ -n "${ZSTD_INCLUDE_DIR:-}" && -n "${ZSTD_LIBRARY:-}" ]]; then
  extra+=("-DCMAKE_CXX_FLAGS=-I$ZSTD_INCLUDE_DIR" "-DZSTD_LIB=$ZSTD_LIBRARY")
fi
cmake -S gpu-simulator -B gpu-simulator/build/release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 "${extra[@]}" > "$setup_root/logs/configure.log" 2>&1
cmake --build gpu-simulator/build/release --parallel "$jobs" > "$setup_root/logs/build.log" 2>&1
cmake --install gpu-simulator/build/release > "$setup_root/logs/install.log" 2>&1
nvbit_dir="$framework/util/tracer_nvbit/nvbit_release"
archive="$setup_root/downloads/nvbit-Linux-x86_64-1.8.tar.bz2"
if [[ ! -d "$nvbit_dir" ]]; then
  curl --fail --location --max-filesize 524288000 https://github.com/NVlabs/NVBit/releases/download/v1.8/nvbit-Linux-x86_64-1.8.tar.bz2 -o "$archive"
  [[ "$(sha256sum "$archive" | cut -d ' ' -f1)" == 72a2b827f9531dcb86b6be13844f267640fb440929d92944177029da6da2b9e1 ]] || { echo 'NVBit release archive checksum mismatch' >&2; exit 1; }
  sha256sum "$archive" > "$setup_root/logs/nvbit_download.sha256"
  mkdir -p "$nvbit_dir"
  tar -xf "$archive" -C "$nvbit_dir" --strip-components=1
else
  echo 'Existing NVBit directory: verify version/source hashes; this script never replaces it automatically.'
fi
[[ "$(sha256sum "$nvbit_dir/core/libnvbit.a" | cut -d ' ' -f1)" == db221829106673bcd69d1e05766136f80e2df4497fb8415d8c2f298b96c302f7 ]] || { echo 'Existing NVBit core differs from reviewed1.8 release' >&2; exit 1; }
make -C util/tracer_nvbit/tracer_tool ARCH=sm_90 -j "$jobs" > "$setup_root/logs/tracer_build.log" 2>&1
postdir="$framework/util/tracer_nvbit/tracer_tool/traces-processing"
if [[ -n "${ZSTD_INCLUDE_DIR:-}" && -n "${ZSTD_LIBRARY:-}" ]]; then
  g++ -std=c++17 -O3 -g -pthread -I"$ZSTD_INCLUDE_DIR" "$postdir/post-traces-processing.cpp" "$ZSTD_LIBRARY" -o "$postdir/post-traces-processing" > "$setup_root/logs/postprocessor_build.log" 2>&1
else
  make -C "$postdir" -j "$jobs" > "$setup_root/logs/postprocessor_build.log" 2>&1
fi
test -x gpu-simulator/bin/release/accel-sim.out
test -x util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing
test -f util/tracer_nvbit/tracer_tool/tracer_tool.so
sha256sum gpu-simulator/bin/release/accel-sim.out util/tracer_nvbit/tracer_tool/tracer_tool.so util/tracer_nvbit/tracer_tool/traces-processing/post-traces-processing > "$setup_root/logs/binary_manifest.sha256"
echo 'Pinned CPU tools built; actual H100 capture/replay/calibration remains pending. Open a fresh shell for native GPU execution.'
