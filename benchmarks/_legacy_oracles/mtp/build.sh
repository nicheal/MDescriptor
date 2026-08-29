#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../.." && pwd)"
archive="$repo_root/.deps/mlip-4-main.zip"
build_root="${MTP_ORACLE_BUILD_ROOT:-/tmp/mdescriptor-mlip4}"
source_root="$build_root/mlip-4-main"
build_dir="$build_root/build"
cache_dir="$build_root/cache"
runtime_home="$build_root/home"
expected_sha256="f2c2ebcfed52aab45c0a94f3361da543935bc6924a671741b5ac60fab5813d97"

actual_sha256="$(sha256sum "$archive" | awk '{print $1}')"
if [[ "$actual_sha256" != "$expected_sha256" ]]; then
  echo "MLIP-4 archive hash mismatch: $actual_sha256" >&2
  exit 1
fi

mkdir -p "$build_root" "$cache_dir" "$runtime_home"
"$repo_root/.venv/bin/python" -m zipfile -e "$archive" "$build_root"
HOME="$runtime_home" "$repo_root/.venv/bin/cmake" \
  -S "$source_root" \
  -B "$build_dir" \
  -D WITH_LIB=OFF \
  -D WITH_LIB_INTERFACE=ON \
  -D WITH_TESTS=OFF \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_FLAGS=-O3 \
  -D CMAKE_CXX_FLAGS_RELEASE="-O3 -DMLIP4_BENCHMARK_CACHE_DIR=\\\"$cache_dir\\\"" >&2
HOME="$runtime_home" "$repo_root/.venv/bin/cmake" --build "$build_dir" --parallel >&2

/usr/bin/c++ -O3 -std=gnu++17 \
  -DMLIP4_BENCHMARK_CACHE_DIR="\"$cache_dir\"" \
  -I"$source_root" \
  -I"$source_root/src" \
  -I"$source_root/src/python" \
  -I"$source_root/make" \
  -I"$source_root/external" \
  -I"$source_root/external/tensor" \
  -I"$source_root/external/program/src" \
  -I"$source_root/external/json-io" \
  -I"$source_root/external/high-precision-float" \
  -I"$source_root/external/mlip-4-eigen" \
  -I"$build_dir" \
  "$script_dir/official_mlip4_mtp.cpp" \
  "$build_dir/lib_mlip_4_interface.a" \
  -ldl -pthread \
  -o "$build_root/official_mlip4_mtp"

printf '%s\n' "$build_root/official_mlip4_mtp"
