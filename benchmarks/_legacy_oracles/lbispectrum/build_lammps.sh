#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../.." && pwd)"
archive="$repo_root/.deps/lammps-stable.tar.gz"
build_root="${LBISPECTRUM_ORACLE_BUILD_ROOT:-/tmp/mdescriptor-lammps-22Jul2025}"
source_root="$build_root/lammps-22Jul2025"
build_dir="$build_root/build-serial"
bin_dir="$build_root/bin"
expected_sha256="21cbbb7424520958c725d99a20d46c8a6cc2a54a92cca0ff35e5854d7d9b9bff"

actual_sha256="$(sha256sum "$archive" | awk '{print $1}')"
if [[ "$actual_sha256" != "$expected_sha256" ]]; then
  echo "LAMMPS archive hash mismatch: $actual_sha256" >&2
  exit 1
fi

mkdir -p "$build_root" "$bin_dir"
tar -xzf "$archive" -C "$build_root"
"$repo_root/.venv/bin/cmake" \
  -S "$source_root/cmake" \
  -B "$build_dir" \
  -D CMAKE_BUILD_TYPE=Release \
  -D BUILD_MPI=off \
  -D BUILD_OMP=off \
  -D PKG_ML-SNAP=on \
  -D BUILD_SHARED_LIBS=off >&2
"$repo_root/.venv/bin/cmake" --build "$build_dir" --parallel >&2
ln -sfn "$build_dir/lmp" "$bin_dir/lmp_serial"
printf '%s\n' "$bin_dir/lmp_serial"
