#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../../.." && pwd)"
archive="$repo_root/.deps/soap_turbo-master.zip"
build_root="${SOAP_TURBO_ORACLE_BUILD_ROOT:-/tmp/mdescriptor-soapturbo-upstream}"
source_root="$build_root/soap_turbo-master"
build_dir="$build_root/build"
expected_sha256="89093dd439dad020526668867f69a136fc229d10e36d7e8357e34e4499410bb8"

actual_sha256="$(sha256sum "$archive" | awk '{print $1}')"
if [[ "$actual_sha256" != "$expected_sha256" ]]; then
  echo "SOAPTurbo archive hash mismatch: $actual_sha256" >&2
  exit 1
fi

mkdir -p "$build_root" "$build_dir"
"$repo_root/.venv/bin/python" -m zipfile -e "$archive" "$build_root"

scipy_libs="$("$repo_root/.venv/bin/python" -c \
  'from pathlib import Path; import scipy; print(Path(scipy.__file__).resolve().parent.parent / "scipy.libs")')"
openblas="$(find "$scipy_libs" -maxdepth 1 -type f -name 'libscipy_openblas*.so' -print -quit)"
if [[ -z "$openblas" ]]; then
  echo "SciPy OpenBLAS library not found below $scipy_libs" >&2
  exit 1
fi

gfortran -O3 -fPIC -shared \
  -J "$build_dir" -I "$build_dir" \
  -o "$build_dir/libsoap_turbo_reference.so" \
  "$source_root/src/mod_types.f90" \
  "$source_root/src/soap_turbo_functions.f90" \
  "$source_root/src/soap_turbo_radial.f90" \
  "$source_root/src/soap_turbo_angular.f90" \
  "$source_root/src/soap_turbo_compress.f90" \
  "$source_root/src/soap_turbo.f90" \
  "$script_dir/soap_turbo_reference.f90" \
  "$script_dir/soap_turbo_lapack_shim.f90" \
  "$script_dir/soap_turbo_lapack_shim_extra.f90" \
  -L "$scipy_libs" \
  -Wl,-rpath,"$scipy_libs" \
  -Wl,-z,lazy \
  -l:"$(basename "$openblas")"

printf '%s\n' "$build_dir/libsoap_turbo_reference.so"
