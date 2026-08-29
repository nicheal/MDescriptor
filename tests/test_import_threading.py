"""Regression coverage for embedded-GUI import and stdin interactions."""

from __future__ import annotations

import subprocess
import sys


def test_windows_preload_loads_packaged_binary_before_native_import(monkeypatch) -> None:
    from mdescriptor import _runtime

    loaded: list[str] = []
    dll_directories: list[str] = []

    class FakePath:
        def __init__(self, value: str) -> None:
            self.value = value

        def resolve(self):
            return self

        @property
        def parent(self):
            return FakePath("C:/package")

        def glob(self, pattern: str):
            assert pattern == "_native*.pyd"
            return [FakePath("C:/package/_native.cp312-win_amd64.pyd")]

        def __str__(self) -> str:
            return self.value

    def add_dll_directory(path: str) -> None:
        dll_directories.append(path)

    def load(path: str):
        loaded.append(path)
        return object()

    monkeypatch.setattr(_runtime, "Path", FakePath)
    monkeypatch.setattr(_runtime.os, "name", "nt")
    monkeypatch.setattr(_runtime.os, "add_dll_directory", add_dll_directory, raising=False)
    monkeypatch.setattr(_runtime.ctypes, "WinDLL", load, raising=False)
    monkeypatch.setattr(_runtime, "_DLL_DIRECTORIES", [])
    monkeypatch.setattr(_runtime, "_NATIVE_HANDLES", [])

    _runtime.preload_native_binary()

    assert dll_directories == ["C:/package"]
    assert loaded == ["C:/package/_native.cp312-win_amd64.pyd"]


def test_windows_preload_failure_marks_native_extension_unavailable(monkeypatch) -> None:
    from mdescriptor import _runtime

    class FakePath:
        def __init__(self, value: str) -> None:
            self.value = value

        def resolve(self):
            return self

        @property
        def parent(self):
            return FakePath("C:/package")

        def glob(self, pattern: str):
            assert pattern == "_native*.pyd"
            return [FakePath("C:/package/_native.cp312-win_amd64.pyd")]

        def __str__(self) -> str:
            return self.value

    def add_dll_directory(path: str) -> None:
        del path

    def load(path: str):
        raise OSError(f"cannot load {path}")

    monkeypatch.setattr(_runtime, "Path", FakePath)
    monkeypatch.setattr(_runtime.os, "name", "nt")
    monkeypatch.setattr(_runtime.os, "add_dll_directory", add_dll_directory, raising=False)
    monkeypatch.setattr(_runtime.ctypes, "WinDLL", load, raising=False)
    monkeypatch.setattr(_runtime, "_DLL_DIRECTORIES", [])
    monkeypatch.setattr(_runtime, "_NATIVE_HANDLES", [])
    monkeypatch.setattr(_runtime, "_NATIVE_PRELOAD_ATTEMPTED", False)

    _runtime.preload_native_binary()

    assert _runtime.native_extension_available() is False


def test_create_descriptor_does_not_block_with_open_stdin_reader() -> None:
    """Keep stdin open while a worker performs the first native import.

    This mirrors the GUI host shape from issue 1.  In particular, the parent
    must wait without closing the child's stdin; ``communicate`` would hide
    the deadlock by sending EOF before waiting.
    """

    script = r'''
import threading
import sys

import mdescriptor

assert "mdescriptor._native" not in sys.modules


def read_stdin():
    for _line in sys.stdin:
        pass


threading.Thread(target=read_stdin, daemon=True).start()
metadata = mdescriptor.describe_descriptor("ACE")
parameters = {
    name: ([1] if name == "species" else schema["default"])
    for name, schema in metadata["parameters"].items()
    if name == "species" or "default" in schema
}
descriptor = mdescriptor.create_descriptor(
    mdescriptor.DescriptorConfiguration(1, "ACE", parameters)
)
descriptor.close()
print("BUILD_DONE", flush=True)
'''
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        try:
            returncode = process.wait(timeout=10)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.wait()
            stdout = process.stdout.read() if process.stdout is not None else ""
            stderr = process.stderr.read() if process.stderr is not None else ""
            raise AssertionError(
                "native import/build blocked while stdin remained open; "
                f"stdout={stdout!r}, stderr={stderr!r}"
            ) from exc
        stdout = process.stdout.read() if process.stdout is not None else ""
        stderr = process.stderr.read() if process.stderr is not None else ""
    finally:
        if process.stdin is not None:
            process.stdin.close()

    assert returncode == 0, stderr
    assert "BUILD_DONE" in stdout
