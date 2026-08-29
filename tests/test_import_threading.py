"""Regression coverage for embedded-GUI import and stdin interactions."""

from __future__ import annotations

import subprocess
import sys


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
