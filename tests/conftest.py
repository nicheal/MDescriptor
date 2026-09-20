"""Repository-wide test hooks for the optional GPU suite."""

from __future__ import annotations

import pytest

from tests._cuda import _strict_mode


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if not _strict_mode() or item.get_closest_marker("gpu") is None:
        return
    if report.outcome == "skipped":
        reason = report.longrepr
        report.outcome = "failed"
        report.longrepr = f"strict GPU gate rejected skip: {reason}"

