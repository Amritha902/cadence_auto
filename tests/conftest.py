"""Shared pytest configuration."""

from __future__ import annotations

import shutil

import pytest

from bias.topology import Testbench

# `Testbench` starts with "Test", so pytest tries to collect it as a test class
# and warns that it cannot. It is a dataclass, not a test.
Testbench.__test__ = False


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "needs_ngspice: test requires ngspice on PATH"
    )


def pytest_collection_modifyitems(config, items) -> None:
    if shutil.which("ngspice"):
        return
    skip = pytest.mark.skip(reason="ngspice not installed")
    for item in items:
        if "needs_ngspice" in item.keywords:
            item.add_marker(skip)
