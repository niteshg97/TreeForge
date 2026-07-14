"""Smoke test ensuring the example script's entry point is importable."""

import importlib


def test_basic_usage_module_imports() -> None:
    module = importlib.import_module("examples.basic_usage")
    assert hasattr(module, "main")
