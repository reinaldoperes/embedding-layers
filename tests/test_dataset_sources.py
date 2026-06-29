"""Tests for optional external dataset loaders."""

import builtins
import sys
import types

import pytest

from embedding_layers.dataset_sources import _load_huggingface_dataset


def test_huggingface_loader_reports_optional_dependency(monkeypatch) -> None:
    """Missing datasets dependency should produce an actionable error."""

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "datasets":
            raise ModuleNotFoundError("No module named 'datasets'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(RuntimeError, match="pip install -e"):
        _load_huggingface_dataset("ag_news", "train")


def test_huggingface_loader_tries_dataset_fallbacks(monkeypatch) -> None:
    """Loader should try later dataset identifiers when an earlier one fails."""

    calls = []

    def fake_load_dataset(dataset_name, split):
        calls.append((dataset_name, split))
        if dataset_name == "primary/name":
            raise ValueError("bad dataset")
        return [{"text": "ok", "label": 0}]

    fake_datasets = types.SimpleNamespace(load_dataset=fake_load_dataset)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    result = _load_huggingface_dataset(("primary/name", "fallback/name"), "train")

    assert result == [{"text": "ok", "label": 0}]
    assert calls == [("primary/name", "train"), ("fallback/name", "train")]
