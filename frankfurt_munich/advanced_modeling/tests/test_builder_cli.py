"""Restricted data checks must keep their outputs separate from full builds."""
import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.mark.parametrize("module_name,arguments", [
    ("scheduled_timetable", ["--sample-journeys", "1"]),
    ("history_features", ["--allow-partial-archive"]),
    ("network_transfer", ["--month", "2025-11"]),
])
def test_restricted_build_requires_separate_output(monkeypatch, capsys, module_name, arguments):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(sys, "argv", [module_name, *arguments])
    monkeypatch.setattr(pd, "read_parquet", lambda *_args, **_kwargs: pytest.fail("input read before validation"))

    with pytest.raises(SystemExit) as error:
        module.main()

    assert error.value.code == 2
    assert "separate" in capsys.readouterr().err
