"""Restricted data checks must keep their outputs separate from full builds."""
import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@pytest.mark.parametrize("module_name,arguments", [
    ("scheduled_timetable", ["--sample-journeys", "1"]),
    ("history_features", ["--allow-partial-archive"]),
    ("network_state", ["--sample-month", "2025-11"]),
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


def test_network_output_follows_selected_data_directory(monkeypatch, tmp_path):
    import network_state

    rows = pd.DataFrame()
    outputs = []
    monkeypatch.setattr(sys, "argv", ["network_state", "--data-dir", str(tmp_path)])
    monkeypatch.setattr(network_state, "_load_prediction_rows", lambda directory, split: rows)
    monkeypatch.setattr(network_state, "build_network_features", lambda *_args, **_kwargs: (rows, {}))
    monkeypatch.setattr(network_state, "_write_output", lambda path, *_args: outputs.append(path))

    network_state.main()

    assert outputs == [tmp_path / "advanced" / "network_cv.parquet"]
