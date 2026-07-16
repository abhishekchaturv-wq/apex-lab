"""Tests for the historical data CLI."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import polars as pl
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "data.py"


def _load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("data_cli_script", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_download_dispatches_to_download_symbol(capsys: pytest.CaptureFixture[str]) -> None:
    """The download command should dispatch to download_symbol."""
    data_cli = _load_script_module()
    expected_df = pl.DataFrame({"close": [1.0, 2.0, 3.0]})

    with (
        patch.object(data_cli, "_resolve_data_dir", return_value=Path("/tmp/lake")),
        patch.object(data_cli, "download_symbol", return_value=expected_df) as mock_download,
    ):
        exit_code = data_cli.main(
            [
                "download",
                "--symbol",
                "BANKNIFTY",
                "--interval",
                "30minute",
                "--from",
                "2016-01-01",
            ]
        )

    assert exit_code == 0
    mock_download.assert_called_once_with("BANKNIFTY", "30minute", "2016-01-01", "today")
    captured = capsys.readouterr()
    assert "Downloading BANKNIFTY" in captured.out
    assert "/tmp/lake/raw/30minute/BANKNIFTY.parquet" in captured.out


def test_update_dispatches_to_update_symbol(capsys: pytest.CaptureFixture[str]) -> None:
    """The update command should dispatch to update_symbol."""
    data_cli = _load_script_module()
    expected_df = pl.DataFrame({"close": [1.0, 2.0]})

    with (
        patch.object(data_cli, "_resolve_data_dir", return_value=Path("/tmp/lake")),
        patch.object(data_cli, "update_symbol", return_value=expected_df) as mock_update,
    ):
        exit_code = data_cli.main(
            [
                "update",
                "--symbol",
                "BANKNIFTY",
                "--interval",
                "30minute",
            ]
        )

    assert exit_code == 0
    mock_update.assert_called_once_with("BANKNIFTY", "30minute")
    captured = capsys.readouterr()
    assert "Updating BANKNIFTY" in captured.out
    assert "Dataset now contains 2 candles" in captured.out


def test_refresh_dispatches_to_refresh_instruments(capsys: pytest.CaptureFixture[str]) -> None:
    """The refresh-instruments command should dispatch to refresh_instruments."""
    data_cli = _load_script_module()
    expected_df = pl.DataFrame({"tradingsymbol": ["BANKNIFTY", "NIFTY"]})

    with (
        patch.object(data_cli, "_resolve_data_dir", return_value=Path("/tmp/lake")),
        patch.object(data_cli, "refresh_instruments", return_value=expected_df) as mock_refresh,
    ):
        exit_code = data_cli.main(["refresh-instruments"])

    assert exit_code == 0
    mock_refresh.assert_called_once_with()
    captured = capsys.readouterr()
    assert "Refreshing instrument master" in captured.out
    assert "/tmp/lake/reference/instruments.parquet" in captured.out


def test_runtime_failure_returns_non_zero_without_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Runtime command failures should return a readable non-zero result."""
    data_cli = _load_script_module()

    with patch.object(data_cli, "update_symbol", side_effect=RuntimeError("boom")):
        exit_code = data_cli.main(
            [
                "update",
                "--symbol",
                "BANKNIFTY",
                "--interval",
                "30minute",
            ]
        )

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Error: boom" in captured.err
    assert "Traceback" not in captured.err


def test_invalid_interval_exits_with_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    """Invalid parser arguments should raise the standard argparse error."""
    data_cli = _load_script_module()

    with pytest.raises(SystemExit) as exc_info:
        data_cli.main(
            [
                "download",
                "--symbol",
                "BANKNIFTY",
                "--interval",
                "weekly",
                "--from",
                "2016-01-01",
            ]
        )

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "invalid choice" in captured.err


@pytest.mark.parametrize(
    ("args", "expected_text"),
    [
        (["--help"], "Historical data CLI for the APEX Lab data engine."),
        (["download", "--help"], "--overwrite"),
        (["update", "--help"], "Download only missing data for a symbol."),
        (["refresh-instruments", "--help"], "Refresh the instrument master file."),
    ],
)
def test_help_output_runs_as_script(args: list[str], expected_text: str) -> None:
    """Help output should work when the script is executed directly."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert expected_text in result.stdout
