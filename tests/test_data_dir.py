"""Focused tests for data root directory resolution.

Tests cover the three-level priority order implemented in
:func:`apex_lab.data._resolve_data_dir`:

1. Explicit ``data_dir`` argument
2. ``APEX_DATA_DIR`` environment variable
3. Default ``~/kite-test/apex-data-lake``
"""

from __future__ import annotations

from pathlib import Path

import pytest

from apex_lab.data import _APEX_DATA_DIR_ENV, _DEFAULT_DATA_LAKE, _resolve_data_dir

# ---------------------------------------------------------------------------
# Default resolution
# ---------------------------------------------------------------------------


def test_default_data_dir_is_external_lake(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without any override the default data lake path should be returned."""
    monkeypatch.delenv(_APEX_DATA_DIR_ENV, raising=False)
    result = _resolve_data_dir()
    assert result == _DEFAULT_DATA_LAKE.expanduser()


def test_default_data_dir_expands_tilde(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default path must have ``~`` expanded to an absolute path."""
    monkeypatch.delenv(_APEX_DATA_DIR_ENV, raising=False)
    result = _resolve_data_dir()
    assert not str(result).startswith("~"), "~ must be expanded"
    assert result.is_absolute()


# ---------------------------------------------------------------------------
# APEX_DATA_DIR environment variable
# ---------------------------------------------------------------------------


def test_apex_data_dir_env_overrides_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Setting APEX_DATA_DIR should override the built-in default."""
    monkeypatch.setenv(_APEX_DATA_DIR_ENV, str(tmp_path))
    result = _resolve_data_dir()
    assert result == tmp_path


def test_apex_data_dir_env_expands_tilde(monkeypatch: pytest.MonkeyPatch) -> None:
    """APEX_DATA_DIR values with ``~`` should be expanded."""
    monkeypatch.setenv(_APEX_DATA_DIR_ENV, "~/some/custom/lake")
    result = _resolve_data_dir()
    assert not str(result).startswith("~"), "~ must be expanded"
    assert result.is_absolute()


# ---------------------------------------------------------------------------
# Explicit data_dir argument
# ---------------------------------------------------------------------------


def test_explicit_data_dir_takes_highest_priority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit ``data_dir`` argument must win over env var and default."""
    custom_dir = tmp_path / "explicit"
    monkeypatch.setenv(_APEX_DATA_DIR_ENV, str(tmp_path / "from_env"))
    result = _resolve_data_dir(custom_dir)
    assert result == custom_dir


def test_explicit_data_dir_expands_tilde(tmp_path: Path) -> None:
    """Explicit paths with ``~`` should also be expanded."""
    result = _resolve_data_dir(Path("~/kite-test/apex-data-lake"))
    assert not str(result).startswith("~"), "~ must be expanded"
    assert result.is_absolute()


# ---------------------------------------------------------------------------
# Precedence order: explicit > env > default
# ---------------------------------------------------------------------------


def test_precedence_explicit_beats_env_beats_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full precedence chain: explicit arg > APEX_DATA_DIR > default."""
    env_dir = tmp_path / "env"
    explicit_dir = tmp_path / "explicit"

    # Level 3 — default alone
    monkeypatch.delenv(_APEX_DATA_DIR_ENV, raising=False)
    assert _resolve_data_dir() == _DEFAULT_DATA_LAKE.expanduser()

    # Level 2 — env var beats default
    monkeypatch.setenv(_APEX_DATA_DIR_ENV, str(env_dir))
    assert _resolve_data_dir() == env_dir

    # Level 1 — explicit arg beats env var
    assert _resolve_data_dir(explicit_dir) == explicit_dir
