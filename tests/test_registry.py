"""Tests for FeatureRegistry."""

from __future__ import annotations

import polars as pl
import pytest

from apex_lab.features.base import FeatureGroup
from apex_lab.features.registry import FeatureRegistry

# ---------------------------------------------------------------------------
# Stub feature group
# ---------------------------------------------------------------------------


class _StubGroup(FeatureGroup):
    """Minimal feature group for testing."""

    def __init__(self, name: str, warm_up: int = 0) -> None:
        self._name = name
        self._warm_up = warm_up

    @property
    def name(self) -> str:
        return self._name

    @property
    def warm_up_periods(self) -> int:
        return self._warm_up

    def compute(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.with_columns(pl.lit(1.0).alias(f"_{self._name}_feature"))


# ---------------------------------------------------------------------------
# Registration tests
# ---------------------------------------------------------------------------


def test_registry_register_single_group():
    """A single group can be registered without error."""
    registry = FeatureRegistry()
    registry.register(_StubGroup("alpha"))
    assert "alpha" in registry
    assert len(registry) == 1


def test_registry_list_groups_insertion_order():
    """Groups are listed in the order they were registered."""
    registry = FeatureRegistry()
    registry.register(_StubGroup("a"))
    registry.register(_StubGroup("b"))
    registry.register(_StubGroup("c"))
    assert registry.list_groups() == ["a", "b", "c"]


def test_registry_duplicate_raises_value_error():
    """Registering the same name twice raises ValueError."""
    registry = FeatureRegistry()
    registry.register(_StubGroup("duplicate"))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_StubGroup("duplicate"))


def test_registry_get_existing_group():
    """get() returns the correct group instance."""
    registry = FeatureRegistry()
    group = _StubGroup("beta")
    registry.register(group)
    retrieved = registry.get("beta")
    assert retrieved is group


def test_registry_get_missing_raises_key_error():
    """get() raises KeyError for an unknown name."""
    registry = FeatureRegistry()
    with pytest.raises(KeyError):
        registry.get("nonexistent")


def test_registry_deregister_removes_group():
    """deregister() removes the group from the registry."""
    registry = FeatureRegistry()
    registry.register(_StubGroup("to_remove"))
    registry.deregister("to_remove")
    assert "to_remove" not in registry
    assert len(registry) == 0


def test_registry_deregister_missing_raises_key_error():
    """deregister() raises KeyError when the name is not present."""
    registry = FeatureRegistry()
    with pytest.raises(KeyError):
        registry.deregister("ghost")


def test_registry_all_groups_returns_instances():
    """all_groups() returns FeatureGroup instances in order."""
    registry = FeatureRegistry()
    g1 = _StubGroup("x")
    g2 = _StubGroup("y")
    registry.register(g1)
    registry.register(g2)
    assert registry.all_groups() == [g1, g2]


def test_registry_contains_operator():
    """'in' operator works correctly."""
    registry = FeatureRegistry()
    registry.register(_StubGroup("present"))
    assert "present" in registry
    assert "absent" not in registry


def test_registry_len():
    """len() reflects the correct count."""
    registry = FeatureRegistry()
    assert len(registry) == 0
    registry.register(_StubGroup("one"))
    assert len(registry) == 1
    registry.register(_StubGroup("two"))
    assert len(registry) == 2


def test_registry_repr():
    """__repr__ includes the group names."""
    registry = FeatureRegistry()
    registry.register(_StubGroup("foo"))
    assert "foo" in repr(registry)


# ---------------------------------------------------------------------------
# Built-in groups registration
# ---------------------------------------------------------------------------


def test_default_registry_has_all_builtin_groups():
    """All built-in groups are registered in the default registry."""
    # Importing apex_lab.features triggers group registration
    from apex_lab.features.registry import default_registry  # noqa: PLC0415

    expected = {"price", "trend", "momentum", "volatility", "volume", "structure", "time"}
    registered = set(default_registry.list_groups())
    assert expected.issubset(registered), f"Missing groups: {expected - registered}"
