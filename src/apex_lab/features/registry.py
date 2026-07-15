"""Feature group registry.

Provides a centralised, duplicate-safe store for :class:`FeatureGroup`
instances.  Feature groups self-register into the module-level
:data:`default_registry` when their module is imported.

Example:
    >>> from apex_lab.features.registry import default_registry
    >>> default_registry.list_groups()
    ['momentum', 'price', 'structure', 'time', 'trend', 'volatility', 'volume']
"""

from __future__ import annotations

import logging

from apex_lab.features.base import FeatureGroup

logger = logging.getLogger(__name__)


class FeatureRegistry:
    """Thread-safe registry that maps group names to :class:`FeatureGroup` instances.

    Groups are stored in insertion order.  Duplicate registration of the same
    name raises a :exc:`ValueError` to prevent silent overwrites.

    Example:
        >>> registry = FeatureRegistry()
        >>> registry.register(MyGroup())
        >>> registry.list_groups()
        ['my_group']
    """

    def __init__(self) -> None:
        """Initialise an empty registry."""
        self._groups: dict[str, FeatureGroup] = {}
        logger.debug("FeatureRegistry initialised")

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def register(self, group: FeatureGroup) -> None:
        """Register a feature group.

        Args:
            group: A concrete :class:`FeatureGroup` instance to register.

        Raises:
            ValueError: If a group with the same :attr:`~FeatureGroup.name`
                is already registered.
        """
        if group.name in self._groups:
            raise ValueError(
                f"Feature group '{group.name}' is already registered. "
                "Use a unique name or deregister the existing group first."
            )
        self._groups[group.name] = group
        logger.info("Registered feature group: '%s'", group.name)

    def deregister(self, name: str) -> None:
        """Remove a feature group from the registry.

        Args:
            name: The :attr:`~FeatureGroup.name` of the group to remove.

        Raises:
            KeyError: If *name* is not found in the registry.
        """
        if name not in self._groups:
            raise KeyError(f"Feature group '{name}' is not registered.")
        del self._groups[name]
        logger.info("Deregistered feature group: '%s'", name)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get(self, name: str) -> FeatureGroup:
        """Retrieve a registered feature group by name.

        Args:
            name: The :attr:`~FeatureGroup.name` of the group.

        Returns:
            The :class:`FeatureGroup` instance.

        Raises:
            KeyError: If *name* is not found in the registry.
        """
        if name not in self._groups:
            raise KeyError(
                f"Feature group '{name}' is not registered. "
                f"Available groups: {self.list_groups()}"
            )
        return self._groups[name]

    def list_groups(self) -> list[str]:
        """Return the names of all registered groups in insertion order.

        Returns:
            Ordered list of group name strings.
        """
        return list(self._groups.keys())

    def all_groups(self) -> list[FeatureGroup]:
        """Return all registered :class:`FeatureGroup` instances in insertion order.

        Returns:
            Ordered list of :class:`FeatureGroup` objects.
        """
        return list(self._groups.values())

    def __len__(self) -> int:
        """Return the number of registered groups."""
        return len(self._groups)

    def __contains__(self, name: str) -> bool:
        """Support ``"price" in registry`` membership test."""
        return name in self._groups

    def __repr__(self) -> str:
        """Return developer-friendly representation."""
        return f"FeatureRegistry(groups={self.list_groups()})"


# ---------------------------------------------------------------------------
# Module-level default registry – populated when group modules are imported
# ---------------------------------------------------------------------------

default_registry: FeatureRegistry = FeatureRegistry()
"""Default global registry.  All built-in feature groups self-register here."""
