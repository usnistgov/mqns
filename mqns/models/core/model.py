from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mqns.models.error import ErrorModel


class QuantumModel(ABC):
    """Abstract backend model for quantum data."""

    @abstractmethod
    def apply_error(self, error: "ErrorModel") -> None:
        """
        Apply an error model.

        Args:
            error: error model with assigned error probability.
        """
