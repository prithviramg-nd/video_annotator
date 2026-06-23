"""
Annotator registry - dynamically manages available annotators.

This makes it easy to add new annotators without modifying the core pipeline.
Just create a new annotator class, import it here, and register it.
"""

from typing import Dict, List, Type

from .base import BaseAnnotator
from .bbox import FaceBBoxAnnotator, PersonBBoxAnnotator
from .keypoints import (
    NoseAnnotator,
    ShoulderAnnotator,
    EarAnnotator,
)
from .eye_scores import EyeScoresAnnotator
from .head_pose import HeadPoseAnnotator
from .event_window import EventWindowAnnotator
from .frame_info import FrameInfoAnnotator
from .mouth import MouthKeypointsAnnotator
from .variance_graphs import MouthYDistVarianceAnnotator, NoseXRatioVarianceAnnotator


# ── Registry ─────────────────────────────────────────────────────────────────

_REGISTRY: Dict[str, Type[BaseAnnotator]] = {}


def _register(cls: Type[BaseAnnotator]):
    """Register an annotator class by its name."""
    _REGISTRY[cls.name] = cls


# Register all built-in annotators
_register(FaceBBoxAnnotator)
_register(PersonBBoxAnnotator)
_register(NoseAnnotator)
_register(ShoulderAnnotator)
_register(EarAnnotator)
_register(EyeScoresAnnotator)
_register(HeadPoseAnnotator)
_register(EventWindowAnnotator)
_register(FrameInfoAnnotator)
_register(MouthKeypointsAnnotator)
_register(MouthYDistVarianceAnnotator)
_register(NoseXRatioVarianceAnnotator)


class AnnotatorRegistry:
    """
    Central registry of all available annotators.

    Usage:
        registry = AnnotatorRegistry()
        # Get all available annotator names
        registry.available()

        # Get specific annotators by name
        annotators = registry.get(["face_bbox", "nose", "shoulders"])

        # Get all annotators
        annotators = registry.get_all()
    """

    @staticmethod
    def available() -> List[str]:
        """Return list of all registered annotator names."""
        return sorted(_REGISTRY.keys())

    @staticmethod
    def get(names: List[str]) -> List[BaseAnnotator]:
        """
        Instantiate and return annotators by name.

        Args:
            names: list of annotator names to activate.

        Returns:
            List of annotator instances.

        Raises:
            ValueError: if an unknown annotator name is requested.
        """
        annotators = []
        for name in names:
            if name not in _REGISTRY:
                raise ValueError(
                    f"Unknown annotator '{name}'. "
                    f"Available: {sorted(_REGISTRY.keys())}"
                )
            annotators.append(_REGISTRY[name]())
        return annotators

    @staticmethod
    def get_all() -> List[BaseAnnotator]:
        """Instantiate and return all registered annotators."""
        return [cls() for cls in _REGISTRY.values()]

    @staticmethod
    def describe() -> Dict[str, str]:
        """Return {name: description} for all registered annotators."""
        return {name: cls.description for name, cls in _REGISTRY.items()}

    @staticmethod
    def register(cls: Type[BaseAnnotator]):
        """Register a custom annotator at runtime."""
        _register(cls)
