"""Domain randomization package.

Invariant: this package must not depend on unilab.base.*
"""

from .manager import DomainRandomizationManager
from .provider import DomainRandomizationProvider
from .types import (
    DomainRandomizationCapabilities,
    GeomSizeOverride,
    InitRandomizationPlan,
    IntervalRandomizationPlan,
    MeshScaleMultiplier,
    ModelVariantSpec,
    ResetPlan,
    ResetRandomizationPayload,
)

__all__ = [
    "DomainRandomizationCapabilities",
    "DomainRandomizationManager",
    "DomainRandomizationProvider",
    "GeomSizeOverride",
    "InitRandomizationPlan",
    "IntervalRandomizationPlan",
    "MeshScaleMultiplier",
    "ModelVariantSpec",
    "ResetPlan",
    "ResetRandomizationPayload",
]
