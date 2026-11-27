"""
Autopoiesis-NCA package for quantifying autopoiesis and organizational closure in Neural Cellular Automata.
"""

from .target import create_target_and_masks
from .model import NCA2D
from .perturbations import apply_excision_lesion, apply_boundary_noise
from .metrics import (
    morphological_l2,
    boundary_l2,
    recovery_ratio,
    compute_transfer_entropy_ksg,
)

__all__ = [
    "create_target_and_masks",
    "NCA2D",
    "apply_excision_lesion",
    "apply_boundary_noise",
    "morphological_l2",
    "boundary_l2",
    "recovery_ratio",
    "compute_transfer_entropy_ksg",
]
