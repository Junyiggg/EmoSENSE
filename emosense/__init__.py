"""EmoSENSE implementation helpers.

The package keeps imports lightweight so configuration and fuzzy-rule tests can
run without loading the image-generation stack.
"""

from .config import EMOTION_ORDER, VAD_COORDINATES

__all__ = ["EMOTION_ORDER", "VAD_COORDINATES"]
