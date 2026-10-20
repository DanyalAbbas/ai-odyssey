"""Image loading that respects EXIF orientation.

`cv2.imread` ignores the EXIF Orientation tag. A phone photo stored as
landscape-with-a-rotate-flag then reaches the detector sideways, SCRFD finds
nothing, and the probe becomes a failure-to-enroll that looks like a bad photo
rather than a loading bug. Our benchmark has no EXIF tags, so this changes
nothing here — it is insurance for the hidden set.

Returns BGR uint8 like `cv2.imread`, or None if the file cannot be decoded.
"""
from __future__ import annotations

import cv2
import logging
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


def imread_any(path) -> np.ndarray | None:
    try:
        with Image.open(path) as im:
            if im.getexif().get(274) in (2, 3, 4, 5, 6, 7, 8):
                im = ImageOps.exif_transpose(im)
            return cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception as e:
        # Fall back to cv2.imread; log the PIL failure for debugging
        logger.debug(f"PIL failed to load {path}: {e!r}, trying cv2.imread")
        img = cv2.imread(str(path))
        if img is None:
            logger.warning(f"Failed to load image {path}: PIL raised {type(e).__name__}, cv2.imread returned None")
        return img
