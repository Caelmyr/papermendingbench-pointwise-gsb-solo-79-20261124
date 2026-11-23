"""Generate the synthetic face/transmitted manuscript photos (data output only)."""
import sys
from pathlib import Path

sys.path.insert(0, "/opt/pointwise-env/site-packages")
sys.path.insert(0, "/workspace")

import cv2 as cv

from paperbench.demo import photograph

out = Path("/workspace/work/photos")
for role, seed in (("face", 91), ("transmitted", 92)):
    rgba = photograph(role, seed)
    cv.imwrite(str(out / f"{role}-paper.png"), cv.cvtColor(rgba, cv.COLOR_RGBA2BGRA))
    print(role, rgba.shape, out / f"{role}-paper.png")
