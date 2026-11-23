"""Cross-check the Lab pipeline against the bench's native paper_tone function."""
import sys
from dataclasses import dataclass

sys.path.insert(0, "/opt/pointwise-env/site-packages")
sys.path.insert(0, "/workspace")

import cv2 as cv
import numpy as np

from paperbench.analysis import paper_tone

DERIVED = "/workspace/work/photos/derived"
WINDOWS = {
    "W_ink": (170, 208, 265, 300),
    "W_back": (565, 160, 690, 240),
    "W_blank": (520, 380, 560, 420),
}


@dataclass
class Frame:
    rgb: np.ndarray
    support: np.ndarray
    step: tuple

    @property
    def useful(self):
        return self.support & (self.rgb.max(axis=2) >= .018) & (self.rgb.min(axis=2) < .993)


rgba = cv.imread(f"{DERIVED}/face-derived.png", cv.IMREAD_UNCHANGED)
h, w = rgba.shape[:2]
rgb = cv.cvtColor(rgba, cv.COLOR_BGRA2RGB).astype(np.float32) / 255.0
face = Frame(rgb, rgba[:, :, 3] >= 254, (222.0 / (w - 1), 148.0 / (h - 1)))


def mark_from_box(box):
    x0, y0, x1, y1 = box
    return {"points": [[x0 / (w - 1), y0 / (h - 1)], [x1 / (w - 1), y0 / (h - 1)],
                       [x1 / (w - 1), y1 / (h - 1)], [x0 / (w - 1), y1 / (h - 1)]]}


def lab_rows(mask, label):
    lab = cv.cvtColor(face.rgb, cv.COLOR_RGB2LAB)
    px = lab[mask & face.useful].reshape(-1, 3)
    p10, med, p90 = np.percentile(px, [10, 50, 90], axis=0)
    return {"group": label, "pixels": int(px.shape[0]),
            "L": [round(float(p10[0]), 2), round(float(med[0]), 2), round(float(p90[0]), 2)],
            "a": [round(float(p10[1]), 2), round(float(med[1]), 2), round(float(p90[1]), 2)],
            "b": [round(float(p10[2]), 2), round(float(med[2]), 2), round(float(p90[2]), 2)]}


summary, rows, _ = paper_tone(face, {"region": mark_from_box(WINDOWS["W_blank"])}, {})
print("bench native paper_tone on W_blank ->", summary)
for row in rows:
    print(" ", row)

x0, y0, x1, y1 = WINDOWS["W_blank"]
blank_mask = np.zeros(face.support.shape, bool)
blank_mask[y0:y1, x0:x1] = True
print("\nmy pipeline, same mask ->", lab_rows(blank_mask, "blank"))

back_strokes = cv.imread(f"{DERIVED}/derived-back-mask.png", cv.IMREAD_GRAYSCALE) > 0
print("my stroke-only back row ->", lab_rows(back_strokes, "back strokes only"))
