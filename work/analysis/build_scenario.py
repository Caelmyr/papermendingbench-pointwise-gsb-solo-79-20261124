"""Derive a registered scenario pair with controlled back-side show-through.

Originals in /workspace/work/photos stay untouched. The derived pair keeps the
same 592x888 geometry, so the existing full-frame corner registration remains
valid. Additions:
  * transmitted side: three clear neutral-gray back-side writing strokes
  * reflected side: a faint, near-neutral show-through of the same strokes
The two pre-existing black squares (missing shots) are preserved.
"""
import sys
from pathlib import Path

sys.path.insert(0, "/opt/pointwise-env/site-packages")
sys.path.insert(0, "/workspace")

import cv2 as cv
import numpy as np

SRC = Path("/workspace/work/photos")
DST = Path("/workspace/work/photos/derived")
DST.mkdir(parents=True, exist_ok=True)

face = cv.imread(str(SRC / "face-paper.png"), cv.IMREAD_UNCHANGED)  # BGRA
trans = cv.imread(str(SRC / "transmitted-paper.png"), cv.IMREAD_UNCHANGED)
assert face.shape == trans.shape == (592, 888, 4)

# Back-side writing ("characters"): three short strokes, empty right-hand area.
STROKES = [((585, 185), (665, 182)), ((590, 205), (660, 212)), ((600, 225), (655, 222))]
BACK_WINDOW = (565, 160, 690, 240)  # x0, y0, x1, y1
TRANS_GRAY = (85, 111, 123)          # BGR == RGB(123,111,85), warm neutral like the demo watermark
SHOW_DARK = (70, 86, 96)             # BGR == RGB(96,86,70), warm dark neutral for reflected show-through
ALPHA_SHOW = 0.32                   # reflected show-through strength (visible but faint)

mask = np.zeros(face.shape[:2], np.uint8)
for a, b in STROKES:
    cv.line(mask, a, b, 255, 3, cv.LINE_AA)

# Transmitted: clear back-side strokes in the same neutral family as the watermark.
t_layer = trans.copy()
for a, b in STROKES:
    cv.line(t_layer, a, b, (*TRANS_GRAY, 255), 3, cv.LINE_AA)

# Reflected: faint neutral show-through blended only over the stroke mask.
f_layer = face.copy()
soft = (mask.astype(np.float32) / 255.0 * ALPHA_SHOW)[:, :, None]
target = np.zeros_like(f_layer[:, :, :3], np.float32)
target[:, :] = SHOW_DARK
base = f_layer[:, :, :3].astype(np.float32)
f_layer[:, :, :3] = np.clip(base * (1 - soft) + target * soft, 0, 255).astype(np.uint8)

cv.imwrite(str(DST / "face-derived.png"), f_layer)
cv.imwrite(str(DST / "trans-derived.png"), t_layer)
cv.imwrite(str(DST / "derived-back-mask.png"), mask)

# Window overlay for the record (both derived frames).
windows = {
    "W_ink": (170, 208, 265, 300),
    "W_back": BACK_WINDOW,
    "W_blank": (520, 380, 560, 420),
}
for name, frame in (("face", f_layer), ("trans", t_layer)):
    viz = frame.copy()
    colors = {"W_ink": (255, 120, 60), "W_back": (90, 200, 90), "W_blank": (80, 160, 255)}
    for label, (x0, y0, x1, y1) in windows.items():
        cv.rectangle(viz, (x0, y0), (x1, y1), colors[label], 2, cv.LINE_AA)
        cv.putText(viz, label, (x0, y0 - 6), cv.FONT_HERSHEY_SIMPLEX, 0.5, colors[label], 1, cv.LINE_AA)
    # original ink-spread transect
    cv.line(viz, (252, 196), (244, 346), (60, 220, 220), 1, cv.LINE_AA)
    # missing-shot tiles
    cv.rectangle(viz, (461, 335), (478, 350), (0, 0, 255), 2)
    cv.rectangle(viz, (425, 290), (442, 321), (0, 0, 255), 2)
    cv.imwrite(str(DST / f"windows-{name}.png"), viz)

print("wrote derived pair and overlays to", DST)
print("back stroke mask pixels:", int((mask > 0).sum()))
