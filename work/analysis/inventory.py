"""Inventory features that differ between the calibrated face and transmitted frames."""
import sys

sys.path.insert(0, "/opt/pointwise-env/site-packages")
sys.path.insert(0, "/workspace")

import cv2 as cv
import numpy as np

from paperbench.demo import create
from paperbench.store import Library
from paperbench.photos import decode, plane

library = Library("/workspace/work/library")
run = library.list() and library.run(library.list()[0]["id"])
photos = {p["role"]: p for p in run["photos"]}
face = plane({**photos["face"], "pixels": decode(library.blob(photos["face"]["blob"]).read_bytes())}, run)
trans = plane({**photos["transmitted"], "pixels": decode(library.blob(photos["transmitted"]["blob"]).read_bytes())}, run)

f, t = face.rgb, trans.rgb
uf, ut = face.useful, trans.useful
br_f = f[:, :, 2] - f[:, :, 0]
br_t = t[:, :, 2] - t[:, :, 0]
luma_f = f @ np.array([.2126, .7152, .0722], np.float32)
luma_t = t @ np.array([.2126, .7152, .0722], np.float32)

def stats(name, mask):
    if mask.sum() == 0:
        print(name, "empty")
        return
    print(f"{name:38s} n={int(mask.sum()):6d}  face RGB=({f[mask,0].mean():.3f},"
          f"{f[mask,1].mean():.3f},{f[mask,2].mean():.3f}) B-R={br_f[mask].mean():+.3f} "
          f"L={luma_f[mask].mean():.3f} | trans RGB=({t[mask,0].mean():.3f},"
          f"{t[mask,1].mean():.3f},{t[mask,2].mean():.3f}) B-R={br_t[mask].mean():+.3f} "
          f"L={luma_t[mask].mean():.3f} uf={int((mask&uf).sum())} ut={int((mask&ut).sum())}")

yy, xx = np.indices(uf.shape)

# Ground-truth regions from demo.py geometry (same plane, full-frame corners)
blue = np.zeros(uf.shape, bool)
for start, end in ((186, 332), (186, 300), (202, 344), (180, 282), (193, 319)):
    py = 215 + (end % 86)
    cv.line(blue, (start, py), (end, py + 7), 1, 4, cv.LINE_AA)
cv.line(blue, (344, 229), (400, 228), 1, 3, cv.LINE_AA)
# watermark (transmitted-only drawing), ellipse center (626,325) axes 30x25 + vertical line
wm = np.zeros(uf.shape, np.uint8)
cv.ellipse(wm, (626, 325), (30, 25), 0, 0, 360, 1, 10, cv.LINE_AA)
cv.line(wm, (623, 301), (626, 349), 1, 4, cv.LINE_AA)
wm = wm.astype(bool)
# holes
holes = np.zeros(uf.shape, np.uint8)
for cx, cy, r in ((475, 238, 10), (487, 332, 5), (364, 351, 6)):
    cv.circle(holes, (cx, cy), r, 1, -1, cv.LINE_AA)
holes = holes.astype(bool)
# black squares: face-only defect and transmitted-only defect
face_black = (xx >= 461) & (xx < 478) & (yy >= 335) & (yy < 350)
trans_black = (xx >= 425) & (xx < 442) & (yy >= 290) & (yy < 321)
# blank paper patch (plain area away from marks): (520..560, 380..420)
blank = (xx >= 520) & (xx < 560) & (yy >= 380) & (yy < 420)
# paper support overall minus all features
sheet = face.support
plain = sheet & ~blue & ~wm & ~holes & ~face_black & ~trans_black

stats("blue ink strokes (ground truth front)", blue & sheet)
stats("watermark zone (drawn trans-only)", wm & sheet)
stats("loss holes", holes & sheet)
stats("face-only black square (defect)", face_black)
stats("trans-only black square (defect)", trans_black)
stats("blank paper patch (520-560,380-420)", blank)
stats("all remaining plain paper", plain)

# Where is face dark/blue but trans differs? scan B-R threshold 0.08 over sheet
inklike_f = sheet & uf & (br_f >= 0.08)
inklike_t = sheet & ut & (br_t >= 0.08)
print("\nB-R>=0.08 in face pixels:", int(inklike_f.sum()), " of which also trans B-R>=0.08:",
      int((inklike_f & inklike_t).sum()), " face-only:", int((inklike_f & ~inklike_t).sum()))
print("B-R>=0.08 in trans pixels:", int(inklike_t.sum()), " trans-only:",
      int((inklike_t & ~inklike_f).sum()))

# unseen counts
print("\nface unseen px on sheet:", int((sheet & ~uf).sum()),
      " trans unseen px on sheet:", int((sheet & ~ut).sum()))
print("face_black useful in face?:", int((face_black & uf).sum()), "/", int(face_black.sum()))
print("trans_black useful in trans?:", int((trans_black & ut).sum()), "/", int(trans_black.sum()))

# check the face appearance at watermark zone vs nearby paper
near = (xx >= 586) & (xx < 666) & (yy >= 290) & (yy < 360)
stats("watermark neighborhood paper", near & ~wm & sheet)
# foxing spots quick stats
fox = np.zeros(uf.shape, np.uint8)
for cx, cy, rx, ry in ((569, 421, 12, 8), (625, 456, 17, 13), (595, 470, 6, 7)):
    cv.ellipse(fox, (cx, cy), (rx, ry), 12, 0, 360, 1, -1, cv.LINE_AA)
stats("foxing spots", fox.astype(bool) & sheet)
