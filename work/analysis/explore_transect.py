"""Build a local demo run via the bench code path and expose calibrated frames."""
import sys

sys.path.insert(0, "/opt/pointwise-env/site-packages")
sys.path.insert(0, "/workspace")

from paperbench.demo import create
from paperbench.store import Library
from paperbench.photos import decode, plane

library = Library("/workspace/work/library")
run = create(library)
print("run", run["id"], run["title"], run["width_mm"], run["height_mm"])
for photo in run["photos"]:
    print(photo["role"], photo["id"], photo["title"])

frames = {}
for photo in run["photos"]:
    frame = plane({**photo, "pixels": decode(library.blob(photo["blob"]).read_bytes())}, run)
    frames[photo["role"]] = frame
    print(photo["role"], frame.rgb.shape, frame.step, "useful", int(frame.useful.sum()))

# Transect endpoints in calibrated pixel coordinates (same as source: full-frame corners)
import numpy as np
import cv2 as cv

face = frames["face"]
trans = frames["transmitted"]
ends = np.array([[252, 196], [244, 346]], dtype=np.float32)
n = 151
positions = np.linspace(ends[0], ends[1], n, dtype=np.float32)
maps = positions.T
args = (maps[0][None], maps[1][None], cv.INTER_LINEAR)
cf = cv.remap(face.rgb, *args)[0]
ct = cv.remap(trans.rgb, *args)[0]
uf = cv.remap(face.useful.astype(np.float32), *args)[0] >= 0.999
ut = cv.remap(trans.useful.astype(np.float32), *args)[0] >= 0.999

print("\nidx  y     face_RGB            B-R     trans_RGB           B-R    uf ut")
for i in range(n):
    if i % 3 == 0 or (cf[i, 2] - cf[i, 0]) > 0.05 or (ct[i, 2] - ct[i, 0]) > 0.05:
        y = 196 + (346 - 196) * i / (n - 1)
        print(f"{i:3d} {y:6.1f}  "
              f"({cf[i,0]:.3f},{cf[i,1]:.3f},{cf[i,2]:.3f})  {cf[i,2]-cf[i,0]:+.3f}  "
              f"({ct[i,0]:.3f},{ct[i,1]:.3f},{ct[i,2]:.3f})  {ct[i,2]-ct[i,0]:+.3f}  "
              f"{int(uf[i])} {int(ut[i])}")
