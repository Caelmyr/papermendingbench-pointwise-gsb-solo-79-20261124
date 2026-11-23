"""Create explicitly synthetic reflected, transmitted, and trial paper photographs."""
import math

import cv2 as cv
import numpy as np

from .photos import png
from .store import identity, stamp

WIDTH, HEIGHT = 888, 592
CORNERS = [[0, 0], [1, 0], [1, 1], [0, 1]]
LANDMARKS = [(147, 112), (723, 111), (708, 453), (164, 455)]
SHEET = [(102, 92), (189, 71), (380, 74), (772, 65), (786, 456), (732, 516), (492, 510),
         (315, 519), (140, 500), (102, 438), (115, 325)]
TEAR = [(380, 189), (401, 208), (416, 239), (430, 261), (414, 282), (433, 306), (422, 327),
        (455, 365), (438, 381), (413, 348), (398, 338), (404, 319), (383, 306), (401, 281),
        (404, 266), (385, 254), (391, 227), (371, 205)]
REGIONS = (
    ("Sheet edges", "sheet", (87, 55, 802, 532)),
    ("Backlit tear and holes", "loss", (340, 167, 506, 406)),
    ("Trial strip over full split", "patch", (365, 191, 471, 388)),
    ("Trial strip clear of writing", "patch", (386, 231, 477, 397)),
    ("Unprinted lower fibers", "fiber", (250, 383, 403, 447)),
    ("Upper fibers", "fiber", (214, 113, 368, 167)),
    ("Foxing around margin", "stain", (508, 389, 664, 482)),
    ("Fold and nearby sheet", "inspect", (235, 181, 603, 452)),
    ("Watermark shape", "watermark", (586, 288, 665, 364)),
    ("Stock near watermark", "control", (665, 300, 728, 369)),
    ("Protected stroke", "protected", (350, 215, 404, 239)),
)
LINES = (
    ("Trial tear width", "tear", [(338, 295), (470, 295)]),
    ("Across blue ink", "ink", [(252, 196), (244, 346)]),
    ("Center fold", "fold", [(350, 206), (515, 431)]),
)


def normalized(points):
    return [[x / (WIDTH - 1), y / (HEIGHT - 1)] for x, y in points]


def polygon(box):
    x0, y0, x1, y1 = box
    return normalized([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def photograph(role, seed):
    rng = np.random.default_rng(seed)
    y, x = np.indices((HEIGHT, WIDTH), dtype=np.float32)
    noise = rng.normal(0, 1.4, (HEIGHT, WIDTH))
    fiber = 4 * np.sin((y - .23 * x) * (2 * math.pi / 10))
    light = 2.8 * np.sin(x / 65) + 3 * np.sin(y / 88)
    paper = np.zeros((HEIGHT, WIDTH), np.uint8)
    cv.fillPoly(paper, [np.asarray(SHEET, np.int32)], 1)
    if role == "face":
        base = (210, 195, 150)
        background = (51, 51, 46)
    else:
        base = (156, 143, 103)
        background = (63, 67, 64)
    rgb = np.broadcast_to(np.array(background), (HEIGHT, WIDTH, 3)).copy().astype(np.float32)
    rgb[paper == 1] = np.array(base) + (noise + fiber + light)[paper == 1, None]
    rgba = cv.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv.COLOR_RGB2RGBA)
    cv.polylines(rgba, [np.asarray(SHEET, np.int32)], True, (157, 136, 94, 255), 2, cv.LINE_AA)
    gap_color = (234, 241, 234, 255) if role != "face" else (63, 59, 48, 255)
    cv.fillPoly(rgba, [np.asarray(TEAR, np.int32)], gap_color, cv.LINE_AA)
    for cx, cy, radius in ((475, 238, 10), (487, 332, 5), (364, 351, 6)):
        cv.circle(rgba, (cx, cy), radius, gap_color, -1, cv.LINE_AA)

    ink = (41, 66, 106, 255) if role == "face" else (51, 72, 112, 255)
    for start, end in ((186, 332), (186, 300), (202, 344), (180, 282), (193, 319)):
        py = 215 + (end % 86)
        cv.line(rgba, (start, py), (end, py + 7), ink, 4, cv.LINE_AA)
    cv.line(rgba, (344, 229), (400, 228), ink, 3, cv.LINE_AA)
    cv.line(rgba, (352, 209), (513, 429), (125, 107, 76, 255), 2, cv.LINE_AA)

    for cx, cy, rx, ry in ((569, 421, 12, 8), (625, 456, 17, 13), (595, 470, 6, 7)):
        cv.ellipse(rgba, (cx, cy), (rx, ry), 12, 0, 360, (151, 103, 58, 255), -1, cv.LINE_AA)

    if role != "face":
        cv.ellipse(rgba, (626, 325), (30, 25), 0, 0, 360, (123, 111, 85, 255), 10, cv.LINE_AA)
        cv.line(rgba, (623, 301), (626, 349), (115, 103, 80, 255), 4, cv.LINE_AA)
    if role == "transmitted":
        rgba[290:321, 425:442] = 0
    if role == "face":
        rgba[335:350, 461:478] = 0
    if role.startswith("trial"):
        wide = role == "trial"
        x0, y0, x1, y1 = (365, 191, 471, 388) if wide else (386, 231, 477, 397)
        box = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        direction = .42 if wide else -.38
        fill = 168 + (6 if wide else 4) * np.sin((x + direction * y) * (2 * math.pi / 9.2))
        rgba[box, :3] = np.clip(fill[box, None] + [0, -12, -44], 0, 255).astype(np.uint8)
        if wide:
            rgba[339:363, 438:455] = 0
    for ix, (cx, cy) in enumerate(LANDMARKS):
        dx = 0 if role == "face" else ix + 1
        cv.drawMarker(rgba, (cx + dx, cy), (235, 172, 82, 255), markerSize=13, thickness=2, line_type=cv.LINE_AA)
    cv.putText(rgba, "SYNTHETIC CONSERVATION SAMPLE", (106, 553), cv.FONT_HERSHEY_SIMPLEX,
               .65, (226, 211, 181, 255), 1, cv.LINE_AA)
    return rgba


def create(library):
    captures, marks = [], []
    for title, role, seed in (
        ("Original sheet in reflected light", "face", 91),
        ("Original sheet in transmitted light", "transmitted", 92),
        ("Wide trial strip in transmitted light", "trial", 93),
        ("Narrow trial strip in transmitted light", "trial_narrow", 94),
    ):
        capture = {"id": identity(), "title": title, "name": title, "role": "trial" if role.startswith("trial") else role,
                   "filename": role + "-paper.png", "blob": library.keep(png(photograph(role, seed))),
                   "corners": CORNERS, "synthetic": True, "uploaded": stamp()}
        captures.append(capture)
        entries = [(name, purpose, polygon(box)) for name, purpose, box in REGIONS]
        entries.extend((name, purpose, normalized(points)) for name, purpose, points in LINES)
        marker_points = normalized([(x + (idx + 1 if role != "face" else 0), y)
                                    for idx, (x, y) in enumerate(LANDMARKS)])
        entries.append(("Matching registration points", "register", marker_points))
        for title, purpose, points in entries:
            marks.append({"id": identity(), "title": title, "purpose": purpose, "points": points, "photo": capture["id"]})
    return library.new_run({"title": "Torn manuscript leaf", "stock": "Synthetic rag paper specimen",
                            "width_mm": 222, "height_mm": 148}, captures, marks)
