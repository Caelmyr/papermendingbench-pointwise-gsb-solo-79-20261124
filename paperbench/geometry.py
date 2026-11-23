"""Calibrated pixel sets and transects for the sheet workbench."""
from dataclasses import dataclass

import cv2 as cv
import numpy as np

from image_lab.validation import Invalid

from .photos import coordinates, image, region


@dataclass(frozen=True)
class Segment:
    first_mm: float
    last_mm: float
    state: str

    def row(self):
        return {
            "state": self.state,
            "start_mm": round(self.first_mm, 3),
            "end_mm": round(self.last_mm, 3),
            "length_mm": round(self.last_mm - self.first_mm, 3),
        }


@dataclass(frozen=True)
class Transect:
    positions: np.ndarray
    coordinate_mm: np.ndarray
    luma: np.ndarray
    known: np.ndarray

    def classify(self, accepted, state):
        categories = np.where(self.known, np.where(accepted, state, "blocked"), "unseen")
        changes = np.flatnonzero(categories[1:] != categories[:-1]) + 1
        transition = [(self.coordinate_mm[i-1] + self.coordinate_mm[i]) / 2 for i in changes]
        boundaries = [0, *transition, self.coordinate_mm[-1]]
        return [Segment(float(first), float(last), categories[index])
                for first, last, index in zip(boundaries[:-1], boundaries[1:], [0, *changes])]


def area(frame, mark, minimum=32):
    domain = region(frame, mark)
    visible = domain & frame.useful
    if visible.sum() < minimum:
        raise Invalid("Mark a larger visible part of the sheet.")
    return domain, visible


def line(frame, mark):
    ends = coordinates(mark["points"], frame.rgb.shape)
    length = float(np.linalg.norm((ends[1] - ends[0]) * frame.step))
    if length < 1:
        raise Invalid("Mark a line at least 1 mm long.")
    count = min(1600, max(16, round(length * 5) + 1))
    positions = np.linspace(ends[0], ends[1], count, dtype=np.float32)
    maps = positions.T
    args = (maps[0][None], maps[1][None], cv.INTER_LINEAR)
    sampled = cv.remap(frame.luma.astype(np.float32), *args)[0]
    known = cv.remap(frame.useful.astype(np.float32), *args)[0] >= .999
    return Transect(positions, np.linspace(0, length, count), sampled, known)


def paint(shown, mask, color, opacity=.67):
    shown[mask] = shown[mask] * (1 - opacity) + np.asarray(color) * opacity


def mark_line(frame, transect, selected):
    shown = (255 * image(frame)).astype(np.uint8)
    for point, known, passed in zip(transect.positions, transect.known, selected):
        color = (58, 213, 173) if passed else (247, 141, 94)
        if not known:
            color = (175, 162, 227)
        cv.circle(shown, tuple(np.rint(point).astype(int)), 2, color, -1, cv.LINE_AA)
    return shown


def components(frame, mask, minimum_mm2=0):
    count, labels, stats, centers = cv.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    unit = frame.step[0] * frame.step[1]
    indices = [i for i in range(1, count) if stats[i, cv.CC_STAT_AREA] * unit >= minimum_mm2]
    indices.sort(key=lambda i: (-int(stats[i, cv.CC_STAT_AREA]), i))
    return labels, [
        {"area_mm2": round(float(stats[i, cv.CC_STAT_AREA] * unit), 3),
         "center_x_mm": round(float(centers[i][0] * frame.step[0]), 3),
         "center_y_mm": round(float(centers[i][1] * frame.step[1]), 3),
         "label": i}
        for i in indices
    ]


def footprint_rows(frame, mask):
    label, pieces = components(frame, mask, .05)
    rows = [{"component": i, **{key: value for key, value in part.items() if key != "label"}}
            for i, part in enumerate(pieces, 1)]
    return label, rows
