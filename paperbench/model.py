"""Validate runs and calibrated annotations for paper conservation measurements."""
import math
import re

from image_lab.validation import Invalid


ROLES = {"face": "Reflected front light", "transmitted": "Transmitted backlight", "trial": "Trial repair or comparison"}
MARKS = {
    "sheet": "Sheet search area",
    "loss": "Loss inspection area",
    "tear": "Tear cross-section",
    "patch": "Trial repair footprint",
    "fiber": "Fiber sample patch",
    "ink": "Cross-ink transect",
    "stain": "Foxing search patch",
    "band": "Foxing border band",
    "centre": "Foxing centre field",
    "window": "Foxing sampling window",
    "inspect": "Shared inspection area",
    "register": "Corresponding paper landmarks",
    "watermark": "Watermark patch",
    "control": "Unmarked reference",
    "fold": "Fold axis",
    "protected": "Protected writing",
}
READINGS = {
    "sheet-edge": ("Sheet boundary", "Measure the largest visible sheet fragment and its edge."),
    "loss-map": ("Backlit loss map", "Separate bright openings, paper, and unseen material."),
    "tear-section": ("Tear section", "Measure opening intervals on a physical line across a tear."),
    "patch-coverage": ("Patch coverage", "Measure how one trial footprint covers loss, paper, and unseen pixels."),
    "fiber-balance": ("Fiber balance", "Read dominant visible fiber direction and coherence."),
    "ink-spread": ("Ink spread", "Measure ink intervals along a calibrated transect."),
    "foxing-map": ("Foxing map", "Locate local discoloration with a stronger yellow component."),
    "foxing-sample": ("Sampled foxing estimate",
                      "Adopt one set of non-overlapping sampling windows from coverage alone, then estimate sheet foxing."),
    "paper-tone": ("Paper tone", "Read Lab tone and variation in a marked sheet patch."),
    "landmark-offset": ("Paper landmark offsets", "Read residual shifts between ordered landmark pairs."),
    "paired-support": ("Paired paper visibility", "Separate joint visibility, single-photo visibility, and unseen material."),
    "transmission-change": ("Transmission change", "Read signed brightness differences in mutually visible paper."),
    "watermark-contrast": ("Watermark contrast", "Read transmitted contrast against nearby unmarked stock."),
    "fold-ligaments": ("Fold ligaments", "Find visible paper components connecting both banks of a marked fold."),
}
POINT_MARKS = {"register", "tear", "ink", "fold"}
IDENTITY = re.compile(r"^[0-9a-f]{24}$")


def identifier(value):
    if not isinstance(value, str) or not IDENTITY.fullmatch(value):
        raise Invalid("The selected item is unavailable.")
    return value


def locate(items, value, label):
    wanted = identifier(value)
    for item in items:
        if item["id"] == wanted:
            return item
    raise Invalid(f"{label} is no longer available.")


def obj(value):
    if not isinstance(value, dict):
        raise Invalid("Expected a set of named fields.")
    return value


def text(value, label, limit=100, optional=False):
    if not isinstance(value, str) or len(value.strip()) > limit or (not optional and not value.strip()):
        raise Invalid(f"Enter {label} using at most {limit} characters.")
    return value.strip()


def number(value, label, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise Invalid(f"Enter a valid {label}.")
    if not minimum <= value <= maximum:
        raise Invalid(f"{label.capitalize()} must be between {minimum:g} and {maximum:g}.")
    return value


def choice(value, values, label):
    if not isinstance(value, str) or value not in values:
        raise Invalid(f"Choose a valid {label}.")
    return value


def points(raw, minimum, maximum, label):
    if not isinstance(raw, list) or not minimum <= len(raw) <= maximum:
        raise Invalid(f"{label.capitalize()} needs between {minimum} and {maximum} ordered points.")
    result = []
    for point in raw:
        if not isinstance(point, list) or len(point) != 2:
            raise Invalid("Positions must be coordinate pairs.")
        result.append([number(point[0], "position", 0, 1), number(point[1], "position", 0, 1)])
    return result


def run_fields(raw):
    data = obj(raw)
    return {
        "title": text(data.get("title"), "a run title"),
        "stock": text(data.get("stock", ""), "the paper description", 100, optional=True),
        "width_mm": number(data.get("width_mm"), "sheet width (mm)", 20, 1500),
        "height_mm": number(data.get("height_mm"), "sheet height (mm)", 20, 1500),
    }


def mark_fields(raw):
    data = obj(raw)
    purpose = choice(data.get("purpose"), MARKS, "mark purpose")
    if purpose in {"tear", "ink", "fold"}:
        minimum, maximum = 2, 2
    elif purpose == "register":
        minimum, maximum = 2, 16
    else:
        minimum, maximum = 3, 32
    return {"title": text(data.get("title"), "a mark title"), "purpose": purpose,
            "points": points(data.get("points"), minimum, maximum, MARKS[purpose])}
