"""Image measurements for paper conservation from calibrated face and transmitted light."""
import math

import cv2 as cv
import numpy as np

from image_lab.validation import Invalid

from .geometry import area, components, footprint_rows, line, mark_line, paint
from .model import number
from .photos import coordinates, image, region


def square_mm(frame, mask):
    return float(mask.sum() * frame.step[0] * frame.step[1])


def percent(mask, total):
    return round(100 * mask.sum() / max(1, total.sum()), 3)


def sheet_edge(frame, marks, opts):
    _, seen = area(frame, marks["region"])
    cutoff = number(opts.get("luma", .47), "sheet luminance", .05, .95)
    labels, fragments = components(frame, seen & (frame.luma >= cutoff), .2)
    if not fragments:
        raise Invalid("No sheet fragment meets the threshold in this region.")
    main = labels == fragments[0]["label"]
    contours, _ = cv.findContours(main.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv.contourArea)
    world = (contour * np.asarray(frame.step)).astype(np.float32)
    hull = cv.convexHull(world)
    shown = image(frame)
    paint(shown, main, (.28, .74, .63), .33)
    cv.polylines(shown, [contour], True, (.99, .71, .31), 2)
    return {"largest_fragment_mm2": round(square_mm(frame, main), 3),
            "hull_mm2": round(cv.contourArea(hull), 3),
            "edge_perimeter_mm": round(cv.arcLength(world, True), 3),
            "visible_fragments": len(fragments)}, fragments[:50], shown


def loss_map(frame, marks, opts):
    domain, seen = area(frame, marks["region"])
    cutoff = number(opts.get("transmission", .78), "loss transmission", .1, .95)
    minimum = number(opts.get("minimum_mm2", .3), "minimum loss area", .01, 100)
    labels, losses = components(frame, seen & (frame.luma >= cutoff), minimum)
    kept = np.isin(labels, [entry["label"] for entry in losses]) & seen
    shown = image(frame)
    paint(shown, kept, (.95, .54, .24))
    paint(shown, domain & ~seen, (.65, .54, .79), .60)
    return {"loss_count": len(losses), "loss_mm2": round(square_mm(frame, kept), 3),
            "unseen_mm2": round(square_mm(frame, domain & ~seen), 3)}, losses[:60], shown


def tear_section(frame, marks, opts):
    probe = line(frame, marks["line"])
    cutoff = number(opts.get("transmission", .78), "tear transmission", .1, .95)
    bright = probe.luma >= cutoff
    sections = probe.classify(bright, "open")
    rows = [{**item.row(), "state": "paper" if item.state == "blocked" else item.state} for item in sections]
    openings = [item.last_mm - item.first_mm for item in sections if item.state == "open"]
    return {"open_intervals": len(openings), "widest_opening_mm": round(max(openings, default=0), 3),
            "known_length_pct": percent(probe.known, np.ones_like(probe.known))}, rows, mark_line(frame, probe, bright)


def patch_coverage(frame, marks, opts):
    domain, seen = area(frame, marks["losses"])
    patch = region(frame, marks["patch"]) & domain
    cutoff = number(opts.get("transmission", .78), "loss transmission", .1, .95)
    loss = seen & (frame.luma >= cutoff)
    groups = (
        ("Patch over visible loss", patch & loss, (.36, .78, .56)),
        ("Patch on visible paper", patch & seen & ~loss, (.93, .68, .27)),
        ("Patch on unseen area", patch & ~seen, (.65, .54, .8)),
        ("Visible loss left open", loss & ~patch, (.92, .46, .34)),
    )
    shown = image(frame)
    rows = []
    for title, pixels, color in groups:
        paint(shown, pixels, color)
        rows.append({"zone": title, "area_mm2": round(square_mm(frame, pixels), 3)})
    return {"patch_mm2": round(square_mm(frame, patch), 3),
            "visible_loss_covered_pct": percent(patch & loss, loss)}, rows, shown


def gradient_tensor(frame, usable):
    kernel = np.ones((3, 3), np.uint8)
    interior = cv.erode(usable.astype(np.uint8), kernel, borderType=cv.BORDER_CONSTANT, borderValue=0).astype(bool)
    if interior.sum() < 40:
        raise Invalid("Mark a fiber patch with a visible interior.")
    pixels = frame.luma.astype(np.float32)
    gx = cv.Sobel(pixels, cv.CV_32F, 1, 0)[interior] / frame.step[0]
    gy = cv.Sobel(pixels, cv.CV_32F, 0, 1)[interior] / frame.step[1]
    tensor = np.array([[np.mean(gx * gx), np.mean(gx * gy)], [np.mean(gx * gy), np.mean(gy * gy)]])
    magnitude, axes = np.linalg.eigh(tensor)
    strength = magnitude.sum()
    if strength <= 1e-9:
        raise Invalid("The visible fiber patch does not have a measurable direction.")
    vector = axes[:, 0]
    angle = math.degrees(math.atan2(vector[1], vector[0])) % 180
    return angle, float((magnitude[1] - magnitude[0]) / strength), interior, vector


def fiber_balance(frame, marks, _opts):
    _, seen = area(frame, marks["region"], 90)
    angle, coherence, used, vector = gradient_tensor(frame, seen)
    shown = (255 * image(frame)).astype(np.uint8)
    yy, xx = np.nonzero(used)
    center = np.array([xx.mean(), yy.mean()])
    pitch = vector / np.asarray(frame.step)
    extent = min(np.ptp(xx) * frame.step[0], np.ptp(yy) * frame.step[1]) * .3
    a = tuple(np.rint(center - extent * pitch).astype(int))
    b = tuple(np.rint(center + extent * pitch).astype(int))
    cv.line(shown, a, b, (25, 111, 111), 3, cv.LINE_AA)
    return {"fiber_angle_deg": round(angle, 3), "directional_coherence": round(coherence, 4),
            "fiber_samples": int(used.sum())}, [{"statistic": "Interior area", "mm2": round(square_mm(frame, used), 3)}], shown


def ink_spread(frame, marks, opts):
    probe = line(frame, marks["line"])
    limit = number(opts.get("chroma", .08), "ink chroma distance", .01, .8)
    color = cv.remap(frame.rgb, probe.positions[:, 0][None], probe.positions[:, 1][None], cv.INTER_LINEAR)[0]
    # Warm paper has a stronger red channel; the blue ink plume changes that relationship.
    ink = color[:, 2] - color[:, 0] >= limit
    regions = probe.classify(ink, "ink")
    runs = [segment for segment in regions if segment.state == "ink"]
    lengths = [segment.last_mm - segment.first_mm for segment in runs]
    rows = [{**segment.row(), "state": "paper" if segment.state == "blocked" else segment.state} for segment in regions]
    return {"ink_intervals": len(runs), "widest_ink_mm": round(max(lengths, default=0), 3)}, rows, mark_line(frame, probe, ink)


def foxing_map(frame, marks, opts):
    _, seen = area(frame, marks["region"])
    threshold = number(opts.get("yellow", .035), "local yellow excess", .01, .5)
    stain = frame.rgb[:, :, 0] - frame.rgb[:, :, 2]
    normalized = np.where(frame.useful, stain, np.median(stain[seen])).astype(np.float32)
    local = cv.GaussianBlur(normalized, (0, 0), 9)
    patches = seen & (normalized - local >= threshold)
    _, rows = footprint_rows(frame, patches)
    shown = image(frame)
    paint(shown, patches, (.85, .33, .23), .50)
    return {"foxing_clusters": len(rows), "affected_mm2": round(square_mm(frame, patches), 3)}, rows[:50], shown


def paper_tone(frame, marks, _opts):
    _, seen = area(frame, marks["region"])
    lab = cv.cvtColor(frame.rgb[seen].reshape(-1, 1, 3), cv.COLOR_RGB2LAB).reshape(-1, 3)
    low, median, high = np.percentile(lab, [10, 50, 90], axis=0)
    shown = image(frame)
    paint(shown, seen, (.38, .71, .85), .20)
    rows = [{"channel": name, "p10": round(float(low[i]), 4), "median": round(float(median[i]), 4),
             "p90": round(float(high[i]), 4)} for i, name in enumerate(("L*", "a*", "b*"))]
    return {"observed_pixels": int(seen.sum()), "median_lab": median.round(4).tolist()}, rows, shown


def landmark_offset(frame, other, marks, _opts):
    left = coordinates(marks["reference"]["points"], frame.rgb.shape) * frame.step
    right = coordinates(marks["comparison"]["points"], other.rgb.shape) * other.step
    if left.shape != right.shape:
        raise Invalid("Mark the same number of landmarks in the same order.")
    shifts = left - right
    median = np.median(shifts, axis=0)
    deviation = np.linalg.norm(shifts - median, axis=1)
    shown = (255 * image(frame)).astype(np.uint8)
    for ref, next_ in zip(left, right):
        start = tuple((next_ / frame.step).round().astype(int))
        end = tuple((ref / frame.step).round().astype(int))
        cv.circle(shown, start, 4, (154, 133, 201), 2)
        if start != end:
            cv.arrowedLine(shown, start, end, (238, 115, 59), 2, cv.LINE_AA, tipLength=.45)
    rows = [{"landmark": i, "shift_x_mm": round(float(dx), 3), "shift_y_mm": round(float(dy), 3),
             "residual_mm": round(float(delta), 3)} for i, ((dx, dy), delta) in enumerate(zip(shifts, deviation), 1)]
    return {"median_shift_mm": median.round(3).tolist(), "maximum_residual_mm": round(float(deviation.max()), 3)}, rows, shown


def paired_support(frame, other, marks, _opts):
    domain, seen = area(frame, marks["region"])
    other_seen = other.useful & domain
    states = (seen & other_seen, seen & ~other_seen, other_seen & ~seen, domain & ~seen & ~other_seen)
    shown = image(frame)
    rows = []
    for mask, title, color in zip(states, ("Both photos", "First only", "Second only", "Unseen in both"),
                                  ((.39, .8, .63), (.91, .68, .27), (.36, .65, .9), (.67, .53, .79))):
        paint(shown, mask, color)
        rows.append({"state": title, "area_mm2": round(square_mm(frame, mask), 3), "window_pct": percent(mask, domain)})
    return {"shared_pct": rows[0]["window_pct"], "window_mm2": round(square_mm(frame, domain), 3)}, rows, shown


def transmission_difference(frame, other, marks, opts):
    _, seen = area(frame, marks["region"])
    both = seen & other.useful
    if both.sum() < 32:
        raise Invalid("The transmission window needs sufficient common evidence.")
    bound = number(opts.get("difference", .12), "transmission difference", .01, .9)
    delta = other.luma - frame.luma
    partitions = (("Brighter in comparison", both & (delta > bound), (.92, .62, .24)),
                  ("Darker in comparison", both & (delta < -bound), (.29, .55, .86)),
                  ("Within threshold", both & (np.abs(delta) <= bound), (.37, .72, .57)))
    shown = image(frame)
    rows = []
    for title, pixels, color in partitions:
        paint(shown, pixels, color)
        rows.append({"state": title, "area_mm2": round(square_mm(frame, pixels), 3), "paired_pct": percent(pixels, both)})
    return {"median_change": round(float(np.median(delta[both])), 4),
            "paired_samples": int(both.sum())}, rows, shown


def watermark_contrast(frame, marks, _opts):
    _, watermark = area(frame, marks["watermark"])
    _, reference = area(frame, marks["reference"])
    evident = np.histogram(frame.luma[watermark], bins=12, range=(0, 1))[0]
    baseline = np.histogram(frame.luma[reference], bins=12, range=(0, 1))[0]
    wt = float(np.median(frame.luma[watermark]))
    ref = float(np.median(frame.luma[reference]))
    shown = image(frame)
    paint(shown, watermark, (.44, .63, .90), .36)
    paint(shown, reference, (.86, .60, .34), .33)
    rows = [{"from_luma": round(index / 12, 4), "watermark_pixels": int(a), "background_pixels": int(b)}
            for index, (a, b) in enumerate(zip(evident, baseline))]
    return {"watermark_median": round(wt, 4), "reference_median": round(ref, 4),
            "contrast_difference": round(wt - ref, 4)}, rows, shown


def fold_ligaments(frame, marks, opts):
    domain, seen = area(frame, marks["region"])
    cutoff = number(opts.get("transmission", .76), "open fold transmission", .1, .95)
    bridge_depth = number(opts.get("depth_mm", 3), "fold evidence depth", .5, 25)
    ends = coordinates(marks["fold"]["points"], frame.rgb.shape) * frame.step
    a, b = ends
    tangent = b - a
    length = float(np.linalg.norm(tangent))
    if length < 1:
        raise Invalid("Mark a fold at least 1 mm long.")
    tangent /= length
    normal = np.array([-tangent[1], tangent[0]])
    yy, xx = np.indices(domain.shape)
    dx, dy = xx * frame.step[0] - a[0], yy * frame.step[1] - a[1]
    along = dx * tangent[0] + dy * tangent[1]
    across = dx * normal[0] + dy * normal[1]
    strip = domain & (along >= 0) & (along <= length) & (np.abs(across) <= bridge_depth)
    solid = seen & strip & (frame.luma < cutoff)
    labels, pieces = components(frame, solid, .1)
    top = set(np.unique(labels[strip & (across <= -.8 * bridge_depth)])) - {0}
    bottom = set(np.unique(labels[strip & (across >= .8 * bridge_depth)])) - {0}
    bridges = top & bottom
    shown = image(frame)
    paint(shown, np.isin(labels, list(bridges)), (.36, .75, .58))
    paint(shown, strip & ~seen, (.65, .54, .79), .60)
    rows = [{**entry, "touches_two_banks": entry["label"] in bridges} for entry in pieces]
    return {"connected_ligaments": len(bridges), "unseen_band_mm2": round(square_mm(frame, strip & ~seen), 3)}, rows[:60], shown
