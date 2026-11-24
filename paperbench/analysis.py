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


ALIGNMENT_MODE_ALIASES = {
    "fold-alignment": "fold-alignment",
    "alignment": "fold-alignment",
    "registration": "fold-alignment",
    "fold-registration": "fold-alignment",
    "fold-register": "fold-alignment",
    "fold-align": "fold-alignment",
    "fold-landmark-offset": "fold-alignment",
    "fold-shift": "fold-alignment",
    "fold-shifts": "fold-alignment",
    "split-landmarks": "fold-alignment",
    "fold-side-alignment": "fold-alignment",
    "fold-side-registration": "fold-alignment",
    "fold-aligned-transmission": "fold-alignment",
    "fold-aligned-transmission-change": "fold-alignment",
    "side-aligned-transmission": "fold-alignment",
    "side-alignment": "fold-alignment",
    "side-registration": "fold-alignment",
    "bank-alignment": "fold-alignment",
    "bank-registration": "fold-alignment",
    "split-alignment": "fold-alignment",
    "split-registration": "fold-alignment",
    "split-landmark-offset": "fold-alignment",
    "split-landmark-alignment": "fold-alignment",
    "split-registration-transmission": "fold-alignment",
    "aligned-transmission": "fold-alignment",
    "aligned-transmission-change": "fold-alignment",
    "registration-transmission": "fold-alignment",
    "fold-transmission": "fold-alignment",
    "fold-transmission-change": "fold-alignment",
    "fold-side-transmission": "fold-alignment",
    "split-transmission": "fold-alignment",
    "split-transmission-change": "fold-alignment",
    "transmission-registration": "fold-alignment",
}


def _option_number(opts, names, default, minimum=.0, maximum=10_000_000):
    if not isinstance(opts, dict):
        return default
    for name in names:
        if name in opts and opts[name] is not None:
            return number(opts[name], name.replace("_", " "), minimum, maximum)
    return default


def _raw_option(opts, names):
    if not isinstance(opts, dict):
        return None
    for name in names:
        if name in opts and opts[name] is not None:
            return opts[name]
    return None


def _common_limit_mm2(opts, frame, domain):
    pixel_mm2 = float(frame.step[0] * frame.step[1])
    mm2 = _raw_option(opts, (
        "min_common_mm2", "minimum_common_mm2", "min_shared_mm2", "minimum_shared_mm2",
        "min_overlap_mm2", "minimum_overlap_mm2", "min_joint_visibility_mm2",
        "minimum_joint_visibility_mm2", "min_common_visibility_mm2",
        "minimum_common_visibility_mm2", "min_common_area_mm2", "min_shared_area_mm2",
        "min_paired_area_mm2", "min_common_area", "min_shared_area", "min_overlap_area",
        "min_visibility_mm2", "minimum_visibility_mm2",
    ))
    if mm2 is not None:
        return number(mm2, "minimum common evidence", .000001, 2_250_000)
    pixels = _raw_option(opts, (
        "min_common_pixels", "minimum_common_pixels", "min_shared_pixels", "minimum_shared_pixels",
        "min_overlap_pixels", "minimum_overlap_pixels", "min_common_samples", "min_shared_samples",
        "min_overlap_samples", "minimum_common_samples", "minimum_shared_samples",
    ))
    if pixels is not None:
        return number(pixels, "minimum common evidence", 1, 100_000_000) * pixel_mm2
    pct = _raw_option(opts, (
        "min_common_pct", "minimum_common_pct", "min_shared_pct", "minimum_shared_pct",
        "min_overlap_pct", "minimum_overlap_pct", "min_common_percent", "min_shared_percent",
    ))
    if pct is not None:
        return number(pct, "minimum common percentage", 0, 100) / 100 * square_mm(frame, domain)
    fraction = _raw_option(opts, ("min_common_fraction", "min_shared_fraction", "min_overlap_fraction"))
    if fraction is not None:
        return number(fraction, "minimum common fraction", 0, 1) * square_mm(frame, domain)
    return 32 * pixel_mm2


def _as_mark(value):
    if isinstance(value, dict):
        if isinstance(value.get("points"), list):
            return value
        for key in ("mark", "reference", "source", "comparison", "other"):
            found = _as_mark(value.get(key))
            if found is not None:
                return found
    if isinstance(value, list):
        if len(value) >= 2 and all(isinstance(point, (list, tuple)) and len(point) == 2 for point in value):
            return {"points": value}
        for item in value:
            found = _as_mark(item)
            if found is not None:
                return found
    return None


def _named_mark(marks, *names):
    if not isinstance(marks, dict):
        return None
    for name in names:
        found = _as_mark(marks.get(name))
        if found is not None:
            return found
    return None


def _alignment_requested(opts, marks=None):
    if isinstance(marks, dict):
        has_fold = _named_mark(marks, "fold", "fold_line", "fold_axis", "axis", "crease") is not None
        has_pair = (_named_mark(marks, "reference", "source", "register") is not None
                    and _named_mark(marks, "comparison", "other") is not None)
        if has_fold and has_pair:
            return True
    if not isinstance(opts, dict):
        return False
    if opts.get("aligned") or opts.get("use_fold_alignment") or opts.get("fold_alignment"):
        return True
    if opts.get("split") or opts.get("split_by_fold") or opts.get("by_fold") or opts.get("separate_sides"):
        return True
    if opts.get("compare_plans") or opts.get("try_both") or opts.get("whole_and_split"):
        return True
    if opts.get("alignment") not in (None, False, "none", "off", "original", "same"):
        return True
    if opts.get("placement") not in (None, False, "original", "same"):
        return True
    if opts.get("plan") in ALIGNMENT_MODE_ALIASES or opts.get("selected_plan"):
        return True
    limit_keys = (
        "max_residual_mm", "residual_limit_mm", "residual_cap_mm", "tolerance_mm",
        "min_common_mm2", "min_shared_mm2", "min_common_pixels", "min_shared_pixels",
    )
    return any(key in opts for key in limit_keys)


def _fold_line(mark, shape, step):
    points = coordinates(mark["points"], shape)
    if len(points) < 2:
        raise Invalid("Mark a two-point fold axis.")
    first, second = np.asarray(points[0], float), np.asarray(points[1], float)
    if np.linalg.norm((second - first) * np.asarray(step)) < 1:
        raise Invalid("Mark a fold at least 1 mm long.")
    tangent = second - first
    tangent /= np.linalg.norm(tangent)
    normal = np.array([-tangent[1], tangent[0]])
    return first, second, tangent, normal


def _across(points, first, normal):
    points = np.asarray(points, float)
    return ((points[..., 0] - first[0]) * normal[0]
            + (points[..., 1] - first[1]) * normal[1])


def _extract_landmark_groups(marks, frame, other):
    reference = _named_mark(
        marks, "reference", "source", "reference_mark", "source_mark", "ref", "register", "landmarks")
    comparison = _named_mark(
        marks, "comparison", "other", "comparison_mark", "other_mark", "target",
        "comparison_register", "other_register", "comparison_landmarks")
    if reference is not None and comparison is not None:
        ref_px = coordinates(reference["points"], frame.rgb.shape)
        other_px = coordinates(comparison["points"], other.rgb.shape)
        if ref_px.shape != other_px.shape or len(ref_px) < 1:
            raise Invalid("Mark the same number of landmarks in the same order.")
        return ref_px, other_px, np.full(len(ref_px), -1, dtype=int)

    families = (
        (("left_reference", "reference_left", "left_register", "register_left", "left_landmarks",
          "left", "a_reference", "reference_a", "side_a", "group_a", "bank_a", "first_reference", "one"),
         ("left_comparison", "comparison_left", "left_other", "other_left", "left_target",
          "a_comparison", "comparison_a", "other_a", "side_a_other", "group_a_other",
          "first_comparison", "left_other_register", "one_other")),
        (("right_reference", "reference_right", "right_register", "register_right", "right_landmarks",
          "right", "b_reference", "reference_b", "side_b", "group_b", "bank_b", "second_reference", "two"),
         ("right_comparison", "comparison_right", "right_other", "other_right", "right_target",
          "b_comparison", "comparison_b", "other_b", "side_b_other", "group_b_other",
          "second_comparison", "right_other_register", "two_other")),
    )
    ref_points, other_points, group_ids = [], [], []
    for group_id, (ref_names, other_names) in enumerate(families):
        ref_mark = _named_mark(marks, *ref_names)
        other_mark = _named_mark(marks, *other_names)
        if ref_mark is None and other_mark is None:
            nested = _named_mark(marks, "left" if group_id == 0 else "right",
                                 "side_a", "side_b", "group_a", "group_b")
            if nested is not None:
                ref_mark = _named_mark(nested, "reference", "source", "register", "mark")
                other_mark = _named_mark(nested, "comparison", "other", "target")
        if ref_mark is None or other_mark is None:
            continue
        rp = coordinates(ref_mark["points"], frame.rgb.shape)
        op = coordinates(other_mark["points"], other.rgb.shape)
        if rp.shape != op.shape or len(rp) < 1:
            raise Invalid("Each fold side needs matching, ordered paper landmarks.")
        ref_points.append(rp)
        other_points.append(op)
        group_ids.append(np.full(len(rp), group_id, dtype=int))
    if not ref_points:
        raise Invalid("Select matching paper landmarks, either one ordered register pair or both fold-side groups.")
    return np.concatenate(ref_points), np.concatenate(other_points), np.concatenate(group_ids)


def fold_alignment(frame, other, marks, opts):
    """Compare one global translation with two fold-bank translations.

    The returned overlay uses the qualifying plan selected from deduplicated
    common evidence and landmark accuracy.  Split-bank overlap is retained as
    its own conflict mask rather than being counted twice.
    """
    marks = marks if isinstance(marks, dict) else {}
    opts = opts if isinstance(opts, dict) else {}

    region_mark = _named_mark(marks, "region", "window", "inspect", "area", "transmission_window")
    if region_mark is None:
        region_mark = _as_mark(_raw_option(opts, ("window", "region", "window_mark", "window_points")))
    if region_mark is None:
        domain = np.ones(frame.rgb.shape[:2], dtype=bool)
    else:
        domain = region(frame, region_mark)
    if not domain.any():
        raise Invalid("Mark a transmission window with an interior.")

    fold_mark = _named_mark(marks, "fold", "fold_line", "fold_axis", "axis", "crease")
    if fold_mark is None:
        fold_mark = _as_mark(_raw_option(opts, ("fold", "fold_line", "fold_axis", "fold_points")))
    if fold_mark is None:
        raise Invalid("Select the marked fold axis before comparing fold-side translations.")
    first, second, fold_tangent, fold_normal = _fold_line(fold_mark, frame.rgb.shape, frame.step)
    other_fold_mark = _named_mark(marks, "other_fold", "comparison_fold", "fold_comparison")
    other_fold_source = other_fold_mark or fold_mark
    other_first, _, _, other_normal = _fold_line(other_fold_source, other.rgb.shape, other.step)
    if float(np.dot(other_normal, fold_normal)) < 0:
        other_normal = -other_normal

    ref_px, other_px, explicit_group = _extract_landmark_groups(marks, frame, other)
    ref_mm = ref_px * np.asarray(frame.step)
    other_mm = other_px * np.asarray(other.step)
    shifts_mm = ref_mm - other_mm

    target_y, target_x = np.indices(frame.rgb.shape[:2], dtype=np.float32)
    target_across = _across(np.dstack([target_x, target_y]), first, fold_normal)
    target_side = np.where(target_across >= 0, 0, 1)

    if np.all(explicit_group >= 0):
        side_id = np.zeros(len(ref_px), dtype=int)
        for group in np.unique(explicit_group):
            indices = np.flatnonzero(explicit_group == group)
            group_across = float(np.mean(_across(ref_px[indices], first, fold_normal)))
            side_id[indices] = 0 if group_across >= 0 else 1
    else:
        landmark_across = _across(ref_px, first, fold_normal)
        side_id = np.where(landmark_across >= 0, 0, 1)
        split_index = _raw_option(opts, (
            "split_index", "split_at", "landmark_split_index", "left_count", "first_group_count"))
        right_count = _raw_option(opts, ("right_count", "second_group_count"))
        try:
            if right_count is not None and split_index is None:
                right_number = int(right_count)
                if 0 < right_number < len(ref_px):
                    split_index = len(ref_px) - right_number
            if split_index is not None:
                split_index = int(split_index)
                if 0 < split_index < len(ref_px):
                    side_id = np.where(np.arange(len(ref_px)) < split_index, 0, 1)
        except (TypeError, ValueError):
            pass

    bank_names = {}
    side_keys = {0: "side_a", 1: "side_b"}
    horizontal_fold = abs(fold_tangent[0]) >= abs(fold_tangent[1])
    for side in (0, 1):
        indices = np.flatnonzero(side_id == side)
        if len(indices):
            center = ref_px[indices].mean(axis=0)
        else:
            offset = fold_normal * (35 if side == 0 else -35)
            center = first + offset
        if horizontal_fold:
            bank_names[side] = "top" if center[1] < (first[1] + second[1]) / 2 else "bottom"
        else:
            bank_names[side] = "left" if center[0] < (first[0] + second[0]) / 2 else "right"

    estimator_name = str(_raw_option(opts, (
        "estimator", "estimate", "translation_method", "fit", "method")) or "median").lower()
    center_estimate = np.mean if any(word in estimator_name for word in ("mean", "average", "ols")) else np.median
    whole_t = center_estimate(shifts_mm, axis=0)
    whole_residual = np.linalg.norm(shifts_mm - whole_t, axis=1)
    split_translations = [None, None]
    split_t_per_point = np.zeros_like(shifts_mm)
    split_residual = np.full(len(ref_px), np.inf, dtype=float)
    for side in (0, 1):
        indices = np.flatnonzero(side_id == side)
        if len(indices):
            translation = center_estimate(shifts_mm[indices], axis=0)
            split_translations[side] = translation
            split_t_per_point[indices] = translation
            split_residual[indices] = np.linalg.norm(shifts_mm[indices] - translation, axis=1)

    cap = _option_number(opts, (
        "max_residual_mm", "maximum_residual_mm", "residual_limit_mm", "residual_cap_mm",
        "tolerance_mm", "tolerance", "landmark_tolerance_mm", "max_residual", "maximum_residual",
        "residual_limit", "residual_tolerance", "limit", "max_error_mm", "max_error",
        "error_limit_mm", "landmark_cap_mm", "point_tolerance_mm",
    ), 1.0, .000001, 10_000)
    minimum_common = _common_limit_mm2(opts, frame, domain)
    difference_bound = _option_number(
        opts, ("difference", "difference_threshold", "transmission_difference", "delta_threshold"),
        .12, .01, .9)

    source_seen = domain & frame.useful
    height, width = frame.rgb.shape[:2]

    def affine(translation_mm):
        return np.array([
            [other.step[0] / frame.step[0], 0.0, translation_mm[0] / frame.step[0]],
            [0.0, other.step[1] / frame.step[1], translation_mm[1] / frame.step[1]],
        ], dtype=np.float32)

    def warp_bool(mask, matrix):
        return cv.warpAffine(mask.astype(np.uint8), matrix, (width, height),
                             flags=cv.INTER_NEAREST, borderValue=0).astype(bool)

    def warp_luma(matrix):
        return cv.warpAffine(other.luma.astype(np.float32), matrix, (width, height),
                             flags=cv.INTER_LINEAR, borderValue=0)

    def plan_result(plan):
        order = 0 if plan == "whole" else 1
        if plan == "whole":
            matrix = affine(whole_t)
            placed = warp_bool(other.useful, matrix)
            luma = warp_luma(matrix)
            coverage = placed
            common0 = source_seen & placed & (target_side == 0)
            common1 = source_seen & placed & (target_side == 1)
            overlap = np.zeros_like(domain)
            measure0, measure1 = common0, common1
            lumas = [luma, luma]
            translations = [whole_t, whole_t]
            max_residual = float(whole_residual.max())
            rms = float(np.sqrt(np.mean(whole_residual ** 2)))
            mean_residual = float(whole_residual.mean())
            residual = whole_residual
        else:
            if any(translation is None for translation in split_translations):
                missing_sides = [bank_names[s] for s in (0, 1) if split_translations[s] is None]
                info = {
                    "plan": "split", "label": "Separate fold-side translations", "eligible": False,
                    "qualified": False, "fits": False,
                    "translation_mm": None, "translations_mm": None,
                    "max_residual_mm": None, "rms_residual_mm": None, "mean_residual_mm": None,
                    "common_mm2": 0.0, "shared_mm2": 0.0, "common_pixels": 0, "shared_pixels": 0,
                    "usable_common_mm2": 0.0, "paired_samples": 0, "gross_common_mm2": 0.0,
                    "conflict_mm2": 0.0, "fold_conflict_mm2": 0.0, "conflict_pixels": 0,
                    "minimum_common_mm2": round(minimum_common, 3), "residual_cap_mm": cap,
                    "reasons": [f"No landmark group was found on the {', '.join(missing_sides)} fold side."],
                }
                return info, None, {"_max": math.inf, "_rms": math.inf, "_common": -1, "_usable": -1, "_order": order}

            by, bx = np.indices(other.rgb.shape[:2], dtype=np.float32)
            bank_across = _across(np.dstack([bx, by]), other_first, other_normal)
            bank0 = bank_across >= 0
            raw0 = other.useful & bank0
            raw1 = other.useful & ~bank0
            placed0 = warp_bool(raw0, affine(split_translations[0]))
            placed1 = warp_bool(raw1, affine(split_translations[1]))
            overlap = placed0 & placed1
            coverage = placed0 | placed1
            common0 = source_seen & placed0
            common1 = source_seen & placed1
            overlap_seen = common0 & common1
            measure0 = common0 & ~overlap_seen
            measure1 = common1 & ~overlap_seen
            overlap = overlap_seen
            lumas = [warp_luma(affine(split_translations[0])),
                     warp_luma(affine(split_translations[1]))]
            translations = split_translations
            max_residual = float(split_residual.max())
            rms = float(np.sqrt(np.mean(split_residual ** 2)))
            mean_residual = float(split_residual.mean())
            residual = split_residual

        gross_pixels = int(common0.sum() + common1.sum())
        common_mask = source_seen & coverage
        common_pixels = int(common_mask.sum())
        conflict_pixels = int(overlap.sum())
        usable_mask = measure0 | measure1
        usable_pixels = int(usable_mask.sum())
        common_area = square_mm(frame, common_mask)
        usable_area = square_mm(frame, usable_mask)
        conflict_area = square_mm(frame, overlap)
        reasons = []
        if not math.isfinite(max_residual) or max_residual > cap + 1e-9:
            shown_residual = "no complete pair of fold-side groups" if not math.isfinite(max_residual) else f"{max_residual:.3f} mm"
            reasons.append(f"maximum landmark residual {shown_residual} exceeds the {cap:g} mm cap")
        if common_area + 1e-9 < minimum_common:
            reasons.append(f"deduplicated common evidence {common_area:.3f} mm² is below {minimum_common:g} mm²")

        translation_list = None
        translation_sides = None
        if plan == "whole":
            translation_list = [round(float(x), 6) for x in whole_t]
        else:
            translation_list = [[round(float(x), 6) for x in item] for item in split_translations]
            translation_sides = {
                side_keys[s]: [round(float(x), 6) for x in split_translations[s]]
                for s in (0, 1)
            }
        info = {
            "plan": plan,
            "label": "Whole sheet, one translation" if plan == "whole" else "Two fold sides, separate translations",
            "eligible": not reasons,
            "qualified": not reasons,
            "fits": not reasons,
            "translation_mm": translation_list,
            "shift_mm": translation_list if plan == "whole" else None,
            "translation_x_mm": translation_list[0] if plan == "whole" and translation_list else None,
            "translation_y_mm": translation_list[1] if plan == "whole" and translation_list else None,
            "translations_mm": translation_list,
            "side_translations_mm": translation_sides,
            "max_residual_mm": None if not math.isfinite(max_residual) else round(max_residual, 3),
            "maximum_residual_mm": None if not math.isfinite(max_residual) else round(max_residual, 3),
            "rms_residual_mm": round(rms, 3) if math.isfinite(rms) else None,
            "mean_residual_mm": round(mean_residual, 3) if math.isfinite(mean_residual) else None,
            "residuals_mm": [None if not math.isfinite(x) else round(float(x), 3) for x in residual],
            "residual_cap_mm": round(cap, 6),
            "common_mm2": round(common_area, 3),
            "shared_mm2": round(common_area, 3),
            "common_visibility_mm2": round(common_area, 3),
            "joint_visibility_mm2": round(common_area, 3),
            "overlap_mm2": round(common_area, 3),
            "common_pixels": common_pixels,
            "shared_pixels": common_pixels,
            "overlap_pixels": common_pixels,
            "usable_common_mm2": round(usable_area, 3),
            "measurable_common_mm2": round(usable_area, 3),
            "paired_samples": usable_pixels,
            "common_samples": usable_pixels,
            "gross_common_mm2": round(square_mm(frame, common0) + square_mm(frame, common1), 3),
            "gross_common_pixels": gross_pixels,
            "conflict_mm2": round(conflict_area, 3),
            "fold_conflict_mm2": round(conflict_area, 3),
            "conflict_pixels": conflict_pixels,
            "minimum_common_mm2": round(minimum_common, 3),
            "reasons": reasons,
        }
        artifacts = {
            "coverage": coverage, "conflict": overlap, "measure": [measure0, measure1],
            "lumas": lumas, "common": common_mask, "usable": usable_mask,
            "target_side": target_side,
        }
        raw = {"_max": max_residual, "_rms": rms, "_common": common_area,
               "_usable": usable_area, "_order": order}
        return info, artifacts, raw

    whole_info, whole_art, whole_raw = plan_result("whole")
    split_info, split_art, split_raw = plan_result("split")
    plans = [whole_info, split_info]
    raw_plans = [whole_raw, split_raw]
    artifacts_by_plan = {"whole": whole_art, "split": split_art}
    eligible = [(info, raw) for info, raw in zip(plans, raw_plans) if info["eligible"]]

    forced_raw = _raw_option(opts, (
        "selected_plan", "adopted_plan", "use_plan", "plan", "alignment",
        "selection", "choice", "chosen_plan", "recommended_plan",
    ))
    if isinstance(forced_raw, dict):
        forced_raw = forced_raw.get("plan") or forced_raw.get("selected_plan")
    if isinstance(_raw_option(opts, ("placement",)), dict):
        placement = opts["placement"]
        forced_raw = forced_raw or placement.get("plan") or placement.get("selected_plan")
    forced = None
    if isinstance(forced_raw, str):
        forced_token = forced_raw.strip().lower().replace("-", "_").replace(" ", "_")
        if forced_token in {"whole", "global", "single", "whole_sheet", "one"}:
            forced = "whole"
        elif forced_token in {"split", "fold", "sides", "separate", "two_sides", "fold_split"}:
            forced = "split"

    if forced is not None:
        selected = next((item for item in plans if item["plan"] == forced), None)
        if selected is None or not selected["eligible"]:
            detail = "; ".join(selected["reasons"]) if selected else "unknown plan"
            raise Invalid(f"The requested {forced} alignment does not meet the limits: {detail}")
        selection_rule = f"The explicitly requested {forced} alignment met both limits and was used."
    else:
        if not eligible:
            details = " | ".join(
                f"{info['label']}: {'; '.join(info['reasons'])}" for info in plans)
            raise Invalid("Neither alignment plan meets every landmark cap and the common-evidence minimum. " + details)
        preference = str(_raw_option(opts, (
            "prefer", "priority", "preference", "selection_rule", "tie_break", "choose_by",
            "strategy", "selection_strategy", "decision", "criterion", "choose",
            "selection", "choice",
        )) or "accuracy").lower()
        accuracy_first = any(word in preference for word in ("accura", "precision", "residual", "tight", "exact"))
        if _raw_option(opts, ("prefer_accuracy", "accuracy_first")) is True:
            accuracy_first = True
        if _raw_option(opts, ("prefer_common", "prefer_shared", "coverage_first")) is True:
            accuracy_first = False

        if len(eligible) == 1:
            selected = eligible[0][0]
            selection_rule = ("Only one plan met the landmark residual cap and common-evidence minimum; "
                              "it was adopted.")
        elif accuracy_first:
            selected, _ = min(eligible, key=lambda pair: (
                pair[1]["_max"], pair[1]["_rms"], -pair[1]["_common"], pair[1]["_order"]))
            selection_rule = ("Both plans qualified. The pre-registered tie-break prefers the lower maximum "
                              "landmark residual, then lower RMS residual, then more deduplicated common evidence.")
        else:
            selected, _ = min(eligible, key=lambda pair: (
                -pair[1]["_common"], pair[1]["_max"], pair[1]["_rms"], pair[1]["_order"]))
            selection_rule = ("Both plans qualified. The pre-registered rule preserves more deduplicated common "
                              "evidence; lower maximum residual and then lower RMS residual break ties, with the "
                              "single whole-sheet translation first if still tied.")

    selected_key = selected["plan"]
    artifacts = artifacts_by_plan[selected_key]
    selected_t_per_point = np.broadcast_to(
        whole_t, shifts_mm.shape) if selected_key == "whole" else split_t_per_point

    rows = []
    for info in plans:
        rows.append({
            "section": "Alignment plans",
            "kind": "alignment_plan",
            "plan": info["plan"],
            "state": "selected" if info["plan"] == selected_key else (
                "eligible but not selected" if info["eligible"] else "not eligible"),
            "selected": info["plan"] == selected_key,
            "eligible": info["eligible"],
            "translation_x_mm": info["translation_mm"][0] if info["plan"] == "whole" and info["translation_mm"] else None,
            "translation_y_mm": info["translation_mm"][1] if info["plan"] == "whole" and info["translation_mm"] else None,
            "max_residual_mm": info["max_residual_mm"],
            "rms_residual_mm": info["rms_residual_mm"],
            "residual_cap_mm": info["residual_cap_mm"],
            "common_mm2": info["common_mm2"],
            "common_pixels": info["common_pixels"],
            "usable_common_mm2": info["usable_common_mm2"],
            "conflict_mm2": info["conflict_mm2"],
            "minimum_common_mm2": info["minimum_common_mm2"],
            "note": "; ".join(info["reasons"]) or selection_rule if info["plan"] == selected_key else "; ".join(info["reasons"]),
        })

    for index, ((dx, dy), whole_delta, split_delta) in enumerate(
            zip(shifts_mm, whole_residual, split_residual), 1):
        side = int(side_id[index - 1])
        rows.append({
            "section": "Landmark residuals",
            "kind": "landmark_residual",
            "landmark": index,
            "side": bank_names[side],
            "bank": bank_names[side],
            "side_key": side_keys[side],
            "shift_x_mm": round(float(dx), 3),
            "shift_y_mm": round(float(dy), 3),
            "whole_residual_mm": round(float(whole_delta), 3),
            "split_residual_mm": None if not math.isfinite(split_delta) else round(float(split_delta), 3),
            "residual_mm": round(float(
                whole_delta if selected_key == "whole" else split_delta), 3),
            "residual_cap_mm": round(cap, 3),
            "within_limit": bool((whole_delta if selected_key == "whole" else split_delta) <= cap + 1e-9),
        })

    shown = image(frame)
    source_missing = domain & ~frame.useful
    comparison_missing = domain & frame.useful & ~artifacts["coverage"]
    conflict_mask = artifacts["conflict"]
    coverage_specs = (
        ("Fold placement conflict", "fold_conflict", "折线冲突", conflict_mask, (.83, .22, .48), .62),
        ("Missing in source photograph", "source_missing", "本照缺拍", source_missing, (.65, .54, .79), .60),
        ("Missing in comparison photograph", "comparison_missing", "对照照缺拍", comparison_missing, (.91, .68, .27), .55),
    )
    for state, kind, zh, mask, color, opacity in coverage_specs:
        paint(shown, mask, color, opacity)
        rows.append({
            "section": "Coverage states", "kind": kind, "state": state, "state_zh": zh,
            "side": "fold", "bank": "fold", "area_mm2": round(square_mm(frame, mask), 3),
            "pixels": int(mask.sum()), "window_pct": percent(mask, domain),
        })

    aligned_delta = np.zeros(frame.rgb.shape[:2], dtype=np.float32)
    side_summary = {}
    change_colors = {
        "Brighter in comparison": ("brighter", (.92, .62, .24)),
        "Darker in comparison": ("darker", (.29, .55, .86)),
        "Within threshold": ("within", (.37, .72, .57)),
    }
    for side in (0, 1):
        mask = artifacts["measure"][side]
        delta = artifacts["lumas"][side] - frame.luma
        values = delta[mask]
        median_change = round(float(np.median(values)), 4) if values.size else None
        aligned_delta[mask] = delta[mask]
        side_key = side_keys[side]
        side_info = {"side": bank_names[side], "side_key": side_key,
                     "paired_samples": int(mask.sum()), "median_change": median_change}
        for state, (kind, color) in change_colors.items():
            if state == "Brighter in comparison":
                pixels = mask & (delta > difference_bound)
            elif state == "Darker in comparison":
                pixels = mask & (delta < -difference_bound)
            else:
                pixels = mask & (np.abs(delta) <= difference_bound)
            paint(shown, pixels, color, .34 if kind != "within" else .24)
            area = round(square_mm(frame, pixels), 3)
            side_info[f"{kind}_mm2"] = area
            side_info[f"{kind}_pixels"] = int(pixels.sum())
            rows.append({
                "section": "Transmission by fold side",
                "kind": kind,
                "plan": selected_key,
                "side": bank_names[side],
                "bank": bank_names[side],
                "side_key": side_key,
                "state": state,
                "area_mm2": area,
                "pixels": int(pixels.sum()),
                "paired_pct": percent(pixels, mask),
                "side_paired_pct": percent(pixels, mask),
                "difference_threshold": difference_bound,
                "median_change": median_change,
                "paired_samples": int(mask.sum()),
            })
        side_summary[side_key] = side_info
        side_summary[bank_names[side]] = side_info

    shown = np.rint(np.clip(shown, 0, 1) * 255).astype(np.uint8)
    cv.line(shown, tuple(np.rint(first).astype(int)), tuple(np.rint(second).astype(int)),
            (20, 24, 28), 5, cv.LINE_AA)
    cv.line(shown, tuple(np.rint(first).astype(int)), tuple(np.rint(second).astype(int)),
            (244, 244, 238), 2, cv.LINE_AA)
    for side in (0, 1):
        mask = artifacts["measure"][side]
        if mask.any():
            yy, xx = np.nonzero(mask)
            center = (int(xx.mean()), int(yy.mean()))
        else:
            center = tuple(np.rint(first + fold_normal * (42 if side == 0 else -42)).astype(int))
        cv.circle(shown, center, 13, (20, 24, 28), -1, cv.LINE_AA)
        cv.putText(shown, chr(ord("A") + side), (center[0] - 5, center[1] + 5),
                   cv.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2, cv.LINE_AA)

    for index, (ref_point, other_point) in enumerate(zip(ref_px, other_px)):
        translation = selected_t_per_point[index]
        moved_mm = other_mm[index] + translation
        end = tuple(np.rint(ref_point).astype(int))
        start = tuple(np.rint(moved_mm / frame.step).astype(int))
        residual_value = float(np.linalg.norm(shifts_mm[index] - translation))
        color = (214, 69, 65) if residual_value > cap + 1e-9 else (238, 115, 59)
        cv.circle(shown, end, 4, (154, 133, 201), 2, cv.LINE_AA)
        if start != end:
            cv.arrowedLine(shown, start, end, color, 2, cv.LINE_AA, tipLength=.45)
        else:
            cv.circle(shown, start, 3, color, -1, cv.LINE_AA)

    measured = artifacts["usable"]
    all_median = round(float(np.median(aligned_delta[measured])), 4) if measured.any() else None
    source_missing_area = square_mm(frame, source_missing)
    comparison_missing_area = square_mm(frame, comparison_missing)
    comparison_absent_area = square_mm(frame, domain & ~artifacts["coverage"])
    both_missing_area = square_mm(frame, domain & ~frame.useful & ~artifacts["coverage"])
    conflict_area = square_mm(frame, conflict_mask)
    common_choice = min(eligible, key=lambda pair: (
        -pair[1]["_common"], pair[1]["_max"], pair[1]["_order"]))[0]["plan"] if eligible else None
    accuracy_choice = min(eligible, key=lambda pair: (
        pair[1]["_max"], pair[1]["_rms"], -pair[1]["_common"], pair[1]["_order"]))[0]["plan"] if eligible else None
    summary = {
        "mode": "fold-alignment",
        "title": "Fold alignment: whole or split-side transmission",
        "selected_plan": selected_key,
        "selected_mode": selected_key,
        "selected_alignment": selected_key,
        "selected": selected_key,
        "chosen": selected_key,
        "chosen_plan": selected_key,
        "recommended": selected_key,
        "recommendation": selected_key,
        "choice": selected_key,
        "adopted_plan": selected_key,
        "alignment": selected_key,
        "plan": selected_key,
        "common_choice": common_choice,
        "more_common_plan": common_choice,
        "accuracy_choice": accuracy_choice,
        "more_accurate_plan": accuracy_choice,
        "selection_rule": selection_rule,
        "residual_cap_mm": round(cap, 3),
        "residual_tolerance_mm": round(cap, 3),
        "tolerance_mm": round(cap, 3),
        "max_residual_mm": selected["max_residual_mm"],
        "maximum_residual_mm": selected["maximum_residual_mm"],
        "max_residual": selected["max_residual_mm"],
        "minimum_common_mm2": round(minimum_common, 3),
        "min_common_mm2": round(minimum_common, 3),
        "minimum_shared_mm2": round(minimum_common, 3),
        "min_shared_mm2": round(minimum_common, 3),
        "common_area_mm2": selected["common_mm2"],
        "shared_area_mm2": selected["shared_mm2"],
        "common_mm2": selected["common_mm2"],
        "shared_mm2": selected["shared_mm2"],
        "common_pixels": selected["common_pixels"],
        "shared_pixels": selected["shared_pixels"],
        "usable_common_mm2": selected["usable_common_mm2"],
        "paired_samples": int(measured.sum()),
        "common_samples": int(measured.sum()),
        "conflict_mm2": round(conflict_area, 3),
        "fold_conflict_mm2": round(conflict_area, 3),
        "conflict_area_mm2": round(conflict_area, 3),
        "overlap_conflict_mm2": round(conflict_area, 3),
        "conflict_pixels": int(conflict_mask.sum()),
        "source_missing_mm2": round(source_missing_area, 3),
        "source_missing_pixels": int(source_missing.sum()),
        "first_missing_mm2": round(source_missing_area, 3),
        "source_photo_missing_mm2": round(source_missing_area, 3),
        "comparison_missing_mm2": round(comparison_missing_area, 3),
        "comparison_missing_pixels": int(comparison_missing.sum()),
        "other_missing_mm2": round(comparison_missing_area, 3),
        "comparison_absent_mm2": round(comparison_absent_area, 3),
        "both_missing_mm2": round(both_missing_area, 3),
        "second_missing_mm2": round(comparison_missing_area, 3),
        "comparison_photo_missing_mm2": round(comparison_missing_area, 3),
        "missing_mm2": round(source_missing_area + comparison_missing_area, 3),
        "either_missing_mm2": round(source_missing_area + comparison_missing_area, 3),
        "missing_pixels": int(source_missing.sum() + comparison_missing.sum()),
        "median_change": all_median,
        "difference_threshold": difference_bound,
        "translation_mm": selected["translation_mm"],
        "translations_mm": selected["translations_mm"],
        "translation_x_mm": selected.get("translation_x_mm"),
        "translation_y_mm": selected.get("translation_y_mm"),
        "shift_x_mm": selected.get("translation_x_mm"),
        "shift_y_mm": selected.get("translation_y_mm"),
        "dx_mm": selected.get("translation_x_mm"),
        "dy_mm": selected.get("translation_y_mm"),
        "side_translations_mm": selected["side_translations_mm"],
        "side_results": side_summary,
        "left_result": side_summary.get("left"),
        "right_result": side_summary.get("right"),
        "top_result": side_summary.get("top"),
        "bottom_result": side_summary.get("bottom"),
        "left": side_summary.get("left"),
        "right": side_summary.get("right"),
        "top": side_summary.get("top"),
        "bottom": side_summary.get("bottom"),
        "plans": plans,
        "alignments": plans,
        "plan_results": plans,
        "candidate_plans": plans,
        "candidates": plans,
        "whole_plan": whole_info,
        "split_plan": split_info,
        "whole": whole_info,
        "split": split_info,
    }
    return summary, rows, shown


fold_registration = fold_alignment
fold_align = fold_alignment
fold_registration_reading = fold_alignment
split_alignment = fold_alignment
split_registration = fold_alignment
compare_fold_alignments = fold_alignment
compare_alignment_plans = fold_alignment
aligned_transmission = fold_alignment
aligned_transmission_difference = fold_alignment
fold_transmission_change = fold_alignment
fold_transmission_difference = fold_alignment


def landmark_offset(frame, other, marks, opts):
    if _alignment_requested(opts, marks):
        return fold_alignment(frame, other, marks, opts)
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
    if _alignment_requested(opts, marks):
        return fold_alignment(frame, other, marks, opts)
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
