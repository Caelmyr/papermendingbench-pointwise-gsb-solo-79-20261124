"""Two-policy adjudication of front pale ink vs. back-side show-through.

Policy A (face only): accept when the reflected photo alone looks ink-like
          relative to confirmed blank paper (Lab deltaE above a noise-anchored cutoff).
Policy B (dual check): additionally require the transmitted photo to be visible
          there and to carry an ink-supporting blue chroma signature.
Pixels missing in either photograph (the two black defect tiles) stay undecided.
"""
import json
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, "/opt/pointwise-env/site-packages")
sys.path.insert(0, "/workspace")

import cv2 as cv
import numpy as np

DERIVED = Path("/workspace/work/photos/derived")
OUT = Path("/workspace/work/analysis/results")
OUT.mkdir(parents=True, exist_ok=True)
RUN_W, RUN_H = 222.0, 148.0
TRANS_CHROMA = 0.08          # bench ink-spread default
RETENTION_CAP = 0.90         # user: front pale-ink retention minimum
BACK_FA_CAP = 0.08           # user: back-character false acceptance maximum


@dataclass
class Frame:
    rgb: np.ndarray
    support: np.ndarray
    step: tuple

    @property
    def luma(self):
        return self.rgb @ np.array([.2126, .7152, .0722], np.float32)

    @property
    def useful(self):
        return self.support & (self.rgb.max(axis=2) >= .018) & (self.rgb.min(axis=2) < .993)

    @property
    def br(self):
        return self.rgb[:, :, 2] - self.rgb[:, :, 0]


def load_frame(path):
    rgba = cv.imread(str(path), cv.IMREAD_UNCHANGED)
    h, w = rgba.shape[:2]
    rgb = cv.cvtColor(rgba, cv.COLOR_BGRA2RGB).astype(np.float32) / 255.0
    support = rgba[:, :, 3] >= 254
    return Frame(rgb, support, (RUN_W / (w - 1), RUN_H / (h - 1)))


def box_mask(shape, box):
    x0, y0, x1, y1 = box
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


face = load_frame(DERIVED / "face-derived.png")
trans = load_frame(DERIVED / "trans-derived.png")
back_strokes = cv.imread(str(DERIVED / "derived-back-mask.png"), cv.IMREAD_GRAYSCALE) > 0

WINDOWS = {
    "W_ink": (170, 208, 265, 300),
    "W_back": (565, 160, 690, 240),
    "W_blank": (520, 380, 560, 420),
}
MISSING = {"face_black_tile": (461, 335, 478, 350), "trans_black_tile": (425, 290, 442, 321)}
win_masks = {k: box_mask(face.support.shape, v) for k, v in WINDOWS.items()}
miss_masks = {k: box_mask(face.support.shape, v) for k, v in MISSING.items()}

# Ground-truth front ink inside the ink window: blue chroma signature in the face photo.
gt_ink = win_masks["W_ink"] & face.useful & (face.br >= TRANS_CHROMA)
gt_back = win_masks["W_back"] & back_strokes
gt_blank = win_masks["W_blank"] & face.useful & trans.useful

# ---- Face ink-likeness: Lab deltaE against confirmed blank paper -------------------
lab_f = cv.cvtColor(face.rgb, cv.COLOR_RGB2LAB)
lab_t = cv.cvtColor(trans.rgb, cv.COLOR_RGB2LAB)
ref_lab = np.median(lab_f[gt_blank].reshape(-1, 3), axis=0)
delta = np.sqrt(((lab_f - ref_lab) ** 2).sum(axis=2))

blank_dev = delta[gt_blank]
print("blank-paper Lab reference:", ref_lab.round(2).tolist())
print("blank deltaE p50/p90/p95/p99/max:",
      np.percentile(blank_dev, [50, 90, 95, 99, 100]).round(2).tolist())
for name, m in (("gt front ink", gt_ink), ("gt back show-through", gt_back & face.useful)):
    d = delta[m]
    print(f"{name:22s} deltaE p05/p50/p95: {np.percentile(d, [5, 50, 95]).round(2).tolist()}")
print("trans B-R  front ink p05/p50:", np.percentile(trans.br[gt_ink], [5, 50]).round(3).tolist(),
      " back strokes p05/p50:", np.percentile(trans.br[gt_back & trans.useful], [5, 50]).round(3).tolist())

# ---- Threshold sweep: pick the two presets ----------------------------------------
sweep = []
for tau in np.round(np.arange(1.0, 30.01, 0.25), 2):
    face_like = delta >= tau
    known_ink = gt_ink & face.useful
    known_back = gt_back & face.useful
    retain = face_like[known_ink].mean()
    fa = face_like[known_back].mean()
    sweep.append((tau, retain, fa))
print("\ntau  retention  backFA")
for tau, retain, fa in sweep:
    print(f"{tau:5.2f}  {retain:.3f}   {fa:.3f}")

# Preset P1 "keep more pale ink": lowest tau still passing both caps (favors retention).
# Preset P2 "recognize fewer back chars": lowest tau driving back false acceptance to zero
# while keeping full retention (the conservative end of the usable range).
passing = [(tau, r, fa) for tau, r, fa in sweep if r >= RETENTION_CAP and fa <= BACK_FA_CAP]
assert passing, "no face threshold passes both caps"
tau_keep = passing[0][0]
zero_fa = [p for p in passing if p[2] == 0.0]
tau_strict = (zero_fa or passing)[0][0]
print("\npassing thresholds:", [(t, round(r, 3), round(fa, 3)) for t, r, fa in passing])
print("P1 keep-more-ink tau =", tau_keep, " P2 fewer-back tau =", tau_strict)


def adjudicate(tau):
    face_like = delta >= tau
    trans_supports = trans.useful & (trans.br >= TRANS_CHROMA)
    known = face.useful & trans.useful
    # adjudication domain: all three windows plus the two missing-shot tiles
    domain = np.zeros(face.support.shape, bool)
    for m in win_masks.values():
        domain |= m
    for m in miss_masks.values():
        domain |= m

    undecided = domain & ~known
    a_accept = known & face_like
    b_accept = known & face_like & trans_supports
    return face_like, trans_supports, domain, known, undecided, a_accept, b_accept


def metrics(tag, accept):
    ki = gt_ink & face.useful & trans.useful
    kb = gt_back & face.useful & trans.useful
    retain = accept[ki].mean()
    fa = accept[kb].mean()
    return {"preset": tag, "retention": round(float(retain), 4),
            "back_false_accept": round(float(fa), 4),
            "retention_pass": bool(retain >= RETENTION_CAP),
            "back_fa_pass": bool(fa <= BACK_FA_CAP),
            "accepted_ink_px": int(accept[ki].sum()), "known_ink_px": int(ki.sum()),
            "accepted_back_px": int(accept[kb].sum()), "known_back_px": int(kb.sum())}


results = {}
for tag, tau in (("P1_keep_more_ink", tau_keep), ("P2_fewer_back", tau_strict)):
    face_like, supports, domain, known, undecided, a_accept, b_accept = adjudicate(tau)
    pol = {
        "tau_deltaE": tau,
        "policy_A_face_only": {**metrics(tag + "/A", a_accept),
                               "global_accept_px": int((a_accept & domain).sum())},
        "policy_B_dual_check": {**metrics(tag + "/B", b_accept),
                                "global_accept_px": int((b_accept & domain).sum())},
        "undecided_px": int((undecided & domain).sum()),
    }
    for tile, m in miss_masks.items():
        pol[f"undecided::{tile}"] = int((undecided & m).sum())
    # blank-paper false ink rate (should be ~0)
    pol["blank_false_ink_A"] = round(float(a_accept[gt_blank].mean()), 4)
    pol["blank_false_ink_B"] = round(float(b_accept[gt_blank].mean()), 4)
    results[tag] = pol

# ---- Paper-tone rows on the face photo ---------------------------------------------
def lab_rows(mask, label):
    px = lab_f[mask & face.useful].reshape(-1, 3)
    p10, med, p90 = np.percentile(px, [10, 50, 90], axis=0)
    return {"group": label, "pixels": int(px.shape[0]),
            "L_p10": round(float(p10[0]), 2), "L_median": round(float(med[0]), 2), "L_p90": round(float(p90[0]), 2),
            "a_p10": round(float(p10[1]), 2), "a_median": round(float(med[1]), 2), "a_p90": round(float(p90[1]), 2),
            "b_p10": round(float(p10[2]), 2), "b_median": round(float(med[2]), 2), "b_p90": round(float(p90[2]), 2)}

tone_rows = [lab_rows(gt_blank, "observed blank paper"),
             lab_rows(gt_back & face.useful, "back-side show-through only")]

report = {
    "thresholds": {"trans_blue_chroma_BminusR": TRANS_CHROMA,
                   "retention_minimum": RETENTION_CAP, "back_false_accept_maximum": BACK_FA_CAP,
                   "blank_lab_reference": ref_lab.round(3).tolist(),
                   "blank_deltaE_p99": round(float(np.percentile(blank_dev, 99)), 3)},
    "presets": results,
    "paper_tone_face_rows": tone_rows,
}
(OUT / "adjudication.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
print("\n" + json.dumps(report, ensure_ascii=False, indent=2))

# ---- Overlay of the adopted policy (filled later by choose step; render both) ------
def overlay(accept, name):
    shown = (face.rgb * 255).astype(np.uint8).copy()
    shown = cv.cvtColor(shown, cv.COLOR_RGB2BGR)
    def paint(mask, color, alpha=.55):
        layer = shown.copy()
        layer[mask] = color
        cv.addWeighted(layer, alpha, shown, 1 - alpha, 0, shown)
    paint(gt_ink & accept, (60, 200, 90))       # accepted true ink: green
    paint(gt_ink & ~accept, (0, 200, 255))      # rejected true ink (loss): orange-yellow
    paint(gt_back & accept, (60, 60, 230))      # back wrongly accepted: red
    paint(gt_back & ~accept & face.useful, (200, 170, 90))  # back correctly rejected: blue
    und = domain & ~(face.useful & trans.useful)
    paint(und, (200, 120, 220), .45)            # undecided: purple
    for label, (x0, y0, x1, y1) in WINDOWS.items():
        cv.rectangle(shown, (x0, y0), (x1, y1), (255, 255, 255), 1, cv.LINE_AA)
    cv.line(shown, (252, 196), (244, 346), (200, 230, 0), 1, cv.LINE_AA)
    cv.imwrite(str(OUT / name), shown)

for tag, tau in (("P1", tau_keep), ("P2", tau_strict)):
    _, _, domain, _, _, a_accept, b_accept = adjudicate(tau)
    overlay(a_accept, f"overlay-{tag}-policyA-face-only.png")
    overlay(b_accept, f"overlay-{tag}-policyB-dual-check.png")
print("\noverlays and JSON written to", OUT)
