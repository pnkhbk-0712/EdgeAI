"""Helmet shell colour from a YOLO helmet box. Shared by export_color_crops.py and build_role_dataset.py.

Region = top ~45% of the box, centre 70% width (the shell, little face). Pixels with s>=SAT_MIN vote for a
hue name (orange needs more saturation: skin is desaturated orange). No dominant hue -> white (bright, almost no saturated
pixels) or unclear (dark/grey/ambiguous; black and grey are not told apart: too few and too noisy). Dim photos keep their hue
(low v is allowed) so shadowed yellow/red/blue helmets do not fall into black/grey.
"""
import colorsys

import numpy as np

CHROMA_MIN = 0.20
WHITE_CHROMA_MAX = 0.10
V_MIN = 0.15
HUE = ((0, 12, "red", 0.30), (12, 40, "orange", 0.40), (40, 70, "yellow", 0.25), (70, 170, "green", 0.22), (170, 260, "blue", 0.18), (260, 361, "red", 0.30))


def region(im, x, y, w, h):
    W, H = im.size
    x0, x1 = (x - 0.35 * w) * W, (x + 0.35 * w) * W
    y0, y1 = (y - 0.50 * h) * H, (y - 0.05 * h) * H
    return im.crop((int(x0), int(y0), max(int(x1), int(x0) + 1), max(int(y1), int(y0) + 1)))


def colour_of(crop):
    rgb = np.asarray(crop).reshape(-1, 3)
    rgb = rgb[:: max(1, len(rgb) // 600)]
    hsv = np.array([colorsys.rgb_to_hsv(*(p / 255)) for p in rgb])
    h, s, v = hsv[:, 0] * 360, hsv[:, 1], hsv[:, 2]
    n = len(hsv)
    frac = {}
    for lo, hi, name, smin in HUE:
        m = (h >= lo) & (h < hi) & (s >= smin) & (v >= V_MIN)
        frac[name] = frac.get(name, 0) + m.sum() / n
    best = max(frac, key=frac.get)
    if frac[best] >= CHROMA_MIN:
        return best
    # no dominant hue: white only if bright AND almost no saturated pixels; everything else is unclear
    if v.mean() >= 0.55 and sum(frac.values()) < WHITE_CHROMA_MAX:
        return "white"
    return "unclear"
