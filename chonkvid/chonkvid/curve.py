"""Render the epidemic curves to a PNG so Le Chonk can *look* at the outbreak.

The WHO agent gets numbers as text from its tools; this gives it the same picture a
human reviewer sees on the dashboard, so it can reason about the shape of the curve
(not just the latest counts). Pure PIL, no plotting library, deterministic output.
"""

from __future__ import annotations

import io

from PIL import Image, ImageDraw

W, H = 720, 400
PAD_L, PAD_R, PAD_T, PAD_B = 54, 128, 36, 40
BG = (11, 13, 18)
GRID = (38, 43, 54)
TEXT = (185, 192, 204)
SERIES = [
    ("S", (110, 168, 254)),
    ("E", (255, 196, 87)),
    ("I", (255, 90, 61)),
    ("R", (86, 214, 140)),
]


def render_curves(history: list[dict], source: str = "live") -> bytes:
    """history: [{"tick": int, "counts": {"S": n, "E": n, "I": n, "R": n}}, ...]

    Returns PNG bytes. Raises ValueError on an empty series.
    """
    if not history:
        raise ValueError("no epidemic history to draw")
    ticks = [h["tick"] for h in history]
    t_max = max(1, max(ticks))
    n_max = max(1, max(sum(h["counts"].values()) for h in history))

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    plot_w, plot_h = W - PAD_L - PAD_R, H - PAD_T - PAD_B

    def x(t: int) -> float:
        return PAD_L + plot_w * (t / t_max)

    def y(v: int) -> float:
        return PAD_T + plot_h * (1 - v / n_max)

    # axes and gridlines
    d.rectangle([PAD_L, PAD_T, PAD_L + plot_w, PAD_T + plot_h], outline=GRID)
    for frac in (0.25, 0.5, 0.75):
        gy = PAD_T + plot_h * frac
        d.line([PAD_L, gy, PAD_L + plot_w, gy], fill=GRID)
    d.text((PAD_L, 12), f"ChonkVid epidemic · {source} · {len(history)} ticks", fill=TEXT)
    d.text((PAD_L, PAD_T + plot_h + 10), "tick", fill=TEXT)
    d.text((8, PAD_T), str(n_max), fill=TEXT)
    d.text((8, PAD_T + plot_h - 10), "0", fill=TEXT)

    for i, (key, colour) in enumerate(SERIES):
        pts = [(x(h["tick"]), y(h["counts"].get(key, 0))) for h in history]
        if len(pts) > 1:
            d.line(pts, fill=colour, width=3)
        for px, py in pts:
            d.ellipse([px - 2, py - 2, px + 2, py + 2], fill=colour)
        lx, ly = PAD_L + plot_w + 18, PAD_T + 18 + i * 26
        d.line([lx, ly + 6, lx + 22, ly + 6], fill=colour, width=3)
        last = history[-1]["counts"].get(key, 0)
        d.text((lx + 30, ly), f"{key} = {last}", fill=TEXT)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
