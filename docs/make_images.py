"""Render docs/results.png (README) and docs/social-preview.png (GitHub social preview, 1280x640).

Numbers are the measurements from the README table. Needs matplotlib; not used by the service.
Usage: python3 docs/make_images.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

OUT = Path(__file__).resolve().parent
ACCENT, GRAY, INK, MUTED, GRID = "#2a78d6", "#b4b2a9", "#1f1f1d", "#6b6a65", "#e1e0d9"
FONT = "Inter" if any("Inter" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
plt.rcParams.update({"font.family": FONT, "font.size": 12, "text.color": INK,
                     "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": INK})

# variant, seconds per compaction on large windows (mean), facts recalled of 96
ROWS = [
    ("Model alone (Hermes default)", 198, 77),
    ("VCC projector", 0.005, 79),
    ("compress-fast rules only", 0.01, 84),
    ("compress-fast hybrid", 34, 85),
]


def _fmt_secs(s: float) -> str:
    return f"{s * 1000:.0f} ms" if s < 1 else f"{s:.0f} s"


def results() -> None:
    names = [r[0] for r in ROWS]
    colors = [ACCENT if "hybrid" in n else GRAY for n in names]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.4), dpi=150, gridspec_kw={"wspace": 0.08})
    fig.patch.set_facecolor("white")
    y = range(len(ROWS))[::-1]
    a.barh(list(y), [r[1] for r in ROWS], color=colors, height=0.55)
    a.set_xlim(0, 260)
    a.set_yticks(list(y), names)
    a.set_title("Wait per compaction (large windows)", loc="left", fontsize=13, color=INK, pad=10)
    for yi, r in zip(y, ROWS):
        a.text(r[1] + 4, yi, _fmt_secs(r[1]), va="center", color=INK, fontsize=11)
    b.barh(list(y), [r[2] for r in ROWS], color=colors, height=0.55)
    b.set_xlim(0, 96)
    b.set_yticks([])
    b.set_title("Facts recalled after compaction (of 96)", loc="left", fontsize=13, color=INK, pad=10)
    for yi, r in zip(y, ROWS):
        b.text(r[2] - 2, yi, str(r[2]), va="center", ha="right", color="white", fontsize=11, weight="bold")
    for ax in (a, b):
        ax.set_facecolor("white")
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", colors=MUTED, labelsize=10)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    fig.text(0.01, -0.04, "Hermes 0.22.5, local Qwen3.8-flash-next IQ3_XXS, 8 compaction prompts, 96 quiz questions. "
             "Details and caveats in the README.", color=MUTED, fontsize=9)
    fig.savefig(OUT / "results.png", bbox_inches="tight", pad_inches=0.25, facecolor="white")
    plt.close(fig)


def social() -> None:
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
    fig.patch.set_facecolor("#14161a")
    t = lambda x, y, s, **k: fig.text(x, y, s, **{"color": "#f0efec", **k})  # noqa: E731
    t(0.07, 0.80, "compress-fast", fontsize=58, weight="bold")
    t(0.07, 0.70, "Context compaction for Hermes Agent on local models", fontsize=24, color="#c3c2b7")
    t(0.07, 0.42, "198 s → 34 s", fontsize=52, weight="bold", color="#6aa8f0")
    t(0.07, 0.34, "wait per compaction", fontsize=20, color="#c3c2b7")
    t(0.52, 0.42, "85 vs 77", fontsize=52, weight="bold", color="#6aa8f0")
    t(0.52, 0.34, "facts kept (of 96)", fontsize=20, color="#c3c2b7")
    t(0.07, 0.12, "Drop-in OpenAI-compatible proxy  ·  no Hermes patches  ·  Python stdlib  ·  MIT",
      fontsize=18, color="#898781")
    fig.savefig(OUT / "social-preview.png", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    results()
    social()
    print("wrote", OUT / "results.png", OUT / "social-preview.png")
