"""Round 14 (D131): display versions of the Round-13 effect figures for the manuscript, by cropping only (no data is
re-plotted). The figures' suptitle strip is removed (it is clipped in the single-panel figures and is replaced by the
manuscript caption), and the six-panel PBC figures are split into two three-panel rows so that they stay legible at
page width; the four single-panel prothro/aids figures are placed side by side in one strip.
Output: reports/figures/ms14_<source stem>[_a|_b].png and ms14_r13_effects_prothro_aids_strip.png."""
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports/figures"
TOP = 30            # px: the suptitle strip of analyse_round13.plot_effects at its default size


def main():
    for src in sorted(FIG.glob("r13_effects_*.png")):
        im = Image.open(src)
        w, h = im.size
        body = im.crop((0, TOP, w, h))
        if w > 1000:                                   # six panels of equal width
            half = w // 2
            body.crop((0, 0, half, h - TOP)).save(FIG / f"ms14_{src.stem}_a.png")
            body.crop((half + 8, 0, w, h - TOP)).save(FIG / f"ms14_{src.stem}_b.png")
        else:
            body.save(FIG / f"ms14_{src.stem}.png")
    singles = [Image.open(FIG / f"ms14_r13_effects_{s}.png") for s in
               ("prothro_b2s_cox", "prothro_b1_fpca1_cox", "aids_b2s_cox", "aids_b1_fpca1_cox")]
    strip = Image.new("RGB", (sum(i.size[0] for i in singles), max(i.size[1] for i in singles)), "white")
    x = 0
    for i in singles:
        strip.paste(i, (x, 0))
        x += i.size[0]
    strip.save(FIG / "ms14_r13_effects_prothro_aids_strip.png")
    print(sorted(p.name for p in FIG.glob("ms14_*.png")))


if __name__ == "__main__":
    main()
