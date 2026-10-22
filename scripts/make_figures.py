"""Generate the README figures (light + dark variants).

    python scripts/make_figures.py

Outputs (written to ./figures):
    architecture-{light,dark}.svg   - preprocessing pipeline and the VGGish + TB-head network
    baseline-auroc-{light,dark}.svg - our baselines against Zhang et al. (2026)
    probe-{light,dark}.svg          - what a linear probe recovers from the cough features

Every number below is copied from the printed outputs of notebooks/01_zambia_baseline.ipynb
(Steps 9-11) and notebooks/02_embedding_analysis.ipynb (Sections 3 and 5).
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES = os.path.join(ROOT, "figures")

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "panel": "#f4f3ef",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "s1": "#2a78d6",
        "s1_soft": "#cde2fb",
        "s2": "#eb6834",
        "s2_soft": "#fbe0d4",
        "s3": "#1baf7a",
        "neutral": "#d5d4ce",
    },
    "dark": {
        "surface": "#1a1a19",
        "panel": "#232322",
        "ink": "#ffffff",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "s1": "#3987e5",
        "s1_soft": "#16375e",
        "s2": "#d95926",
        "s2_soft": "#4a2614",
        "s3": "#199e70",
        "neutral": "#4a4a47",
    },
}

FONT = 'system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif'
MONO = 'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", monospace'


# --------------------------------------------------------------------------- #
# Architecture diagram
# --------------------------------------------------------------------------- #
def architecture_svg(t):
    W, H = 960, 624
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'font-family=\'{FONT}\' role="img" aria-labelledby="title desc">',
        '<title id="title">Cough-audio pipeline and VGGish architecture</title>',
        '<desc id="desc">Session recordings from four simultaneous devices are relabelled and audited, '
        "cut into single coughs with a CLAP-based extractor, converted to 64 by 97 log-mel spectrograms, "
        "balanced and augmented for training only, and resized to 96 by 64. VGGish applies six 3 by 3 "
        "convolutions with 64, 128, 256, 256, 512 and 512 channels and four max-pooling layers, then fully "
        "connected layers of 4096, 4096 and 128 units. A head of 128 to 16 to 1 units outputs a TB logit. "
        "A participant's score is the median clip probability over all clips and four inner-fold models.</desc>",
        f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{t["muted"]}"/></marker></defs>',
        f'<rect width="{W}" height="{H}" rx="16" fill="{t["surface"]}"/>',
    ]

    def text(x, y, s, size=13, fill=None, weight=400, anchor="middle", family=None):
        fam = f" font-family='{family}'" if family else ""
        out.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill or t["ink"]}" text-anchor="{anchor}"{fam}>{s}</text>'
        )

    def arrow(x1, y1, x2, y2):
        out.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{t["muted"]}" stroke-width="1.5" marker-end="url(#arrow)"/>'
        )

    # ---- Part 1: data pipeline -------------------------------------------- #
    text(40, 44, "1 · DATA PIPELINE", 12, t["muted"], 700, "start")
    stages = [
        ("Session .wav", "4 devices", "recorded simultaneously", None),
        ("Relabel + audit", "62 fixes", "cross-correlation check", None),
        ("Cough extractor", "clips &lt; 1 s", "CLAP, same cuts per device", None),
        ("Log-mel", "64 × 97", "25 ms win · 10 ms hop", None),
        ("Balance + augment", "phone×site×TB", "train only", "dash"),
        ("Resize", "96 × 64", "VGGish input", None),
    ]
    bw, bh, gap, x0, y0 = 134, 84, 20, 40, 62
    for i, (title, big, sub, style) in enumerate(stages):
        x = x0 + i * (bw + gap)
        dash = ' stroke-dasharray="5 4"' if style == "dash" else ""
        stroke = t["s2"] if style == "dash" else t["axis"]
        fill = t["s2_soft"] if style == "dash" else t["panel"]
        out.append(
            f'<rect x="{x}" y="{y0}" width="{bw}" height="{bh}" rx="10" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="1.5"{dash}/>'
        )
        text(x + bw / 2, y0 + 24, title, 12.5, t["ink"], 600)
        text(x + bw / 2, y0 + 49, big, 13.5 if len(big) < 12 else 12, t["ink"], 700, family=MONO)
        text(x + bw / 2, y0 + 70, sub, 10.5, t["ink2"])
        if i < len(stages) - 1:
            arrow(x + bw + 2, y0 + bh / 2, x + bw + gap - 3, y0 + bh / 2)

    # elbow connector from Resize into the network panel
    py = 214
    fx = x0 + 5 * (bw + gap) + bw / 2
    out.append(
        f'<path d="M{fx},{y0 + bh + 4} L{fx},{y0 + bh + 24} L72,{y0 + bh + 24} L72,{py - 4}" '
        f'fill="none" stroke="{t["muted"]}" stroke-width="1.5" marker-end="url(#arrow)"/>'
    )

    # ---- Part 2: network -------------------------------------------------- #
    out.append(
        f'<rect x="24" y="{py}" width="{W - 48}" height="{H - py - 24}" rx="14" '
        f'fill="none" stroke="{t["grid"]}" stroke-width="1.5"/>'
    )
    text(40, py + 30, "2 · VGGISH (AudioSet-pretrained, fine-tuned end to end) + TB HEAD  ·  72.14 M parameters",
         12, t["muted"], 700, "start")

    # (kind, box width, box height, label above, label below)
    # conv stages: width grows with channels, height shrinks with the spatial size (time frames)
    items = [
        ("in", 12, 176, "1", "96×64"),
        ("conv", 22, 176, "64", "96×64"),
        ("conv", 30, 112, "128", "48×32"),
        ("conv", 42, 72, "256 ×2", "24×16"),
        ("conv", 56, 46, "512 ×2", "12×8"),
        ("flat", 12, 176, "12 288", "flatten"),
        ("fc", 22, 150, "4 096", "FC1"),
        ("fc", 22, 150, "4 096", "FC2"),
        ("emb", 22, 44, "128", "embed"),
        ("head", 22, 24, "16", "FC"),
        ("head", 22, 12, "1", "logit"),
    ]
    cy = py + 170
    left, right = 70, W - 70
    total_w = sum(w for _, w, *_ in items)
    step_gap = (right - left - total_w) / (len(items) - 1)
    x = left
    prev = None
    for kind, w, h, above, below in items:
        y = cy - h / 2
        if kind == "in" or kind == "flat":
            fill, stroke = t["neutral"], t["axis"]
        elif kind == "head":
            fill, stroke = t["s2"], t["s2"]
        elif kind == "emb":
            fill, stroke = t["s3"], t["s3"]
        else:
            fill, stroke = t["s1"], t["s1"]
        if kind == "conv":
            # a stacked look for the feature-map volume
            out.append(
                f'<rect x="{x + 5:.1f}" y="{y - 5:.1f}" width="{w}" height="{h}" rx="4" '
                f'fill="{t["s1_soft"]}" stroke="{stroke}" stroke-width="1"/>'
            )
        out.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w}" height="{h}" rx="4" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1"/>'
        )
        text(x + w / 2, cy - 100, above, 12, t["ink"], 700, family=MONO)
        text(x + w / 2, cy + 108, below, 11, t["ink2"], 600, family=MONO if "×" in below else None)
        if prev is not None:
            arrow(prev + 4, cy, x - 4, cy)
        prev = x + w + (5 if kind == "conv" else 0)
        x += w + step_gap

    # bracket labels for the two halves of the backbone
    def bracket(x1, x2, label):
        yb = cy + 126
        out.append(
            f'<path d="M{x1:.1f},{yb - 6} L{x1:.1f},{yb} L{x2:.1f},{yb} L{x2:.1f},{yb - 6}" '
            f'fill="none" stroke="{t["axis"]}" stroke-width="1.2"/>'
        )
        text((x1 + x2) / 2, yb + 18, label, 11.5, t["ink2"], 600)

    xs = [left]
    for _, w, *_ in items:
        xs.append(xs[-1] + w + step_gap)
    bracket(xs[1] - 4, xs[5] - step_gap + 4, "convolutional features · 4.50 M params")
    bracket(xs[6] - 4, xs[9] - step_gap + 4, "embedding MLP · 67.64 M")
    bracket(xs[9] - 4, xs[11] - step_gap + 4, "head · 2 081")

    # legend
    ly = H - 58
    chips = [
        (t["s1"], "Conv 3×3 + ReLU, then MaxPool 2×2"),
        (t["s3"], "128-d embedding"),
        (t["s2"], "Linear 128→16 · ReLU · Dropout 0.5 · Linear 16→1"),
    ]
    cx = 48
    for color, label in chips:
        out.append(f'<rect x="{cx}" y="{ly - 10}" width="14" height="14" rx="3" fill="{color}"/>')
        text(cx + 20, ly + 2, label, 11.5, t["ink2"], 600, "start")
        cx += 34 + 6.3 * len(label)
    text(48, ly + 26,
         "Participant score = median of sigmoid(logit) over all of the participant's clips and the 4 inner-fold models.",
         11.5, t["ink2"], 400, "start")

    out.append("</svg>")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def _style(ax, fig, t):
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    ax.grid(axis="y", color=t["grid"], lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["ink2"], length=0, labelsize=10)


def _rc():
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
                         "svg.fonttype": "none"})


def baseline_svg(t, path):
    """Our AUROC (mean ± s.d. over test folds) against the values published by Zhang et al."""
    _rc()
    groups = ["Questionnaire\n(all 1,823)", "Questionnaire\n(audio cohort, 325)", "Cough · VGGish\n(audio cohort, 325)"]
    ours = [(0.759, 0.029), (0.749, 0.094), (0.740, 0.086)]
    published = [(0.767, 0.025), None, (0.755, 0.056)]

    fig, ax = plt.subplots(figsize=(8.4, 4.4), dpi=100)
    _style(ax, fig, t)
    w = 0.34
    for i, (o, p) in enumerate(zip(ours, published)):
        ax.bar(i - w / 2, o[0], w, yerr=o[1], color=t["s1"], ecolor=t["ink2"], capsize=4,
               label="ours" if i == 0 else None)
        ax.text(i - w / 2, 0.42, f"{o[0]:.3f}", ha="center", va="bottom", rotation=90,
                color="#ffffff", fontsize=10, fontweight="bold")
        if p:
            ax.bar(i + w / 2, p[0], w, yerr=p[1], color=t["neutral"], ecolor=t["ink2"], capsize=4,
                   label="Zhang et al. (2026)" if i == 0 else None)
            ax.text(i + w / 2, 0.42, f"{p[0]:.3f}", ha="center", va="bottom", rotation=90,
                    color=t["ink"], fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(groups)), groups)
    ax.set_ylim(0.4, 0.95)
    ax.set_ylabel("AUROC (mean ± s.d. over test folds)", color=t["ink2"], fontsize=10.5)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=10.5, ncol=2)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    ax.set_title("Baselines on Zambia CIDRZ against the published values", loc="left",
                 color=t["ink"], fontsize=13, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(path, format="svg", facecolor=t["surface"])
    plt.close(fig)


def probe_svg(t, path):
    """Linear-probe AUROC on unseen participants: phone identity versus TB status."""
    _rc()
    feats = ["Spectrogram summary\n(mean + s.d. per mel band)", "VGGish embedding\n(pretrained, frozen)"]
    phone = [0.994, 0.912]
    tb_person = [0.665, 0.651]

    fig, ax = plt.subplots(figsize=(8.4, 4.2), dpi=100)
    _style(ax, fig, t)
    w = 0.34
    for i in range(2):
        ax.bar(i - w / 2, phone[i], w, color=t["s1"], label="recover the phone" if i == 0 else None)
        ax.bar(i + w / 2, tb_person[i], w, color=t["s2"], label="recover TB status (per person)" if i == 0 else None)
        ax.text(i - w / 2, phone[i] + 0.01, f"{phone[i]:.3f}", ha="center", va="bottom", color=t["ink"],
                fontsize=10.5, fontweight="bold")
        ax.text(i + w / 2, tb_person[i] + 0.01, f"{tb_person[i]:.3f}", ha="center", va="bottom",
                color=t["ink"], fontsize=10.5, fontweight="bold")
    ax.axhline(0.5, color=t["muted"], lw=1.2, ls="--")
    ax.text(1.62, 0.505, "chance", color=t["muted"], fontsize=10, ha="right", va="bottom")
    ax.set_xticks(range(2), feats)
    ax.set_xlim(-0.6, 1.65)
    ax.set_ylim(0.4, 1.05)
    ax.set_ylabel("AUROC on unseen participants", color=t["ink2"], fontsize=10.5)
    leg = ax.legend(loc="upper right", frameon=False, fontsize=10.5)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    ax.set_title("The features encode the phone far better than TB", loc="left",
                 color=t["ink"], fontsize=13, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(path, format="svg", facecolor=t["surface"])
    plt.close(fig)


def main():
    os.makedirs(FIGURES, exist_ok=True)
    for mode, t in THEMES.items():
        with open(os.path.join(FIGURES, f"architecture-{mode}.svg"), "w", encoding="utf-8") as f:
            f.write(architecture_svg(t))
        baseline_svg(t, os.path.join(FIGURES, f"baseline-auroc-{mode}.svg"))
        probe_svg(t, os.path.join(FIGURES, f"probe-{mode}.svg"))
    print("Figures written to", FIGURES)


if __name__ == "__main__":
    main()
