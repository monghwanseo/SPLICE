from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PAPER_FIG = ROOT / "paper" / "figures"
PAPER_FIG_PDF = PAPER_FIG / "pdf"
PAPER_FIG.mkdir(parents=True, exist_ok=True)
PAPER_FIG_PDF.mkdir(parents=True, exist_ok=True)

COLOR = {
    "usdt":      "#1f4e79",
    "usdc":      "#6e8b3d",
    "stress":    "#a4161a",
    "iv":        "#a4161a",
    "rigobon":   "#6f1d1b",
    "neutral":   "#2d2d2d",
    "highlight": "#d18700",
    "muted":     "#888888",
    "grid":      "#cccccc",
    "fill_pos":  "#bce4d8",
    "fill_neg":  "#f0c8c8",
}

EVENTS = [
    ("2021-12-05", "Structural\nbreak"),
    ("2022-05-12", "LUNA"),
    ("2022-11-08", "FTX"),
    ("2023-03-10", "USDC/SVB"),
    ("2024-01-10", "BTC ETF"),
    ("2024-08-05", "Yen carry"),
]

def setup():
    plt.rcdefaults()
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.06,
        "figure.constrained_layout.use": True,

        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "legend.title_fontsize": 9,

        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.edgecolor": COLOR["neutral"],
        "axes.labelcolor": COLOR["neutral"],
        "xtick.color": COLOR["neutral"],
        "ytick.color": COLOR["neutral"],
        "axes.grid": True,
        "grid.color": COLOR["grid"],
        "grid.alpha": 0.4,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,

        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "patch.linewidth": 0.8,

        "legend.frameon": False,
        "legend.borderpad": 0.4,
        "legend.handlelength": 1.6,
    })

def caption(fig, text, fontsize=9.5, y=0.04):
    fig.text(0.5, y, text, ha="center", va="center",
             fontsize=fontsize, color=COLOR["neutral"])

def make_fig(figsize, n_panels=1, layout="single", bottom=0.26):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=figsize, constrained_layout=False)
    if layout == "single":
        ax = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(left=0.12, right=0.96, top=0.95, bottom=bottom)
        return fig, ax
    if layout == "vertical":
        axes = [fig.add_subplot(n_panels, 1, i + 1) for i in range(n_panels)]
        fig.subplots_adjust(left=0.12, right=0.96, top=0.95,
                            bottom=bottom, hspace=0.32)
        for a in axes[:-1]:
            a.sharex(axes[-1])
            plt.setp(a.get_xticklabels(), visible=False)
        return fig, axes
    if layout == "horizontal":
        axes = [fig.add_subplot(1, n_panels, i + 1) for i in range(n_panels)]
        fig.subplots_adjust(left=0.10, right=0.97, top=0.95,
                            bottom=bottom, wspace=0.32)
        return fig, axes
    raise ValueError(f"Unknown layout: {layout}")

def panel_label_below(ax, text, color, fontsize=10.5):
    ax.set_title(text, loc="center", y=-0.20, fontsize=fontsize,
                 fontweight="bold", color=color, pad=2)

def subcaption(ax, panel, text, y=-0.27, fontsize=9.5):
    rendered = f"({panel}) {text}" if panel else text
    ax.text(0.5, y, rendered,
            transform=ax.transAxes, ha="center", va="top",
            fontsize=fontsize, color="#000000")

def tighten_y_positive(ax, vmin=0.0, headroom=0.18):
    ymin, ymax = ax.get_ylim()
    if ymin >= -1e-9:
        ax.set_ylim(vmin, ymax * (1.0 + headroom) if ymax > 0 else 1.0)

def save(fig, stem):
    setup_called = mpl.rcParams.get("font.family") in (["serif"], "serif")
    if not setup_called:
        setup()
    png_path = PAPER_FIG / f"{stem}.png"
    pdf_path = PAPER_FIG_PDF / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    print(f"  Saved {png_path.relative_to(ROOT)}")
    print(f"  Saved {pdf_path.relative_to(ROOT)}")
    return png_path
