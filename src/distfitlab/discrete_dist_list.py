# discrete_dist_list.py
# Renders a single PNG containing analytical PMFs (in dropdown order)
# and displays it in Jupyter. No external LaTeX required.

import os
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import rcParams
from IPython.display import Image, display

# Robust rendering: use mathtext (no external LaTeX needed)
rcParams["text.usetex"] = False
rcParams["mathtext.fontset"] = "dejavusans"
rcParams["figure.dpi"] = 150

# Keep this order aligned with your dropdown
ALL_DISTS = ("Poisson", "Binomial", "Geometric", "Negative Binomial", "Zero-Inflated Poisson")

def get_pmf_formulas():
    """
    Returns list of (name, latex_pmf, latex_support) in the SAME order as the dropdown.
    Uses mathtext-friendly syntax (no \\dfrac; only \\frac, etc.). Avoids \\text{...}.
    """
    items = [
        ("Poisson",
         r"f(k)=\frac{\lambda^{k} e^{-\lambda}}{k!}",
         r"k=0,1,2,\ldots"),
        ("Binomial",
         r"f(k)=\binom{n}{k}\,p^{k}(1-p)^{\,n-k}",
         r"k=0,1,\ldots,n"),
        ("Geometric",
         r"f(k)=(1-p)^{\,k-1}\,p",
         r"k=1,2,\ldots"),
        ("Negative Binomial",
         r"f(k)=\binom{k+r-1}{k}\,(1-p)^{\,k}\,p^{\,r}",
         r"k=0,1,2,\ldots"),
        ("Zero-Inflated Poisson",
         # NOTE: use \geq (NOT \ge)
         r"f(0)=\pi+(1-\pi)\,e^{-\lambda},\quad f(k)=(1-\pi)\,\frac{\lambda^{k} e^{-\lambda}}{k!},\ k\geq 1",
         r"k=0,1,2,\ldots"),
    ]
    # (Optional) sanity check to ensure we didn't reorder accidentally:
    names = [n for n,_,_ in items]
    assert tuple(names) == ALL_DISTS, f"Order mismatch: {names} != {ALL_DISTS}"
    return items

def render_pmf_formulas_image(
    save_path="artifacts/discrete_distribution_pmfs.png",
    title="Discrete PMFs (by dropdown order)",
    *,
    fontsize=12,
    support_fontsize=10,
    title_size=18,
    left_name=0.02,
    left_formula=0.20,
    line_h=0.05,
    support_off=0.03,
    top=0.96,
    title_gap=0.06,
    bottom_margin=0.05,
    dpi=200,
    debug=False,
):
    """
    Renders a single PNG with the PMF formulas and returns the path.
    The layout uses figure-level text, so no subplots/tight_layout issues.
    """
    folder = os.path.dirname(save_path) or "."
    os.makedirs(folder, exist_ok=True)

    items = get_pmf_formulas()
    blocks = sum(2 if sup else 1 for _,_,sup in items)

    # Choose a generous height to avoid layout warnings
    fig_w = 14
    fig_h = max(6.8, (title_gap + blocks*line_h + bottom_margin) * 18.5)

    if debug:
        import matplotlib
        print(f"matplotlib {matplotlib.__version__}, usetex={rcParams.get('text.usetex')}")
        print(f"Saving → {os.path.abspath(save_path)}")
        print(f"Figure size: {fig_w} x {fig_h} in")

    # built without pyplot: see the note in continuous_dist_list
    fig = Figure(figsize=(fig_w, fig_h), facecolor="white")
    FigureCanvasAgg(fig)

    # Title
    fig.text(left_name, top, title, fontsize=title_size, weight='bold', va='top')

    # Content
    y = top - title_gap
    for name, pmf, sup in items:
        # Distribution name
        fig.text(left_name, y, f"{name}:", fontsize=fontsize, va='top')
        # PMF formula (mathtext)
        fig.text(left_formula, y, f"$ {pmf} $", fontsize=fontsize, va='top')
        # Support line
        if sup:
            fig.text(left_formula, y - support_off, f"support: $ {sup} $",
                     fontsize=support_fontsize, va='top')
            y -= line_h
        y -= line_h

    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    # nothing to close: the figure was never registered with pyplot
    return save_path

def display_discrete_formulas(save_path="artifacts/discrete_distribution_pmfs.png", **kwargs):
    """
    Convenience wrapper: renders and displays the image in a notebook cell.
    Returns the image path.
    """
    out_path = render_pmf_formulas_image(save_path=save_path, **kwargs)
    display(Image(filename=out_path))
    return out_path

if __name__ == "__main__":
    # Run standalone from terminal/python
    print(display_discrete_formulas(debug=True))
