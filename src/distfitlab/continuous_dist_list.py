# Helper cell: formulas + renderer (single matplotlib plot, no subplots/colors)
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
# add near top of file if not already present
import os
from IPython.display import Image, display


def render_pdf_formulas_image(
    save_path="artifacts/distribution_pdfs.png",
    title="Analytical PDFs (by dropdown order)"
):
    """
    Renders a single image with LaTeX-like PDF formulas in your dropdown order.
    Saves to `save_path` (directory will be created if missing) and returns the path.
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    items = get_pdf_formulas()
    n = len(items)

    # Spacing & sizing (a bit more generous to avoid tight_layout warnings)
    line_h = 0.050                # vertical step per distribution
    support_extra = 0.60           # extra fraction when also printing 'support'
    top_pad = 0.08                 # space for title
    bottom_pad = 0.03

    # Estimate height: each row often has 1.0 + support_extra lines on average
    est_lines = 0
    for _, _, sup in items:
        est_lines += 1.0 + (support_extra if sup else 0.0)

    fig_h = max(22.0, 1.0 + top_pad + bottom_pad + line_h * (est_lines + 2))
    fig_w = 12.0

    # Use constrained layout instead of tight_layout
    # built without pyplot: under an interactive backend a pyplot figure becomes a live
    # canvas widget and gets displayed, which put a stray figure outside the panel
    fig = Figure(figsize=(fig_w, fig_h), constrained_layout=True)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_axis_off()

    # Title
    y = 1.0 - top_pad * 0.6
    ax.text(0.02, y, title, fontsize=18, fontweight='bold',
            transform=ax.transAxes, va='top')
    y -= top_pad

    # Lines
    for name, pdf, sup in items:
        ax.text(0.02, y, f"{name}:", fontsize=14, transform=ax.transAxes, va='top')
        ax.text(0.18, y, f"$ {pdf} $", fontsize=14, transform=ax.transAxes, va='top')
        if sup:
            ax.text(0.18, y - line_h*0.75, f"support: $ {sup} $",
                    fontsize=12, transform=ax.transAxes, va='top')
            y -= line_h * (1.0 + support_extra)
        else:
            y -= line_h

    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    # nothing to close: the figure was never registered with pyplot
    return save_path

def get_pdf_formulas():
    """
    Returns a list of (name, latex_pdf, latex_support) in the *same order*
    as your dropdown (excluding 'External'). Parameter names match your UI:
    - Normal uses (μ, σ)
    - Others use loc, plus the parameters shown in your notebook (λ, α, θ, k, s, scale, c, μ, ξ, b, ν, m, σ, etc.)
    """
    items = [
        ("Normal",
         r"f(x)=\frac{1}{\sigma\sqrt{2\pi}}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)",
         r"x\in\mathbb{R}"),
        ("Exponential",
         r"f(x)=\lambda\,\exp\!\left(-\lambda\,(x-\mathrm{loc})\right)",
         r"x\geq \mathrm{loc}"),
        ("Gamma",
         r"f(x)=\frac{(x-\mathrm{loc})^{\alpha-1}}{\Gamma(\alpha)\,\theta^{\alpha}}\exp\!\left(-\frac{x-\mathrm{loc}}{\theta}\right)",
         r"x\geq \mathrm{loc}"),
        ("Rayleigh",
         r"f(x)=\frac{x-\mathrm{loc}}{\sigma^{2}}\exp\!\left(-\frac{(x-\mathrm{loc})^{2}}{2\sigma^{2}}\right)",
         r"x\geq \mathrm{loc}"),
        ("Weibull",
         r"f(x)=\frac{k}{\lambda}\left(\frac{x-\mathrm{loc}}{\lambda}\right)^{k-1}\exp\!\left(-\left(\frac{x-\mathrm{loc}}{\lambda}\right)^{k}\right)",
         r"x\geq \mathrm{loc}"),
        ("Lognormal",
         r"f(x)=\frac{1}{(x-\mathrm{loc})\,s\sqrt{2\pi}}\exp\!\left(-\frac{\left[\ln\!\left(\frac{x-\mathrm{loc}}{\mathrm{scale}}\right)\right]^{2}}{2s^{2}}\right)",
         r"x> \mathrm{loc}"),
        ("Loglogistic",
         r"f(x)=\frac{c}{\mathrm{scale}}\frac{\left(\frac{x-\mathrm{loc}}{\mathrm{scale}}\right)^{\,c-1}}{\left(1+\left(\frac{x-\mathrm{loc}}{\mathrm{scale}}\right)^{c}\right)^{2}}",
         r"x> \mathrm{loc}"),
        ("Inverse Gaussian",
         r"f(x)=\frac{1}{\mathrm{scale}}\frac{1}{\sqrt{2\pi\,y^{3}}}\exp\!\left(-\frac{(y-\mu)^{2}}{2\mu^{2}y}\right),\ y=\frac{x-\mathrm{loc}}{\mathrm{scale}}",
         r"x> \mathrm{loc},\ \mu>0"),
        ("Beta",
         r"f(x)=\frac{1}{B(\alpha,\beta)\,\mathrm{scale}}\left(\frac{x-\mathrm{loc}}{\mathrm{scale}}\right)^{\alpha-1}\left(1-\frac{x-\mathrm{loc}}{\mathrm{scale}}\right)^{\beta-1}",
         r"\mathrm{loc}<x<\mathrm{loc}+\mathrm{scale}"),
        ("GEV",
         r"f(x)=\frac{1}{\mathrm{scale}}\,t^{-1-1/\xi}\exp\!\left(-t^{-1/\xi}\right),\ t=1+\xi\,\frac{x-\mathrm{loc}}{\mathrm{scale}}",
         r"t>0"),
        ("Logistic",
         r"f(x)=\frac{\exp\!\left(-\frac{x-\mathrm{loc}}{s}\right)}{s\left(1+\exp\!\left(-\frac{x-\mathrm{loc}}{s}\right)\right)^{2}}",
         r"x\in\mathbb{R}"),
        ("Laplace",
         r"f(x)=\frac{1}{2b}\exp\!\left(-\frac{|x-\mathrm{loc}|}{b}\right)",
         r"x\in\mathbb{R}"),
        ("Chi-squared",
         r"f(x)=\frac{1}{\mathrm{scale}\,2^{\nu/2}\Gamma(\nu/2)}\,y^{\nu/2-1}\,e^{-y/2},\ y=\frac{x-\mathrm{loc}}{\mathrm{scale}}",
         r"x\geq \mathrm{loc},\ \nu>0"),
        ("Chi",
         r"f(x)=\frac{1}{\mathrm{scale}}\,\frac{2^{\,1-\nu/2}}{\Gamma(\nu/2)}\,y^{\nu-1}\,e^{-y^{2}/2},\ y=\frac{x-\mathrm{loc}}{\mathrm{scale}}",
         r"x\geq \mathrm{loc},\ \nu>0"),
        ("Nakagami",
         r"f(x)=\frac{1}{\mathrm{scale}}\,\frac{2\,m^{m}}{\Gamma(m)}\,y^{2m-1}\,e^{-m\,y^{2}},\ y=\frac{x-\mathrm{loc}}{\mathrm{scale}}",
         r"x\geq \mathrm{loc},\ m\geq \frac{1}{2}"),
        ("Rician",
         r"f(x)=\frac{y}{\sigma^{2}}\exp\!\left(-\frac{y^{2}+\nu^{2}}{2\sigma^{2}}\right)\,\mathrm{I}_{0}\!\left(\frac{y\,\nu}{\sigma^{2}}\right),\ y=x-\mathrm{loc}",
         r"x\geq \mathrm{loc},\ \nu\geq 0,\ \sigma>0"),
        ("Cauchy",
         r"f(x)=\frac{1}{\pi\,s}\,\frac{1}{1+\left(\frac{x-\mathrm{loc}}{s}\right)^{2}}",
         r"x\in\mathbb{R}"),
        ("Student-T",
         r"f(x)=\frac{\Gamma\!\left(\frac{\nu+1}{2}\right)}{\Gamma\!\left(\frac{\nu}{2}\right)\sqrt{\nu\pi}\,s}\left(1+\frac{(x-\mathrm{loc})^{2}}{\nu\,s^{2}}\right)^{-\frac{\nu+1}{2}}",
         r"x\in\mathbb{R},\ \nu>0"),
    ]
    return items

def display_formula():
    """
    Convenience display that saves to a local artifact path and shows it.
    """
    out_path = render_pdf_formulas_image("artifacts/distribution_pdfs.png")
    display(Image(filename=out_path))

if __name__ == "__main__":
    display_formula()