"""One UI for both toolboxes: a Continuous | Discrete switch above the chosen app.

Switching swaps the displayed panel in place rather than rebuilding it, so each
app keeps its data, fitted table, and plots while the other one is showing.
"""
import matplotlib as mpl
import ipywidgets as widgets
from IPython.display import display

# discrete_helper sets figure.dpi at import time, which would otherwise leak into
# continuous plots in the shared kernel. Record each app's value around the
# imports and re-apply it on every switch.
from . import continuous_helper as _cont   # imports pyplot, so the backend's rc is active
_DPI = {"Continuous": mpl.rcParams["figure.dpi"]}
from . import discrete_helper as _disc      # side effect: figure.dpi = 120
_DPI["Discrete"] = mpl.rcParams["figure.dpi"]


def _continuous_formulas():
    from . import continuous_dist_list
    continuous_dist_list.display_formula()


def _discrete_formulas():
    from . import discrete_dist_list        # sets rcParams on import; contained by rc_context
    discrete_dist_list.display_discrete_formulas()


def _formula_panel(render):
    """Collapsed accordion that renders its formula sheet the first time it opens."""
    out = widgets.Output()
    acc = widgets.Accordion(children=[out], selected_index=None)
    acc.set_title(0, "Formula sheet")
    rendered = False

    def _on_open(change):
        nonlocal rendered
        if change["new"] == 0 and not rendered:
            rendered = True
            with out, mpl.rc_context():
                render()

    acc.observe(_on_open, names="selected_index")
    return acc


_PANELS = {
    "Continuous": widgets.VBox([_cont._ui, _formula_panel(_continuous_formulas)]),
    "Discrete": widgets.VBox([_disc.ui, _formula_panel(_discrete_formulas)]),
}

mode_toggle = widgets.ToggleButtons(
    options=[("Continuous  ·  18 distributions", "Continuous"),
             ("Discrete  ·  5 distributions", "Discrete")],
    value="Continuous",
    style={"button_width": "230px"},
)
_body = widgets.Box()


def _show(mode):
    mpl.rcParams["figure.dpi"] = _DPI[mode]
    _body.children = [_PANELS[mode]]


mode_toggle.observe(lambda change: _show(change["new"]), names="value")
_show(mode_toggle.value)

app = widgets.VBox([mode_toggle, _body])


def main(mode="Continuous"):
    """Display the combined app, opening on `mode` ("Continuous" or "Discrete").

    The Run chart is interactive - pan and zoom with its toolbar - while every other
    figure is a static image. Both apps share one matplotlib backend, so it is set up
    once here.
    """
    _cont.setup_figures()
    mode_toggle.value = mode
    _show(mode)
    display(app)
