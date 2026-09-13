"""Smoke tests: the package imports, both apps fit, and files of several formats load."""
import io

import matplotlib
matplotlib.use("Agg")                 # no display, no widget backend, during tests

import numpy as np
import pandas as pd
import pytest

import distfitlab
from distfitlab import continuous_helper as ch, discrete_helper as dh
from distfitlab.file_readers import read_uploaded_dataframe


def test_version_and_lazy_main():
    assert distfitlab.__version__
    assert callable(distfitlab.main)


def test_continuous_ranks_the_right_distribution():
    x = np.random.default_rng(0).normal(5, 2, 3000)
    res = ch.fit_subset_and_summarize(x, bins=100)
    assert len(res["table"]) == 18
    errors = res["table"]["Fitting Error"].astype(float).to_numpy()
    assert (np.diff(errors) >= 0).all(), "table must be sorted best first"
    assert res["best_error"] < 0.02
    assert res["table"]["Distribution"].iloc[0] == res["best_name"]


def test_continuous_loc_can_be_held_at_zero():
    x = 2.0 * np.random.default_rng(1).weibull(1.8, 2000)
    res = ch.fit_subset_and_summarize(x, bins=100, fix_loc=True)
    assert res["table"]["Parameter 1"].str.endswith("= 0 (fixed)").all()


def test_continuous_aic_bic_match_a_hand_calculation():
    import scipy.stats as st
    x = np.random.default_rng(5).normal(5, 2, 800)
    res = ch.fit_subset_and_summarize(x, bins=100)
    ll = st.norm.logpdf(x, *st.norm.fit(x)).sum()
    assert np.isclose(res["aic"]["Normal"], 2 * 2 - 2 * ll)
    assert np.isclose(res["bic"]["Normal"], 2 * np.log(len(x)) - 2 * ll)
    row = res["table"].set_index("Distribution").loc["Normal"]
    assert float(row["AIC"]) == pytest.approx(res["aic"]["Normal"], abs=0.05)
    assert res["best_aic_name"] in {"Normal", "Student-T"}      # T nests Normal as df grows


def test_continuous_aic_does_not_count_a_fixed_loc():
    import scipy.stats as st
    x = 2.0 * np.random.default_rng(6).weibull(1.8, 1000)
    res = ch.fit_subset_and_summarize(x, bins=100, subset=("Weibull",), fix_loc=True)
    p = st.weibull_min.fit(x, floc=0.0)
    ll = st.weibull_min.logpdf(x, *p).sum()
    assert np.isclose(res["aic"]["Weibull"], 2 * 2 - 2 * ll)     # shape + scale only


def test_discrete_aic_bic_and_out_of_support_dash():
    import scipy.stats as st
    counts = np.random.default_rng(7).poisson(3, 1500)            # has zeros
    res = dh.fit_all_discrete(counts)
    ll = st.poisson.logpmf(counts, counts.mean()).sum()
    assert np.isclose(res["aic"]["Poisson"], 2 * 1 - 2 * ll)
    assert np.isclose(res["bic"]["Poisson"], np.log(len(counts)) - 2 * ll)
    table = res["table"].set_index("Distribution")
    assert table.loc["Geometric", "AIC"] == "—"                 # Geometric support starts at 1
    assert res["best_aic_name"] in {"Poisson", "Negative Binomial", "Zero-Inflated Poisson"}


def test_theory_ppf_inverts_theory_cdf():
    q = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    for name, pars in [("Normal", [0.0, 1.0]), ("Gamma", [2.0, 0.0, 1.5]),
                       ("Weibull", [1.8, 0.0, 2.0]), ("Student-T", [8.0, 0.0, 1.0])]:
        back = ch.theory_cdf(name, pars, ch.theory_ppf(name, pars, q))
        assert np.allclose(back, q, atol=1e-9), name


def test_discrete_prefers_the_generating_law():
    counts = np.random.default_rng(2).poisson(4, 3000)
    res = dh.fit_all_discrete(counts)
    assert res["best_name"] in {"Poisson", "Negative Binomial"}   # NB nests Poisson
    assert res["best_error"] < 0.03


def test_discrete_rejects_non_integer_data():
    s = pd.Series(np.random.default_rng(3).normal(5, 1, 100))
    with pytest.raises(RuntimeError, match="integer"):
        dh._clean_counts(s)


@pytest.mark.parametrize("suffix", ["csv", "tsv", "json", "jsonl"])
def test_reader_round_trips(suffix):
    df = pd.DataFrame({"g": ["a", "b"] * 50, "k": np.arange(100)})
    buf = io.BytesIO()
    if suffix == "csv":
        df.to_csv(buf, index=False)
    elif suffix == "tsv":
        df.to_csv(buf, sep="\t", index=False)
    elif suffix == "json":
        buf.write(df.to_json(orient="records").encode())
    else:
        buf.write(df.to_json(orient="records", lines=True).encode())
    out = read_uploaded_dataframe(buf.getvalue(), f"data.{suffix}")
    assert list(out.columns) == ["g", "k"]
    assert len(out) == 100


@pytest.mark.parametrize("helper", [ch, dh])
def test_sorted_run_chart_plots_values_in_order(helper, monkeypatch):
    shown = []
    monkeypatch.setattr(helper, "_emit", lambda fig, live=False: shown.append(fig))
    helper.mode_dd.value = "sim"
    helper._viz_run(sort=True)
    y = shown[0].axes[0].lines[0].get_ydata()
    assert len(y) > 1 and (np.diff(y) >= 0).all()


@pytest.mark.parametrize("helper", [ch, dh])
def test_run_chart_grid_toggles_in_place(helper, monkeypatch):
    import matplotlib.pyplot as plt
    shown = []
    monkeypatch.setattr(helper, "_emit", lambda fig, live=False: shown.append(fig))
    helper.mode_dd.value = "sim"
    helper.viz_grid.value = False
    helper._viz_run()
    ax = shown[0].axes[0]
    gridlines = lambda: any(g.get_visible() for g in ax.get_xgridlines() + ax.get_ygridlines())
    assert not gridlines()
    helper.viz_grid.value = True                     # the already-drawn chart picks it up
    assert gridlines()
    helper.viz_grid.value = False
    assert not gridlines()
    plt.close("all")


def test_filter_stack_and_cleaning():
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"machine": rng.choice(["A", "B"], 400),
                       "v": rng.gamma(4, 1.2, 400)})
    ch._use_dataframe(df, origin="path")
    ch.mode_dd.value = "file"
    ch.col_dd.value = "v"
    ch._clear_filters()
    ch.filter_col.value, ch.filter_op.value, ch.filter_val1.value = "machine", "==", "A"
    ch._add_filter()
    assert len(ch._current_df()) == int((df["machine"] == "A").sum())
    ch._clear_filters()
    ch.mode_dd.value = "sim"
