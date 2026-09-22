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


def test_event_probabilities_under_a_model():
    import scipy.stats as st
    from distfitlab.probability import model_probability
    cdf = lambda t: st.poisson.cdf(t, 4.0)
    mp = lambda op, a, b=None: model_probability(cdf, op, a, b, discrete=True)
    assert np.isclose(mp("<=", 2), st.poisson.cdf(2, 4))
    assert np.isclose(mp("<", 2), st.poisson.cdf(1, 4))            # strict bound drops k = 2
    assert np.isclose(mp(">", 2), st.poisson.sf(2, 4))
    assert np.isclose(mp(">=", 2), st.poisson.sf(1, 4))
    assert np.isclose(mp("==", 2), st.poisson.pmf(2, 4))
    assert np.isclose(mp("between", 3, 1), st.poisson.cdf(3, 4) - st.poisson.cdf(0, 4))
    assert mp("<", 0) == 0.0                                        # nothing below zero counts
    zip_cdf = lambda t: dh.theory_cdf("Zero-Inflated Poisson", (0.3, 4.0), t)
    assert model_probability(zip_cdf, "<", 0, discrete=True) == 0.0  # ZIP's CDF formula is not 0 at -1
    norm = lambda t: st.norm.cdf(t)
    assert np.isclose(model_probability(norm, ">", 1.96), 0.025, atol=1e-3)
    assert model_probability(norm, "==", 0.0) == 0.0


def test_wilson_interval_and_conditional_summary():
    from distfitlab.probability import wilson_interval, conditional_summary
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and np.isclose(hi, 0.2775, atol=1e-3)
    x_all = np.arange(1, 101)                  # 1..100
    x_cond = np.arange(51, 101)                # the condition keeps 51..100
    r = conditional_summary(x_cond, x_all, ">", 90)
    assert (r["k"], r["n"], r["k_all"], r["n_all"]) == (10, 50, 10, 100)
    assert np.isclose(r["p"], 0.2) and np.isclose(r["p_all"], 0.1) and np.isclose(r["ratio"], 2.0)
    assert np.isclose(r["p_cond_given_e"], 1.0)                     # every value > 90 is in the condition
    assert np.isclose(r["p_cond"], 0.5)                             # the condition keeps 50 of 100 rows
    # Bayes' rule closes exactly on the plug-in counts
    assert np.isclose(r["p_cond_given_e"] * r["p_all"], r["p"] * r["p_cond"])
    r_trim = conditional_summary(x_cond, x_all, ">", 90, subset=False)
    assert np.isnan(r_trim["p_cond_given_e"]) and np.isnan(r_trim["p_cond"])


def test_conditional_probability_button_in_both_apps():
    rng = np.random.default_rng(8)
    season = rng.choice(["summer", "winter"], 2000)
    counts = np.where(season == "summer", rng.poisson(9, 2000), rng.poisson(4, 2000))
    df = pd.DataFrame({"season": season, "count": counts})
    summer = df.loc[df.season == "summer", "count"]
    for app in (dh, ch):
        app._use_dataframe(df, origin="path") if app is ch else _load_discrete(df)
        app.mode_dd.value = "file"; app.col_dd.value = "count"
        app._clear_filters()
        app.filter_col.value, app.filter_op.value, app.filter_val1.value = "season", "==", "summer"
        app._add_filter()
        app.prob_op.value, app.prob_v1.value = ">", 6
        app._on_prob()
        html = app.prob_out.value
        assert f"{(summer > 6).mean():.3f}" in html                     # P(E | season == summer)
        assert f"{(df['count'] > 6).mean():.3f}" in html                # P(E) over all rows
        both = ((df["count"] > 6) & (df.season == "summer")).sum() / (df["count"] > 6).sum()
        assert f"{both:.3f}" in html                                    # P(condition | E), Bayes
        assert "season == summer" in html
        app._clear_filters(); app.mode_dd.value = "sim"


def _load_discrete(df):
    """What the Discrete app's upload handler does, minus reading the file."""
    dh.uploader.df_raw = df.copy(); dh.uploader.df = df
    dh.filter_col.options = list(df.columns)
    dh.group_col.options = ["(none)"] + list(df.columns); dh.group_col.value = "(none)"
    dh._refresh_columns()


def test_formula_box_follows_filters_and_event():
    rng = np.random.default_rng(9)
    df = pd.DataFrame({"season": rng.choice(["winter", "summer"], 300),
                       "year": rng.integers(0, 2, 300), "count": rng.poisson(5, 300)})
    _load_discrete(df)
    dh.mode_dd.value = "file"; dh.col_dd.value = "count"; dh._clear_filters()
    dh.prob_op.value, dh.prob_v1.value = ">", 7
    assert "P( count &gt; 7 )" in dh.prob_formula.value                  # no condition yet
    for col, val in (("season", "winter"), ("year", "1")):
        dh.filter_col.value, dh.filter_op.value, dh.filter_val1.value = col, "==", val
        dh._add_filter()
    f = dh.prob_formula.value
    assert "P( count &gt; 7 | season == winter and year == 1 )" in f
    assert "= P( count &gt; 7 and season == winter and year == 1 ) / P( season == winter and year == 1 )" in f
    dh.prob_op.value, dh.prob_v2.value = "between", 3                   # event edits update it too
    assert "3 ≤ count ≤ 7 | season" in dh.prob_formula.value
    dh._clear_filters(); dh.mode_dd.value = "sim"


def test_groups_follow_natural_order():
    from distfitlab.grouping import natural_key, values_in_order
    assert sorted(["10", "2", "1", "12", "-3", "2.5"], key=natural_key) == ["-3", "1", "2", "2.5", "10", "12"]
    assert sorted(["press-10", "press-2", "Press-1"], key=natural_key) == ["Press-1", "press-2", "press-10"]
    assert sorted(["summer", "3", "fall"], key=natural_key) == ["3", "fall", "summer"]   # numbers first
    rng = np.random.default_rng(10)
    df = pd.DataFrame({"month": rng.integers(1, 13, 1200), "v": rng.gamma(3, 1.0, 1200)})
    assert values_in_order(df["month"]) == [str(m) for m in range(1, 13)]
    ch._use_dataframe(df, origin="path"); ch.mode_dd.value = "file"; ch.col_dd.value = "v"
    ch.group_col.value = "month"; ch.viz_split.value = True
    assert list(ch.group_val.options) == ["(all groups)"] + [str(m) for m in range(1, 13)]
    assert [label for label, _v in ch._data_by_group()] == [str(m) for m in range(1, 11)]   # first 10, in order
    ch.viz_split.value = False; ch.group_col.value = "(none)"; ch.mode_dd.value = "sim"


def test_empirical_pdf_uses_pdf_cdf(monkeypatch):
    shown = []
    monkeypatch.setattr(ch, "_emit", lambda fig, live=False: shown.append((fig, live)))
    ch.mode_dd.value = "sim"; ch.bins_int.value = 60; ch.seed_txt.value = "5"   # same sample twice
    ch._viz_epdf()
    fig, live = shown[0]
    assert live                                                   # zoomable, like the run charts
    line = fig.axes[0].lines[0]
    r, _ = ch._prepare_data()
    xs, dens = ch.pdf_cdf(r, bins=60, PDF=True)
    assert np.allclose(line.get_xdata(), xs) and np.allclose(line.get_ydata(), dens)
    ch.bins_int.value = 100; ch.seed_txt.value = ""


@pytest.mark.parametrize("helper, chart", [(ch, "_viz_ecdf"), (ch, "_viz_epdf"), (dh, "_viz_ecdf"), (dh, "_viz_bar")])
def test_distribution_charts_are_zoomable_with_grid(helper, chart, monkeypatch):
    import matplotlib.pyplot as plt
    shown = []
    monkeypatch.setattr(helper, "_emit", lambda fig, live=False: shown.append((fig, live)))
    helper.mode_dd.value = "sim"; helper.viz_grid.value = False
    getattr(helper, chart)()
    fig, live = shown[0]
    ax = fig.axes[0]
    gridlines = lambda: any(g.get_visible() for g in ax.get_xgridlines() + ax.get_ygridlines())
    assert live and not gridlines()
    helper.viz_grid.value = True                                  # toggles the chart on screen
    assert gridlines()
    helper.viz_grid.value = False
    plt.close("all")


@pytest.mark.parametrize("helper", [ch, dh])
def test_split_checkbox_names_the_group_column(helper):
    df = pd.DataFrame({"month": np.tile(np.arange(1, 13), 50), "count": np.arange(600) % 9})
    if helper is ch:
        ch._use_dataframe(df, origin="path")
    else:
        _load_discrete(df)
    helper.mode_dd.value = "file"
    helper.group_col.value = "(none)"
    assert helper.viz_split.disabled and helper.viz_split.description == "split by group"
    helper.group_col.value = "month"
    assert not helper.viz_split.disabled and helper.viz_split.description == "split by month"
    helper.mode_dd.value = "sim"                                   # simulated data has no groups
    assert helper.viz_split.disabled
    helper.group_col.value = "(none)"


@pytest.mark.parametrize("helper", [ch, dh])
@pytest.mark.parametrize("sort", [False, True])
def test_run_charts_split_by_group(helper, sort, monkeypatch):
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(11)
    df = pd.DataFrame({"shift": np.repeat(["day", "night", "late"], [300, 200, 100]),
                       "count": np.concatenate([rng.poisson(9, 300), rng.poisson(4, 200), rng.poisson(6, 100)])})
    ch._use_dataframe(df, origin="path") if helper is ch else _load_discrete(df)
    helper.mode_dd.value = "file"; helper.col_dd.value = "count"
    helper.group_col.value = "shift"; helper.viz_split.value = True
    shown = []
    monkeypatch.setattr(helper, "_emit", lambda fig, live=False: shown.append(fig))
    helper._viz_run(sort=sort)
    fig = shown[0]
    # sorted: one overlaid chart; run chart: one panel per group (its data line comes first)
    lines = fig.axes[0].lines if sort else [ax.lines[0] for ax in fig.axes]
    if not sort:
        assert len(fig.axes) == 3
    assert [l.get_label().split(" (")[0] for l in lines] == ["day", "late", "night"]   # one line per group, in order
    assert sorted(len(l.get_ydata()) for l in lines) == [100, 200, 300]
    if sort:
        for l in lines:
            x, y = l.get_xdata(), l.get_ydata()
            assert (np.diff(y) >= 0).all() and 0 < x.min() and x.max() < 100          # percentile rank axis
    helper.viz_split.value = False; helper.group_col.value = "(none)"; helper.mode_dd.value = "sim"
    plt.close("all")


@pytest.mark.parametrize("helper", [ch, dh])
def test_event_label_names_the_use_column(helper):
    df = pd.DataFrame({"count": np.arange(200) % 7, "rentals": np.arange(200) % 5, "g": ["a", "b"] * 100})
    ch._use_dataframe(df, origin="path") if helper is ch else _load_discrete(df)
    helper.mode_dd.value = "file"
    helper.col_dd.value = "count"
    assert "Event on count:" in helper.prob_label.value
    helper.col_dd.value = "rentals"
    assert "Event on rentals:" in helper.prob_label.value
    helper.mode_dd.value = "sim"
    assert "Event on the simulated sample:" in helper.prob_label.value
