# === Discrete DistFit Toolbox: Poisson, Binomial, Geometric, Negative Binomial, Zero-Inflated Poisson ===
import io, os, math, numpy as np, pandas as pd, matplotlib.pyplot as plt
import scipy.stats as st
from scipy.special import gammaln
import ipywidgets as widgets
from ipywidgets import GridspecLayout
from IPython.display import display, HTML, clear_output, FileLink, Image
from IPython import get_ipython

from .file_readers import read_uploaded_dataframe as _read_uploaded_dataframe

plt.rcParams["figure.dpi"] = 120

# ---------------- Empirical discrete PMF/CDF ----------------
def empirical_pmf_cdf(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise ValueError("No data.")
    if np.any(~np.isfinite(x)):
        x = x[np.isfinite(x)]
    # require integer counts
    xi = np.rint(x).astype(int)
    if np.any(np.abs(x - xi) > 1e-6):
        raise ValueError("External data must be integer-valued for discrete fitting.")
    if xi.min() < 0:
        raise ValueError("Counts must be nonnegative.")
    k, cnt = np.unique(xi, return_counts=True)
    pmf = cnt / cnt.sum()
    cdf = np.cumsum(pmf)
    return k, pmf, cdf

# Helper: extend support for plotting/tail comparison
def support_for_plot(x: np.ndarray, extra=25):
    xi = np.rint(np.asarray(x)).astype(int)
    kmin = max(0, xi.min())
    kmax = xi.max() + int(max(10, extra))
    return np.arange(kmin, kmax + 1, dtype=int)

# ---------------- Theory PMF/CDF ----------------
def theory_pmf(dist, pars, k):
    k = np.asarray(k, dtype=int)
    if dist == "Poisson":
        lam, = pars
        return st.poisson.pmf(k, mu=lam)
    elif dist == "Binomial":
        n, p = pars
        return st.binom.pmf(k, n=int(round(n)), p=p)
    elif dist == "Geometric":  # trials to first success (support 1..)
        p, = pars
        return st.geom.pmf(k, p=p)
    elif dist == "Negative Binomial":  # failures before r successes (support 0..)
        r, p = pars
        return st.nbinom.pmf(k, n=r, p=p)
    elif dist == "Zero-Inflated Poisson":
        pi, lam = pars
        base = st.poisson.pmf(k, mu=lam)
        pmf = (1.0 - pi) * base
        pmf = pmf.astype(float)
        # adjust mass at zero
        pmf = np.where(k == 0, pi + (1.0 - pi) * np.exp(-lam), pmf)
        return pmf
    else:
        raise ValueError("Unknown distribution.")

def theory_cdf(dist, pars, k):
    k = np.asarray(k, dtype=int)
    if dist == "Poisson":
        lam, = pars
        return st.poisson.cdf(k, mu=lam)
    elif dist == "Binomial":
        n, p = pars
        return st.binom.cdf(k, n=int(round(n)), p=p)
    elif dist == "Geometric":
        p, = pars
        return st.geom.cdf(k, p=p)
    elif dist == "Negative Binomial":
        r, p = pars
        return st.nbinom.cdf(k, n=r, p=p)
    elif dist == "Zero-Inflated Poisson":
        pi, lam = pars
        return pi + (1.0 - pi) * st.poisson.cdf(k, mu=lam)
    else:
        raise ValueError("Unknown distribution.")

# ---------------- Manual Fits (closed-form / small search) ----------------
def fit_poisson(x):
    lam = float(np.mean(x))
    lam = max(lam, 1e-9)
    return (lam,)

def fit_binomial(x):
    # Small integer search for n >= max(x) .. max(x)+20, p = mean/n
    xi = np.rint(np.asarray(x)).astype(int)
    mx = int(xi.max())
    mean = float(xi.mean())
    best = None
    k_emp, _, F_emp = empirical_pmf_cdf(xi)
    # extend support for tail error
    k_sup = np.arange(0, mx + max(10, int(np.sqrt(mean)*6)) + 1, dtype=int)
    for n in range(max(1, mx), mx + 21):
        p = mean / n if n > 0 else 0.5
        p = min(max(p, 1e-9), 1 - 1e-9)
        F_th = theory_cdf("Binomial", (n, p), k_sup)
        # empirical cdf at k_sup (carry last value)
        F_emp_sup = np.interp(k_sup, k_emp, F_emp, left=0.0, right=1.0)
        err = float(np.max(np.abs(F_th - F_emp_sup)))
        if best is None or err < best[0]:
            best = (err, (float(n), float(p)))
    return best[1]

def fit_geometric(x):
    # Geometric (trials to first success) requires min >= 1
    xi = np.rint(np.asarray(x)).astype(int)
    if xi.min() < 1:
        # Shift zeros up for a fair attempt
        xi = xi + (1 - xi.min())
    mean = float(np.mean(xi))
    p = 1.0 / max(mean, 1 + 1e-9)
    p = min(max(p, 1e-9), 1 - 1e-9)
    return (p,)

from scipy.special import gammaln
import numpy as np, math
import scipy.optimize as opt

def fit_nbinom(x):
    """
    Stable NB fit via MLE:
      - NB parameterization used by SciPy: (r, p) with support k=0,1,...
      - mean = r*(1-p)/p, var = r*(1-p)/p**2 = mean + mean^2 / r
    Uses unconstrained variables: log r, logit p to avoid boundary issues.
    Falls back to Poisson-like when variance ≈ mean.
    """
    x = np.asarray(x, dtype=float)
    m = float(np.mean(x))
    v = float(np.var(x, ddof=0))

    # Near-Poisson guard: don't try to estimate dispersion when there isn't any.
    if v <= m * 1.02:  # within ~2% of Poisson variance
        r = 1e6
        p = r / (r + max(m, 1e-9))
        return (r, p)

    # Method-of-moments starts
    r0 = max(1e-6, (m*m) / (v - m))
    p0 = r0 / (r0 + m)
    # transform to unconstrained
    def to_unconstrained(r, p):
        return np.array([math.log(r), math.log(p/(1.0 - p))])

    def from_unconstrained(theta):
        log_r, logit_p = theta
        r = math.exp(log_r)
        p = 1.0 / (1.0 + math.exp(-logit_p))
        # clamp very slightly away from boundaries
        r = min(max(r, 1e-6), 1e9)
        p = min(max(p, 1e-9), 1.0 - 1e-9)
        return r, p

    # Negative log-likelihood for NB
    xi = np.rint(x).astype(int)
    logfac = gammaln(xi + 1.0)
    def nll(theta):
        r, p = from_unconstrained(theta)
        # log pmf: Γ(k+r) - Γ(r) - Γ(k+1) + r log p + k log(1-p)
        return -np.sum(
            gammaln(xi + r) - gammaln(r) - logfac
            + r*np.log(p) + xi*np.log1p(-p)
        )

    theta0 = to_unconstrained(r0, p0)
    res = opt.minimize(nll, theta0, method="L-BFGS-B")
    if not res.success:
        # fallback to MOM
        return (r0, p0)

    r, p = from_unconstrained(res.x)
    return (float(r), float(p))


def fit_zip(x):
    # Simple robust MLE via coarse grid on (pi, lam) using log-likelihood
    xi = np.rint(np.asarray(x)).astype(int)
    n = len(xi)
    zfrac = (xi == 0).mean()
    mu = float(np.mean(xi))
    # Grids
    pi_grid = np.linspace(0.0, min(0.98, max(zfrac + 0.2, 0.98)), 49)
    lam_hi = max(10.0, 5.0 * mu + 10.0)
    lam_grid = np.linspace(max(1e-6, mu*0.2), lam_hi, 100)
    # Precompute log(x!) via gammaln
    logfac = gammaln(xi + 1.0)
    best = None
    for pi in pi_grid:
        for lam in lam_grid:
            # log-likelihood
            # xi == 0: log(pi + (1-pi)*e^{-lam})
            # xi > 0:  log(1-pi) + x*log(lam) - lam - log(x!)
            e_mlam = math.exp(-lam)
            term0 = np.log(pi + (1.0 - pi) * e_mlam + 1e-300)
            mask0 = (xi == 0)
            maskp = ~mask0
            ll = 0.0
            if mask0.any():
                ll += term0 * mask0.sum()
            if maskp.any():
                xpos = xi[maskp]
                ll += (math.log(1.0 - pi + 1e-300) + xpos * math.log(lam + 1e-300) - lam).sum() - logfac[maskp].sum()
            if (best is None) or (ll > best[0]):
                best = (ll, (float(pi), float(lam)))
    return best[1]

# ---------------- Generate synthetic data ----------------
def gen_data(source, N, P, seed=None):
    if seed not in (None, ""):
        np.random.seed(int(seed))
    if source == "Poisson":
        lam, = P; return np.random.poisson(lam, size=N)
    elif source == "Binomial":
        n, p = P; return np.random.binomial(int(round(n)), p, size=N)
    elif source == "Geometric":  # trials to first success (1..)
        p, = P; return np.random.geometric(p, size=N)
    elif source == "Negative Binomial":
        r, p = P; return st.nbinom.rvs(n=r, p=p, size=N)
    elif source == "Zero-Inflated Poisson":
        pi, lam = P
        z = (np.random.rand(N) < pi).astype(int)  # 1 means forced zero
        base = np.random.poisson(lam, size=N)
        return np.where(z == 1, 0, base)
    elif source == "External":
        raise RuntimeError("Use the uploader for External.")
    else:
        raise ValueError("Unknown source.")

# ---------------- Fit-all & summarize ----------------
ALL_DISTS = ("Poisson","Binomial","Geometric","Negative Binomial","Zero-Inflated Poisson")

def fit_all_discrete(x, subset=ALL_DISTS):
    k_emp, pmf_emp, F_emp = empirical_pmf_cdf(x)
    # Build a common support extended for tail checking
    k_sup = support_for_plot(x, extra=25)
    F_emp_sup = np.interp(k_sup, k_emp, F_emp, left=0.0, right=1.0)
    fitted = {}
    errors = {}
    for d in subset:
        if d == "Poisson":
            pars = fit_poisson(x)
        elif d == "Binomial":
            pars = fit_binomial(x)
        elif d == "Geometric":
            pars = fit_geometric(x)
        elif d == "Negative Binomial":
            pars = fit_nbinom(x)
        elif d == "Zero-Inflated Poisson":
            pars = fit_zip(x)
        else:
            continue
        F_th = theory_cdf(d, pars, k_sup)
        err = float(np.max(np.abs(F_th - F_emp_sup)))
        fitted[d] = pars
        errors[d] = err
    # table
    rows = []
    for d in subset:
        if d not in fitted: 
            continue
        pars = fitted[d]
        if d == "Poisson":
            ptxt = f"λ = {pars[0]:.6f}"
        elif d == "Binomial":
            ptxt = f"n = {int(round(pars[0]))}, p = {pars[1]:.6f}"
        elif d == "Geometric":
            ptxt = f"p = {pars[0]:.6f} (support 1,2,...)"
        elif d == "Negative Binomial":
            ptxt = f"r = {pars[0]:.6f}, p = {pars[1]:.6f} (support 0,1,...)"
        elif d == "Zero-Inflated Poisson":
            ptxt = f"π = {pars[0]:.6f}, λ = {pars[1]:.6f}"
        else:
            ptxt = str(pars)
        rows.append({"Distribution": d, "Fitted parameters": ptxt, "Max CDF error": f"{errors[d]:.3f}"})
    # Best fit first: order the table, the error chart and the saved CSV by ascending error (NaN last)
    errors = dict(sorted(errors.items(), key=lambda kv: (np.isnan(kv[1]), kv[1])))
    rank = {d: i for i, d in enumerate(errors)}
    df = pd.DataFrame(rows).sort_values("Distribution", key=lambda s: s.map(rank), kind="stable", ignore_index=True)
    # best
    best_name = min(errors, key=errors.get)
    return {
        "k_emp": k_emp, "pmf_emp": pmf_emp, "F_emp": F_emp,
        "k_sup": k_sup, "fitted": fitted, "errors": errors,
        "table": df, "best_name": best_name, "best_error": float(errors[best_name])
    }

# ---------------- Widgets: controls ----------------
# Where the data comes from; the distribution list is only used for simulated data
mode_dd   = widgets.Dropdown(options=[("Simulated Random Data", "sim"), ("From file", "file")], value="sim",
                             description="Data Source:", layout=widgets.Layout(width="340px"))
source_dd = widgets.Dropdown(
    options=list(ALL_DISTS),
    value="Poisson", description="Distribution:", layout=widgets.Layout(width="260px")
)

def _is_external():
    return mode_dd.value == "file"
N_int     = widgets.BoundedIntText(value=2000, min=50, max=2_000_000, step=50,
                                   description="N:", layout=widgets.Layout(width="160px"))
seed_txt  = widgets.Text(value="", description="Seed:", layout=widgets.Layout(width="160px"))

# Params
poi_lam = widgets.FloatText(value=3.0, description="", layout=widgets.Layout(width="120px"))
bin_n   = widgets.BoundedIntText(value=20, min=1, max=10_000, description="", layout=widgets.Layout(width="120px"))
bin_p   = widgets.FloatText(value=0.3, description="", layout=widgets.Layout(width="120px"))
geo_p   = widgets.FloatText(value=0.2, description="", layout=widgets.Layout(width="120px"))
nb_r    = widgets.FloatText(value=5.0, description="", layout=widgets.Layout(width="120px"))
nb_p    = widgets.FloatText(value=0.6, description="", layout=widgets.Layout(width="120px"))
zip_pi  = widgets.FloatText(value=0.3, description="", layout=widgets.Layout(width="120px"))
zip_lam = widgets.FloatText(value=2.0, description="", layout=widgets.Layout(width="120px"))

# External upload (CSV/TSV/TXT/Excel)
uploader = widgets.FileUpload(
    accept=".csv,.tsv,.psv,"
           ".xlsx,.xls,"
           ".json,.jsonl,.ndjson,"
           ".parquet,.feather,.arrow,"
           ".sav,.dta,.sas7bdat,.xpt,"
           ".h5,.hdf5,"
           ".txt",
    multiple=False,
    description="Upload CSV/XLSX/Parquet/...",
    tooltip="CSV, TSV, PSV, TXT, Excel, JSON/JSONL, Parquet, Feather/Arrow, "
            "SPSS, Stata, SAS, HDF5")
uploader.layout.width = "340px"
col_dd   = widgets.Dropdown(options=[], layout=widgets.Layout(width="280px"))   # labelled by "Use column:" in the row

# File-data shaping: a stack of conditions, cleaning, group-by
filter_col   = widgets.Dropdown(options=[], layout=widgets.Layout(width="170px"))
filter_op    = widgets.Dropdown(options=["==","!=",">",">=","<","<=","between","in","contains","regex"],
                                value="==", layout=widgets.Layout(width="115px"))
filter_val1  = widgets.Text(placeholder="value / list / pattern", layout=widgets.Layout(width="175px"))
filter_val2  = widgets.Text(placeholder="upper (for between)", layout=widgets.Layout(width="140px"))
filter_ci    = widgets.Checkbox(value=True, description="case-insensitive", indent=False, layout=widgets.Layout(width="145px"))
filter_add   = widgets.Button(description="+ Add filter:", button_style="warning",
                              layout=widgets.Layout(width="auto", flex="0 0 auto"))
filter_clear = widgets.Button(description="Clear all filters", icon="times",
                              layout=widgets.Layout(width="auto", flex="0 0 auto"))
filter_list  = widgets.VBox([])
filter_state = widgets.HTML("")

clean_dropna = widgets.Checkbox(value=True,  description="drop NaN / non-numeric", indent=False, layout=widgets.Layout(width="195px"))
clean_nozero = widgets.Checkbox(value=False, description="drop zeros", indent=False, layout=widgets.Layout(width="120px"))
clean_dedup  = widgets.Checkbox(value=False, description="drop duplicate rows", indent=False, layout=widgets.Layout(width="175px"))
clean_trim   = widgets.Checkbox(value=False, description="trim percentiles", indent=False, layout=widgets.Layout(width="145px"))
trim_lo      = widgets.BoundedFloatText(value=1.0, min=0.0, max=49.0, step=0.5, description="low %:",
                                        style={"description_width": "initial"}, layout=widgets.Layout(width="130px"))
trim_hi      = widgets.BoundedFloatText(value=99.0, min=51.0, max=100.0, step=0.5, description="high %:",
                                        style={"description_width": "initial"}, layout=widgets.Layout(width="135px"))

group_col = widgets.Dropdown(options=["(none)"], value="(none)", description="Group by:",
                             style={"description_width": "initial"}, layout=widgets.Layout(width="250px"))
group_val = widgets.Dropdown(options=["(all groups)"], value="(all groups)", description="Group:",
                             style={"description_width": "initial"}, layout=widgets.Layout(width="230px"))
group_val.layout.display = "none"

_filters = []     # [{"col","op","v1","v2","ci","on"}] - all enabled ones must match

def _cond_text(c):
    if c["op"] == "between":
        return f"{c['col']}  between  {c['v1']} .. {c['v2']}"
    return f"{c['col']}  {c['op']}  {c['v1']}"

def _mask_for(df, c):
    series, op, v1, v2, ci = df[c["col"]], c["op"], c["v1"], c["v2"], c["ci"]
    if op in ["==","!=",">",">=","<","<="]:
        s_num  = pd.to_numeric(series, errors="coerce")
        v1_num = pd.to_numeric(pd.Series([v1]), errors="coerce").iloc[0]
        if s_num.notna().any() and pd.notna(v1_num):
            series, val = s_num, v1_num
        else:
            series, val = series.astype(str), str(v1)
        return {"==": series == val, "!=": series != val, ">": series > val,
                ">=": series >= val, "<": series < val, "<=": series <= val}[op]
    if op == "between":
        lo = pd.to_numeric(pd.Series([v1]), errors="coerce").iloc[0]
        hi = pd.to_numeric(pd.Series([v2]), errors="coerce").iloc[0]
        return pd.to_numeric(series, errors="coerce").between(lo, hi, inclusive="both")
    if op == "in":
        items = [i.strip() for i in str(v1).split(",") if i.strip() != ""]
        s_num = pd.to_numeric(series, errors="coerce")
        if s_num.notna().sum() > len(s_num) / 2:
            return s_num.isin(pd.to_numeric(pd.Series(items), errors="coerce").dropna().tolist())
        return series.astype(str).isin(items)
    if op == "contains":
        return series.astype(str).str.contains(str(v1), case=not ci, na=False)
    if op == "regex":
        rx = re.compile(v1, re.IGNORECASE if ci else 0)
        return series.astype(str).str.match(rx, na=False)
    return pd.Series([True] * len(df), index=df.index)

def _current_df():
    if not hasattr(uploader, "df_raw"):
        raise RuntimeError("Please upload a file first.")
    df = uploader.df_raw
    if clean_dedup.value:
        df = df.drop_duplicates()
    for c in _filters:
        if c["on"] and c["col"] in df.columns:
            df = df.loc[_mask_for(df, c)]
    return df

def _clean_counts(series):
    """Integer counts ready to fit."""
    n0 = len(series)
    s  = pd.to_numeric(series, errors="coerce")
    bad = int(s.isna().sum())
    if bad and not clean_dropna.value:
        raise RuntimeError(f"{bad} of {n0} values are NaN or non-numeric - "
                           "tick 'drop NaN / non-numeric' to ignore them.")
    v  = s.dropna().to_numpy(dtype=float)
    si = np.rint(v)
    if np.any(np.abs(v - si) > 1e-6):
        raise RuntimeError("Selected column must contain integer counts.")
    si = si.astype(int)
    if si.size and si.min() < 0:
        raise RuntimeError("Counts must be nonnegative.")
    if clean_nozero.value:
        si = si[si > 0]
    if clean_trim.value and si.size:
        lo, hi = np.percentile(si, [float(trim_lo.value), float(trim_hi.value)])
        si = si[(si >= lo) & (si <= hi)]
    if si.size < 10:
        raise RuntimeError(f"Only {si.size} usable counts after filtering and cleaning.")
    return si

def _refresh_group_values(df=None):
    if group_col.value == "(none)":
        group_val.options = ["(all groups)"]; group_val.value = "(all groups)"
        group_val.layout.display = "none"; return
    group_val.layout.display = ""
    try:
        if df is None: df = _current_df()
        vals = [str(v) for v in pd.Series(df[group_col.value]).dropna().unique().tolist()][:200]
    except Exception:
        vals = []
    keep = group_val.value
    group_val.options = ["(all groups)"] + vals
    group_val.value = keep if keep in group_val.options else "(all groups)"

def _refresh_columns():
    try:
        df = _current_df()
    except Exception as e:
        filter_state.value = f"<span style='color:#b91c1c'>{e}</span>"; return
    total = len(uploader.df_raw)
    keep  = col_dd.value
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    col_dd.options = numeric if numeric else list(df.columns)
    if keep in col_dd.options:  col_dd.value = keep
    elif col_dd.options:        col_dd.value = col_dd.options[0]
    uploader.df = df
    _refresh_group_values(df)
    n = len(df)
    colour = "#b91c1c" if n == 0 else "#111"
    filter_state.value = (f"<span style='color:{colour}'>Rows: <b>{n}</b> of {total}"
                          + (" - no rows match" if n == 0 else "") + "</span>")

def _render_filters():
    rows = []
    for i, c in enumerate(_filters):
        chk = widgets.Checkbox(value=c["on"], indent=False, layout=widgets.Layout(width="26px", flex="0 0 auto"))
        lab = widgets.HTML(f"<code>{_cond_text(c)}</code>", layout=widgets.Layout(width="340px"))
        rm  = widgets.Button(description="\u2715", tooltip="remove this condition",
                             layout=widgets.Layout(width="34px", flex="0 0 auto"))
        def _toggle(change, i=i): _filters[i]["on"] = bool(change["new"]); _after_filter_change()
        def _remove(_b, i=i):     _filters.pop(i); _after_filter_change()
        chk.observe(_toggle, names="value"); rm.on_click(_remove)
        rows.append(widgets.HBox([chk, lab, rm]))
    filter_list.children = tuple(rows)

def _after_filter_change():
    _render_filters(); _refresh_columns()

def _add_filter(_b=None):
    if not hasattr(uploader, "df_raw"):
        filter_state.value = "<span style='color:#b91c1c'>Load a file first.</span>"; return
    if not filter_col.value: return
    c = {"col": filter_col.value, "op": filter_op.value, "v1": filter_val1.value,
         "v2": filter_val2.value, "ci": bool(filter_ci.value), "on": True}
    try:
        _mask_for(uploader.df_raw, c)
    except Exception as e:
        filter_state.value = f"<span style='color:#b91c1c'>Filter error: {e}</span>"; return
    _filters.append(c)
    _after_filter_change()

def _clear_filters(_b=None):
    _filters.clear(); _after_filter_change()

filter_add.on_click(_add_filter)
filter_clear.on_click(_clear_filters)
group_col.observe(lambda ch: _refresh_group_values(), names="value")
clean_dedup.observe(lambda ch: _after_filter_change(), names="value")

status_html = widgets.HTML("")
out      = widgets.Output()
find_out = widgets.Output()
save_btn = widgets.Button(description="Save results", icon="download", disabled=True)

# Buttons
fit_all_btn = widgets.Button(description="Fit All", icon="play", button_style="primary", layout=widgets.Layout(width="120px"))
# flex "0 0 auto": never shrink below its label; buttons clip overflowing text
find_btn    = widgets.Button(description="Fit", icon="play", button_style="primary", layout=widgets.Layout(width="auto", flex="0 0 auto"))

# Right-aligned labels
def lbl(text): return widgets.HTML(f"<div style='text-align:right; white-space:nowrap; padding-right:6px;'>{text}</div>")

# Dynamic param row (single source row like your continuous UI)
_process_grid = None
def _update_process_row(*_):
    if _process_grid is None: return
    g = _process_grid
    dash = widgets.HTML("<div style='text-align:center;'>—</div>")
    src = source_dd.value
    g[1,0] = widgets.Label(src)
    def set_cells(lab1, w1, lab2=None, w2=None):
        g[1,1] = lbl(lab1); g[1,2] = w1 if w1 is not None else dash
        g[1,3] = lbl(lab2) if lab2 else dash; g[1,4] = w2 if w2 is not None else dash
    if src == "Poisson": set_cells("λ", poi_lam)
    elif src == "Binomial": set_cells("n", bin_n, "p", bin_p)
    elif src == "Geometric": set_cells("p", geo_p)
    elif src == "Negative Binomial": set_cells("r", nb_r, "p", nb_p)
    elif src == "Zero-Inflated Poisson": set_cells("π", zip_pi, "λ", zip_lam)
    else: set_cells("—", None, "—", None)

def make_process_table():
    global _process_grid
    grid = GridspecLayout(2, 5, grid_gap="2px")
    hdr_r = "font-weight:bold; text-align:right; white-space:nowrap; padding-right:6px;"
    hdr_l = "font-weight:bold; text-align:left; white-space:nowrap;"
    grid[0,0] = widgets.HTML(f"<div style='{hdr_l}'>Source</div>")
    grid[0,1] = widgets.HTML(f"<div style='{hdr_r}'>param 1</div>")
    grid[0,2] = widgets.HTML(f"<div style='{hdr_l}'>value</div>")
    grid[0,3] = widgets.HTML(f"<div style='{hdr_r}'>param 2</div>")
    grid[0,4] = widgets.HTML(f"<div style='{hdr_l}'>value</div>")
    _process_grid = grid
    _update_process_row()
    return grid

# Find-error strip
find_proc_dd = widgets.Dropdown(options=list(ALL_DISTS), value="Poisson", layout=widgets.Layout(width="220px"))
p1_lbl, p2_lbl = lbl("λ"), lbl("—")
p1_in,  p2_in  = widgets.FloatText(value=3.0, layout=widgets.Layout(width="120px")), widgets.FloatText(value=0.0, layout=widgets.Layout(width="120px"), disabled=True)
find_status = widgets.HTML("")

def _update_find_labels(*_):
    d = find_proc_dd.value
    p2_in.disabled = False
    if d == "Poisson":
        p1_lbl.value = lbl("λ").value; p2_lbl.value = lbl("—").value; p2_in.disabled = True
    elif d == "Binomial":
        p1_lbl.value = lbl("n").value; p2_lbl.value = lbl("p").value
    elif d == "Geometric":
        p1_lbl.value = lbl("p").value; p2_lbl.value = lbl("—").value; p2_in.disabled = True
    elif d == "Negative Binomial":
        p1_lbl.value = lbl("r").value; p2_lbl.value = lbl("p").value
    elif d == "Zero-Inflated Poisson":
        p1_lbl.value = lbl("π").value; p2_lbl.value = lbl("λ").value
_update_find_labels()
find_proc_dd.observe(_update_find_labels, names="value")

# ------------- Data prep -------------
def _synthetic_params_for(src):
    if src == "Poisson": return (poi_lam.value,)
    if src == "Binomial": return (bin_n.value, bin_p.value)
    if src == "Geometric": return (geo_p.value,)
    if src == "Negative Binomial": return (nb_r.value, nb_p.value)
    if src == "Zero-Inflated Poisson": return (zip_pi.value, zip_lam.value)
    return ()

def _validate_params(src, P):
    def in01(name, v): 
        if not (0 < v < 1): raise ValueError(f"{name} must be in (0,1).")
    def ge0(name, v):
        if v <= 0: raise ValueError(f"{name} must be > 0.")
    if src == "Poisson": ge0("λ", P[0])
    elif src == "Binomial": 
        if int(P[0]) < 1: raise ValueError("n must be >= 1.")
        in01("p", P[1])
    elif src == "Geometric": in01("p", P[0])
    elif src == "Negative Binomial":
        ge0("r", P[0]); in01("p", P[1])
    elif src == "Zero-Inflated Poisson":
        if not (0 <= P[0] < 1): raise ValueError("π must be in [0,1).")
        ge0("λ", P[1])

def _prepare_data():
    src = source_dd.value
    if _is_external():
        col = col_dd.value
        if not col:
            raise RuntimeError("Choose a column.")
        df = _current_df()
        if group_col.value != "(none)" and group_val.value != "(all groups)":
            df = df.loc[df[group_col.value].astype(str) == group_val.value]
        if df.empty:
            raise RuntimeError("No rows left after filtering.")
        return _clean_counts(df[col])
    else:
        N = int(N_int.value)
        P = _synthetic_params_for(src)
        _validate_params(src, P)
        return gen_data(src, N, P, seed=seed_txt.value.strip() or None)

# ------------- Event handlers -------------
_last = {"table":None, "cdf_fig":None, "pmf_fig":None, "errors_fig":None, "samples":None}

def _run_fit_by_group():
    """Fit every group of the group-by column and compare their best fits."""
    col, gcol = col_dd.value, group_col.value
    if not col:
        status_html.value = "<span style='color:#b91c1c'>Choose a column.</span>"; return
    try:
        df = _current_df()
    except Exception as e:
        status_html.value = f"<span style='color:#b91c1c'>{e}</span>"; return
    rows, names, errs, frames, skipped = [], [], [], [], []
    for g, sub in df.groupby(df[gcol].astype(str), sort=True):
        try:
            x = _clean_counts(sub[col])
        except Exception as e:
            skipped.append(f"{g} ({e})"); continue
        if len(x) < 30:
            skipped.append(f"{g} (only {len(x)} values)"); continue
        try:
            res = fit_all_discrete(x, subset=ALL_DISTS)
        except Exception as e:
            skipped.append(f"{g} ({e})"); continue
        best = res["best_name"]
        prow = res["table"].loc[res["table"]["Distribution"] == best].iloc[0]
        rows.append({gcol: g, "n": len(x), "Best fit": best,
                     "Fitted parameters": prow["Fitted parameters"],
                     "Max CDF error": f"{res['best_error']:.3f}"})
        names.append(g); errs.append(float(res["best_error"]))
        frames.append(pd.DataFrame({gcol: g, "counts": x}))
    if not rows:
        status_html.value = ("<span style='color:#b91c1c'>No group had 30+ usable counts. "
                             + ("Skipped: " + "; ".join(skipped[:6]) if skipped else "") + "</span>")
        return
    table = pd.DataFrame(rows).sort_values("Max CDF error", kind="stable", ignore_index=True)
    order = list(np.argsort(np.asarray(errs, float)))
    names = [names[i] for i in order]; errs = [errs[i] for i in order]
    with out:
        clear_output()
        html_style = """
        <style>
        .res-table { border-collapse: collapse; width: 100%; font-family: Arial,sans-serif; font-size: 14px; }
        .res-table th { background:#4CAF50; color:#fff; padding:10px; text-align:left; }
        .res-table td { border:1px solid #ddd; padding:8px; }
        .res-table tr:nth-child(even) { background:#f6f6f6; }
        .res-table th:last-child, .res-table td:last-child { text-align:center; }
        </style>
        """
        display(HTML(f"<h4>Best fit per group of '{gcol}'</h4>" + _data_context() + html_style +
                     table.to_html(index=False, classes="res-table", escape=False)))
        figg = plt.figure(num="d_groups", clear=True); figg.set_size_inches(max(6.0, 0.6 * len(names)), 4.6, forward=True)
        plt.bar(np.arange(len(names)), errs, width=0.6)
        pad = max(0.02, 0.08 * (max(errs) if errs else 1.0))
        plt.ylim(0, (max(errs) if errs else 1.0) + pad * 3)
        for i, (nm, ev) in enumerate(zip(names, errs)):
            plt.text(i, ev + pad * 0.2, f"{ev:.3f}", ha="center", va="bottom", fontsize=9)
            plt.text(i, ev + pad * 1.2, str(nm), ha="center", va="bottom", rotation=90, fontsize=9)
        plt.xticks([]); plt.ylabel("Max |Empirical CDF - Theory CDF|")
        plt.xlabel(f"Best fit per group of '{gcol}' (lower is better)")
        plt.title("Best fit error by group")
        plt.tight_layout(); _emit(figg)
        _last.update({"table": table, "errors_fig": figg, "cdf_fig": None, "pmf_fig": None,
                      "samples": pd.concat(frames, ignore_index=True)})
    save_btn.disabled = False
    skip_note = (f" <span style='color:#b45309'>Skipped: {'; '.join(skipped[:6])}.</span>" if skipped else "")
    status_html.value = (f"<span>Done. Fitted <b>{len(rows)}</b> groups of '{gcol}'. "
                         f"Pick one in <b>Group</b> to see its charts.</span>{skip_note}")

def _run_fit_all(_=None):
    results_tabs.selected_index = 0
    status_html.value = ""
    out.clear_output(); save_btn.disabled = True
    if _is_external() and group_col.value != "(none)" and group_val.value == "(all groups)":
        _run_fit_by_group(); return
    try:
        x = _prepare_data()
    except Exception as e:
        status_html.value = f"<span style='color:#b91c1c'>{e}</span>"; return
    try:
        res = fit_all_discrete(x, subset=ALL_DISTS)
    except Exception as e:
        status_html.value = f"<span style='color:#b91c1c'>Fit error: {e}</span>"; return

    with out:
        clear_output()
        # Table
        html_style = """
        <style>
        .res-table { border-collapse: collapse; width: 100%; font-family: Arial,sans-serif; font-size: 14px; }
        .res-table th { background:#4CAF50; color:#fff; padding:10px; text-align:left; }
        .res-table td { border:1px solid #ddd; padding:8px; }
        .res-table tr:nth-child(even) { background:#f6f6f6; }
        .res-table th:last-child, .res-table td:last-child { text-align:center; }
        </style>
        """
        display(HTML("<h4>Fitted Parameters (Discrete)</h4>" + _data_context() + html_style +
                     res["table"].to_html(index=False, classes="res-table", escape=False)))

        # Error bars with vertical names
        names = list(res["errors"].keys())
        errs  = np.array([res["errors"][n] for n in names], float)
        xidx  = np.arange(len(names))
        figw  = max(6.0, 0.6*len(names))
        fig_err = plt.figure(num="d_err", clear=True); fig_err.set_size_inches(figw, 4.6, forward=True)
        bars = plt.bar(xidx, errs, width=0.6)
        ymax = float(errs.max()) if errs.size else 1.0
        pad  = max(0.02, 0.08*ymax)
        plt.ylim(0, ymax + pad*3)
        for i,(nm,ev) in enumerate(zip(names,errs)):
            plt.text(i, ev + pad*0.2, f"{ev:.3f}", ha="center", va="bottom", fontsize=9)
            plt.text(i, ev + pad*1.2, nm, ha="center", va="bottom", rotation=90, fontsize=9)
        plt.xticks([]); plt.ylabel("Max |Empirical CDF − Theory CDF|")
        plt.xlabel(f"Best = {res['best_name']} (error = {res['best_error']:.3f})")
        plt.title("Fit errors (discrete)")
        plt.tight_layout(); _emit(fig_err)

        # CDF overlay (steps)
        k_sup = res["k_sup"]
        F_emp_sup = np.interp(k_sup, res["k_emp"], res["F_emp"], left=0.0, right=1.0)
        F_best = theory_cdf(res["best_name"], res["fitted"][res["best_name"]], k_sup)
        fig_cdf = plt.figure(num="d_cdf", clear=True); fig_cdf.set_size_inches(7.5, 4, forward=True)
        plt.step(k_sup, F_emp_sup, where="post", label="Empirical CDF")
        plt.step(k_sup, F_best, where="post", label=f"Theory CDF ({res['best_name']})")
        plt.legend(); plt.xlabel("k"); plt.ylabel("CDF"); plt.title("Empirical vs Best-fit CDF")
        plt.tight_layout(); _emit(fig_cdf)

        # PMF overlay (bars + step)
        k_emp, pmf_emp = res["k_emp"], res["pmf_emp"]
        k_plot = np.arange(0, max(k_emp.max(), k_sup.max())+1, dtype=int)
        pmf_best = theory_pmf(res["best_name"], res["fitted"][res["best_name"]], k_plot)
        fig_pmf = plt.figure(num="d_pmf", clear=True); fig_pmf.set_size_inches(7.5, 4, forward=True)
        plt.bar(k_emp, pmf_emp, width=0.9, alpha=0.6, label="Empirical PMF")
        plt.step(k_plot, pmf_best, color="orange",  where="mid", label=f"Theory PMF ({res['best_name']})")
        plt.legend(); plt.xlabel("k"); plt.ylabel("PMF"); plt.title("Empirical vs Best-fit PMF")
        plt.tight_layout(); _emit(fig_pmf)

        _last.update({"table":res["table"], "errors_fig":fig_err, "cdf_fig":fig_cdf, "pmf_fig":fig_pmf,
                      "samples": pd.DataFrame({"counts": np.rint(np.asarray(x)).astype(int)})})

    save_btn.disabled = False
    status_html.value = f"<span>Done. Best fit: <b>{res['best_name']}</b> (error = {res['best_error']:.3f}).</span>"

fit_all_btn.on_click(_run_fit_all)

def _on_find(_=None):
    results_tabs.selected_index = 1
    find_status.value = ""
    find_out.clear_output()
    try:
        x = _prepare_data()
    except Exception as e:
        find_status.value = f"<span style='color:#b91c1c'>{e}</span>"; return
    k_emp, pmf_emp, F_emp = empirical_pmf_cdf(x)
    k_sup = support_for_plot(x, extra=25)
    F_emp_sup = np.interp(k_sup, k_emp, F_emp, left=0.0, right=1.0)

    d = find_proc_dd.value
    p1 = float(p1_in.value)
    p2 = float(p2_in.value)
    # validate
    try:
        if d == "Poisson": 
            pars = (max(p1,1e-9),)
        elif d == "Binomial":
            n = int(round(p1)); 
            if n < 1: raise ValueError("n must be >=1")
            p = min(max(p2, 1e-9), 1-1e-9)
            pars = (n, p)
        elif d == "Geometric":
            p = min(max(p1, 1e-9), 1-1e-9); pars = (p,)
        elif d == "Negative Binomial":
            r = max(p1, 1e-9); p = min(max(p2, 1e-9), 1-1e-9); pars = (r, p)
        elif d == "Zero-Inflated Poisson":
            pi = min(max(p1, 0.0), 0.999999); lam = max(p2, 1e-9); pars = (pi, lam)
        else:
            raise ValueError("Unknown distribution")
    except Exception as e:
        find_status.value = f"<span style='color:#b91c1c'>Parameter error: {e}</span>"; return

    try:
        F_th = theory_cdf(d, pars, k_sup)
        err = float(np.max(np.abs(F_th - F_emp_sup)))
    except Exception as e:
        find_status.value = f"<span style='color:#b91c1c'>Computation error: {e}</span>"; return

    find_status.value = (f"<span>Max CDF error for <b>{d}</b> with parameters <tt>{tuple(pars)}</tt> = <b>{err:.3f}</b>.</span>"
                         + _data_context())

    with find_out:
        clear_output()
        # CDF
        fig1 = plt.figure(num="d_find_cdf", clear=True); fig1.set_size_inches(6.8, 3.2, forward=True)
        plt.step(k_sup, F_emp_sup, where="post", label="Empirical CDF")
        plt.step(k_sup, F_th, where="post", label=f"Theory CDF ({d})")
        plt.legend(); plt.xlabel("k"); plt.ylabel("CDF"); plt.title("Find error: CDF")
        plt.tight_layout(); _emit(fig1)
        # PMF
        k_plot = np.arange(0, max(k_sup.max(), k_emp.max())+1)
        pmf_th = theory_pmf(d, pars, k_plot)
        fig2 = plt.figure(num="d_find_pmf", clear=True); fig2.set_size_inches(6.8, 3.2, forward=True)
        plt.bar(k_emp, pmf_emp, width=0.9, alpha=0.6, label="Empirical PMF")
        plt.step(k_plot, pmf_th, color="orange", where="mid", label=f"Theory PMF ({d})")
        plt.legend(); plt.xlabel("k"); plt.ylabel("PMF"); plt.title("Find error: PMF")
        plt.tight_layout(); _emit(fig2)

find_btn.on_click(_on_find)

# ------------- Save results -------------
def _save(_=None):
    if _last["table"] is None:
        status_html.value = "<span style='color:#b91c1c'>Run a fit first.</span>"; return
    import time
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = os.path.join("outputs", f"discrete_fit_outputs_{ts}")
    os.makedirs(base, exist_ok=True)
    table_path  = f"{base}/fitted_parameters.csv"
    samples_path= f"{base}/samples.csv"
    _last["table"].to_csv(table_path, index=False)
    _last["samples"].to_csv(samples_path, index=False)
    links = [FileLink(table_path, result_html_prefix="Fitted parameters: "),
             FileLink(samples_path, result_html_prefix="Samples: ")]
    for key, fname, label in (("errors_fig", "fit_errors.png", "Fit errors plot: "),
                              ("cdf_fig", "cdf_overlay.png", "CDF overlay plot: "),
                              ("pmf_fig", "pmf_overlay.png", "PMF overlay plot: ")):
        fig = _last.get(key)
        if fig is None:      # group mode has no single overlay
            continue
        path = f"{base}/{fname}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        links.append(FileLink(path, result_html_prefix=label))
    status_html.value = "<b>Saved.</b> Open: " + " | ".join([l._repr_html_() for l in links])

save_btn.on_click(_save)

# ------------- Uploader -------------
def _get_upload(upl: widgets.FileUpload):
    v = upl.value
    if not v: return None, None
    if isinstance(v, dict):  # ipywidgets v7
        first = next(iter(v.items()))[1]
        return first.get("content"), first.get("name","upload")
    else:  # v8
        uf = v[0]
        return getattr(uf,"content",None), getattr(uf,"name","upload")

def _on_upload(_):
    content, name = _get_upload(uploader)
    if not content:
        return
    try:
        df = _read_uploaded_dataframe(content, name)
    except Exception as e:
        status_html.value = f"<span style='color:#b91c1c'>Load error: {e}</span>"; return
    uploader.df_raw = df.copy()
    uploader.df = df
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    col_dd.options = numeric_cols if numeric_cols else list(df.columns)
    if col_dd.options: col_dd.value = col_dd.options[0]
    _filters.clear()
    filter_col.options = list(df.columns)
    if filter_col.options: filter_col.value = filter_col.options[0]
    group_col.options = ["(none)"] + list(df.columns)
    group_col.value = "(none)"
    _render_filters(); _refresh_group_values(df)
    rows, cols = df.shape
    status_html.value = f"<span>Loaded file <b>{name}</b> ({rows} rows × {cols} cols).</span>"

uploader.observe(_on_upload, names="value")

def _data_context(as_html=True):
    """One line naming exactly which data a result came from: column, group, filters."""
    esc = lambda t: str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    if _is_external():
        parts = [f"column '{col_dd.value}'"]
        if group_col.value != "(none)":
            parts.append(f"all groups of '{group_col.value}'" if group_val.value == "(all groups)"
                         else f"group '{group_val.value}' of '{group_col.value}'")
        active = sum(1 for c in _filters if c["on"])
        if active:
            parts.append(f"{active} filter{'s' if active > 1 else ''}")
        try:                       # count the rows actually fitted, group narrowing included
            df = _current_df()
            if group_col.value != "(none)" and group_val.value != "(all groups)":
                df = df.loc[df[group_col.value].astype(str) == group_val.value]
            parts.append(f"{len(df)} rows")
        except Exception:
            pass
        text = "File data: " + ", ".join(parts)
    else:
        seed = seed_txt.value.strip()
        text = f"Simulated {source_dd.value}, N = {int(N_int.value)}" + (f", seed {seed}" if seed else "")
    return f"<div style='color:#555; margin: 2px 0 6px 0;'>{esc(text)}</div>" if as_html else text

# ------------- Data visualisation -------------
viz_out    = widgets.Output()
viz_status = widgets.HTML("")
def _viz_btn(label):
    return widgets.Button(description=label, layout=widgets.Layout(width="auto", flex="0 0 auto"))
viz_bar   = _viz_btn("Counts bar chart")
viz_ecdf  = _viz_btn("ECDF")
viz_box   = _viz_btn("Box plot")
viz_run   = _viz_btn("Run chart")
viz_srun  = _viz_btn("Sorted run chart")
viz_stats = _viz_btn("Summary stats")
viz_split = widgets.Checkbox(value=False, description="split by group", indent=False,
                             layout=widgets.Layout(width="140px"))
viz_grid  = widgets.Checkbox(value=False, description="grid lines", indent=False,
                             layout=widgets.Layout(width="110px"),
                             tooltip="Grid lines on the run charts")
# Figures: the run chart is interactive (pan/zoom toolbar), everything else is a
# static image. Both need the ipympl backend active - the static ones are rendered to
# PNG by hand, because under that backend pyplot would hand back a live canvas.
try:
    import ipympl                                    # noqa: F401
    _HAVE_IPYMPL = True
except Exception:
    _HAVE_IPYMPL = False

def setup_figures():
    """Activate the backend the run chart needs. Safe to call more than once."""
    if not _HAVE_IPYMPL:
        return False
    ip = get_ipython()
    if ip is not None:
        ip.run_line_magic("matplotlib", "widget")
    else:                                            # plain interpreter, e.g. a test run
        plt.switch_backend("module://ipympl.backend_nbagg")
    return True

def _interactive_backend():
    """True when figures are ipympl canvases.

    %matplotlib widget reports the backend as "widget", while switch_backend reports
    "module://ipympl.backend_nbagg" - accept either spelling.
    """
    return any(k in plt.get_backend().lower() for k in ("ipympl", "widget", "nbagg"))

def _emit(fig, live=False):
    """Show a figure: a live canvas when `live`, otherwise a static PNG."""
    if live and _interactive_backend():
        display(fig.canvas)              # the canvas is the widget - leave the figure open
    else:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=fig.get_dpi(), bbox_inches="tight")
        display(Image(data=buf.getvalue()))
        plt.close(fig)

def _data_by_group():
    """[(label, counts)] - one entry per group when splitting, otherwise one entry."""
    if viz_split.value and _is_external() and group_col.value != "(none)":
        df = _current_df()
        groups = []
        for g, sub in df.groupby(df[group_col.value].astype(str), sort=True):
            try:
                v = _clean_counts(sub[col_dd.value])
            except Exception:
                continue
            if len(v) >= 5:
                groups.append((str(g), v))
        if groups:
            return groups[:10]
    return [("data", _prepare_data())]

def _viz_start(title):
    results_tabs.selected_index = 2
    viz_out.clear_output()
    viz_status.value = f"<span>{title}</span>" + _data_context()

def _viz_guard(fn):
    def wrapped(_b=None):
        try:
            fn()
        except Exception as e:
            viz_out.clear_output()
            viz_status.value = f"<span style='color:#b91c1c'>{e}</span>"
    return wrapped

def _viz_bar():
    sets = _data_by_group()
    _viz_start("Empirical PMF - share of rows at each count")
    with viz_out:
        clear_output()
        fig = plt.figure(num="d_viz_bar", clear=True); fig.set_size_inches(8, 4.2, forward=True)
        kmax = max(int(v.max()) for _l, v in sets)
        width = 0.8 / len(sets)
        for i, (label, v) in enumerate(sets):
            ks = np.arange(0, kmax + 1)
            pmf = np.array([(v == k).mean() for k in ks], float)
            plt.bar(ks + (i - (len(sets) - 1) / 2) * width, pmf, width=width,
                    alpha=0.85, label=f"{label} (n={len(v)})")
        plt.xlabel("k"); plt.ylabel("share of rows"); plt.title("Empirical PMF")
        plt.legend(fontsize=8); plt.tight_layout(); _emit(fig)

def _viz_ecdf():
    sets = _data_by_group()
    _viz_start("Empirical CDF")
    with viz_out:
        clear_output()
        fig = plt.figure(num="d_viz_ecdf", clear=True); fig.set_size_inches(8, 4.2, forward=True)
        for label, v in sets:
            xs = np.sort(v); ys = np.arange(1, len(xs) + 1) / len(xs)
            plt.step(xs, ys, where="post", label=f"{label} (n={len(xs)})")
        plt.xlabel("k"); plt.ylabel("F(k)"); plt.title("Empirical CDF")
        plt.legend(fontsize=8); plt.tight_layout(); _emit(fig)

def _viz_box():
    sets = _data_by_group()
    _viz_start("Box plot")
    with viz_out:
        clear_output()
        fig = plt.figure(num="d_viz_box", clear=True); fig.set_size_inches(max(6.0, 1.4 * len(sets) + 3), 4.2, forward=True)
        plt.boxplot([v for _l, v in sets], tick_labels=[l for l, _v in sets], showmeans=True)
        plt.ylabel("count"); plt.title("Box plot"); plt.grid(axis="y", alpha=0.3)
        plt.tight_layout(); _emit(fig)

_run_fig = None                                      # the run chart on screen, for the grid toggle

def _apply_grid(fig):
    for ax in fig.axes:
        if viz_grid.value:
            ax.grid(True, alpha=0.35, lw=0.6)
        else:
            ax.grid(False)                           # no line kwargs, or matplotlib turns it back on
    fig.canvas.draw_idle()

def _on_grid_toggle(_change):
    """Update the live run chart in place, without redrawing it or losing the zoom."""
    if _run_fig is not None and plt.fignum_exists(_run_fig.number):
        _apply_grid(_run_fig)

def _viz_run(sort=False):
    global _run_fig
    x = _prepare_data()
    if sort:
        x = np.sort(x)
        _viz_start("Sorted run chart - counts in ascending order, to show range, gaps and outliers")
    else:
        _viz_start("Run chart - counts in row order, to show drift or steps")
    title = "Sorted run chart" if sort else "Run chart"
    with viz_out:
        clear_output()
        fig = plt.figure(num="d_viz_run_sorted" if sort else "d_viz_run", clear=True)
        fig.set_size_inches(7, 3.6, forward=True)
        plt.plot(np.arange(len(x)), x, lw=0.7, drawstyle="steps-post" if sort else "default")
        plt.axhline(float(np.mean(x)), color="orange", lw=1, label=f"mean = {np.mean(x):.3f}")
        if sort:
            plt.axhline(float(np.median(x)), color="green", lw=1, ls="--", label=f"median = {np.median(x):g}")
        plt.xlabel("rank (sorted)" if sort else "row order"); plt.ylabel("count"); plt.title(title)
        plt.legend(fontsize=8); _apply_grid(fig); plt.tight_layout()
        _run_fig = fig; _emit(fig, live=True)

def _viz_stats():
    sets = _data_by_group()
    _viz_start("Summary statistics")
    rows = []
    for label, v in sets:
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        vf = np.asarray(v, float)
        rows.append({"group": label, "n": len(v), "mean": f"{vf.mean():.4f}", "var": f"{vf.var(ddof=1):.4f}",
                     "var/mean": f"{(vf.var(ddof=1) / vf.mean()):.3f}" if vf.mean() else "-",
                     "min": int(vf.min()), "25%": f"{q1:g}", "median": f"{med:g}", "75%": f"{q3:g}",
                     "max": int(vf.max()), "zeros": int((vf == 0).sum()),
                     "skew": f"{float(st.skew(vf)):.3f}", "kurtosis": f"{float(st.kurtosis(vf)):.3f}"})
    with viz_out:
        clear_output()
        style = """<style>.viz-stats{border-collapse:collapse;font-family:Arial,sans-serif;font-size:14px;}
        .viz-stats th{background:#4CAF50;color:#fff;padding:8px;text-align:left;}
        .viz-stats td{border:1px solid #ddd;padding:6px;}
        .viz-stats tr:nth-child(even){background:#f6f6f6;}</style>"""
        display(HTML(style + pd.DataFrame(rows).to_html(index=False, classes="viz-stats", escape=False)))

viz_bar.on_click(_viz_guard(_viz_bar));   viz_ecdf.on_click(_viz_guard(_viz_ecdf))
viz_box.on_click(_viz_guard(_viz_box));   viz_run.on_click(_viz_guard(_viz_run))
viz_stats.on_click(_viz_guard(_viz_stats)); viz_srun.on_click(_viz_guard(lambda: _viz_run(sort=True)))
viz_grid.observe(_on_grid_toggle, names="value")

def _sep(label):
    """Captioned rule marking the start of a block of the UI."""
    return widgets.HTML(
        "<div style='border-top:1px solid #d9d9d9; margin:10px 0 4px 0; padding-top:4px;"
        " font-size:11px; letter-spacing:.06em; text-transform:uppercase; color:#8a8a8a;'>"
        f"{label}</div>")

# ------------- Layout -------------
top = widgets.HBox([source_dd, N_int, seed_txt])
process_table = make_process_table()
find_row = widgets.HBox([find_btn, find_proc_dd,
                         p1_lbl, p1_in, p2_lbl, p2_in])

controls = widgets.HBox([fit_all_btn, save_btn])

ext_note = widgets.HTML("<em style='padding-left:12px'>The selected column must contain "
                        "integer counts (0, 1, 2, ...).</em>")
upload_area = widgets.VBox([uploader,
                            widgets.HBox([widgets.HTML("<b style='white-space:nowrap'>Use column:</b>",
                                                       layout=widgets.Layout(width="86px")),
                                          col_dd, ext_note])])
filter_row1  = widgets.HBox([filter_add, filter_col, filter_op, filter_val1, filter_val2, filter_ci])
filter_row2  = widgets.HBox([filter_clear, filter_state])
clean_row    = widgets.HBox([widgets.HTML("<b style='white-space:nowrap'>Clean:</b>", layout=widgets.Layout(width="58px")),
                             clean_dropna, clean_nozero, clean_dedup, clean_trim, trim_lo, trim_hi])
group_row    = widgets.HBox([group_col, group_val])
filter_panel = widgets.VBox([filter_row1, filter_list, filter_row2, clean_row, group_row])
filter_panel.layout.display = "none"

def _on_mode_change(change=None):
    """Show the file controls, hide what only applies to simulated data."""
    ext = _is_external()
    upload_area.layout.display = "" if ext else "none"
    ext_note.layout.display = "" if ext else "none"
    filter_panel.layout.display = "" if ext else "none"
    source_dd.layout.display = "none" if ext else ""
    N_int.layout.display = "none" if ext else ""
    seed_txt.layout.display = "none" if ext else ""   # seed only affects simulated sampling
    process_table.layout.display = "none" if ext else ""

mode_dd.observe(_on_mode_change, names="value")
_on_mode_change()

# Results in tabs so Fit output isn't buried under the Fit All table; each button selects its tab
# ipywidgets caps tab headers near 140px, which truncates these labels
_tab_css = widgets.HTML("<style>"
    # scoped to widget tab bars only, so JupyterLab / VS Code document tabs are untouched
    ".jupyter-widget-TabPanel-tabBar .lm-TabBar-tab, .widget-tab-bar .lm-TabBar-tab,"
    ".jupyter-widget-TabPanel-tabBar .p-TabBar-tab, .widget-tab-bar .p-TabBar-tab"
    " { min-width: 215px !important; max-width: 360px !important; }"
    ".jupyter-widget-TabPanel-tabBar .lm-TabBar-tabLabel, .widget-tab-bar .lm-TabBar-tabLabel,"
    ".jupyter-widget-TabPanel-tabBar .p-TabBar-tabLabel, .widget-tab-bar .p-TabBar-tabLabel"
    " { overflow: visible !important; text-overflow: clip !important; }"
    "</style>")
viz_row = widgets.HBox([widgets.HTML("<b style='white-space:nowrap'>Visualize:</b>",
                                     layout=widgets.Layout(width="72px")),
                        viz_bar, viz_ecdf, viz_box, viz_run, viz_srun, viz_stats, viz_split, viz_grid])

results_tabs = widgets.Tab(children=[out, widgets.VBox([find_status, find_out]),
                                     widgets.VBox([viz_status, viz_out])])
results_tabs.set_title(0, "All-distribution results")   # filled by Fit All
results_tabs.set_title(1, "Single-distribution results")  # filled by Fit
results_tabs.set_title(2, "Data view")                    # filled by the Visualize buttons

ui = widgets.VBox([
    _tab_css,                 # widen the result tab headers
    _sep("Data source"),
    mode_dd,                  # Simulated | From file
    top,
    process_table,
    upload_area,              # file mode only
    filter_panel,             # file mode only
    _sep("Fit one distribution"),
    find_row,
    _sep("Fit all distributions"),
    controls,
    _sep("Visualize data"),
    viz_row,                  # plots of the data itself
    _sep("Results"),
    status_html,
    results_tabs,
])

source_dd.observe(_update_process_row, names="value")
_update_process_row()
def main():
    """Display the app. The run chart is interactive; other figures are images."""
    setup_figures()
    display(ui)

if "__name__" == "__main__":
    main()