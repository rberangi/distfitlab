# === Full app: multi-format upload, load-from-path, filtering, Fit-all, Find-error (CDF+PDF) ===
import io, os, tempfile, math, re
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import scipy.stats as st
from ipywidgets import GridspecLayout
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output, FileLink, Image
from IPython import get_ipython

from .file_readers import read_uploaded_dataframe as _read_uploaded_dataframe

# ---------------- Core helpers ----------------
def pdf_cdf(x, bins, PDF=True):
    """Empirical PDF or CDF via histogram-based CDF (stable on tails)."""
    x = np.asarray(x, dtype=float)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if xmin == xmax:
        xmax = xmin + 1e-9
    rangex = np.linspace(xmin, xmax, num=bins + 1)
    H = np.histogram(x, rangex)
    denom = np.sum(H[0][0:bins])
    if denom == 0:
        xs = np.sort(x)
        y  = np.arange(1, len(xs)+1) / len(xs)
        return [xs, y] if not PDF else [xs, np.gradient(y, xs)]
    cdf = np.cumsum(H[0][0:bins]) / denom
    if PDF:
        pdf = (cdf[1:bins] - cdf[0:bins-1]) / ((xmax - xmin) / bins)
        range_pdf = (rangex[1:bins] + rangex[2:bins+1]) / 2
        return [range_pdf, pdf]
    else:
        range_cdf = rangex[1:bins+1]
        return [range_cdf, cdf]

# ---------- synthetic / external data ----------
def datasource(Source, N, P, upload_df=None, upload_col=None, seed=None):
    """Generate or load data. UI convention: loc is always first in P for synthetic sources."""
    if seed is not None and str(seed).strip() != "":
        np.random.seed(int(seed))

    S = Source
    if S == 'Normal':            # [loc(μ), σ]
        loc, sig = P;             return np.random.normal(loc, sig, N)
    elif S == 'Exponential':     # [loc, scale]
        loc, sc = P;              return loc + np.random.exponential(sc, N)
    elif S == 'Rayleigh':        # [loc, σ]
        loc, sig = P;             return loc + np.random.rayleigh(sig, N)
    elif S == 'Gamma':           # [loc, k, θ]
        loc, k, th = P;           return loc + np.random.gamma(k, th, N)
    elif S == 'Weibull':         # [loc, k, λ]
        loc, k, lam = P;          return st.weibull_min.rvs(c=k, loc=loc, scale=lam, size=N)

    # Extended synthetic sources
    elif S == 'Lognormal':       # [loc, s(shape), scale]
        loc, s, sc = P;           return loc + np.random.lognormal(mean=0.0, sigma=s, size=N) * sc
    elif S == 'Loglogistic':     # Fisk: [loc, c(shape), scale]
        loc, c, sc = P;           return st.fisk.rvs(c=c, loc=loc, scale=sc, size=N)
    elif S == 'Inverse Gaussian':# [loc, μ(shape), scale]
        loc, mu, sc = P;          return st.invgauss.rvs(mu=mu, loc=loc, scale=sc, size=N)
    elif S == 'Beta':            # [loc, α, β, scale]
        loc, a, b, sc = P;        return st.beta.rvs(a=a, b=b, loc=loc, scale=sc, size=N)
    elif S == 'GEV':             # [loc, ξ(shape), scale]
        loc, xi, sc = P;          return st.genextreme.rvs(c=xi, loc=loc, scale=sc, size=N)
    elif S == 'Logistic':        # [loc, s]
        loc, s = P;               return st.logistic.rvs(loc=loc, scale=s, size=N)
    elif S == 'Laplace':         # [loc, b(scale)]
        loc, b = P;               return st.laplace.rvs(loc=loc, scale=b, size=N)

    # Previously added extras
    elif S == 'Chi-squared':     # [loc, ν, scale]
        loc, nu, sc = P;          return st.chi2.rvs(df=nu, loc=loc, scale=sc, size=N)
    elif S == 'Chi':             # [loc, ν, scale]
        loc, nu, sc = P;          return st.chi.rvs(df=nu, loc=loc, scale=sc, size=N)
    elif S == 'Nakagami':        # [loc, m, scale]
        loc, m, sc = P;           return st.nakagami.rvs(nu=m, loc=loc, scale=sc, size=N)
    elif S == 'Rician':          # [loc, ν(noncentral), σ]
        loc, nu, sig = P;         return st.rice.rvs(b=nu, loc=loc, scale=sig, size=N)
    elif S == 'Cauchy':          # [loc, s]
        loc, s = P;               return st.cauchy.rvs(loc=loc, scale=s, size=N)
    elif S == 'Student-T':       # [loc, ν, s]
        loc, nu, s = P;           return st.t.rvs(df=nu, loc=loc, scale=s, size=N)

    elif S == 'External':
        if upload_df is None or upload_col is None:
            raise ValueError("Please upload a data file and choose a column.")
        s = pd.to_numeric(upload_df[upload_col], errors="coerce").dropna()
        if s.empty:
            raise ValueError("Selected column has no numeric data.")
        return s.to_numpy()
    else:
        raise ValueError("Unknown source.")

# ---------- theory CDF/PDF (SciPy stable) ----------
def theory_cdf(process, parameters, x):
    x = np.asarray(x, dtype=float); P = parameters
    if process == "Normal":
        mu, sig = P;                     return st.norm.cdf(x, loc=mu, scale=sig)
    elif process == "Exponential":
        loc, rate = P; sc = 1.0 / max(rate, np.finfo(float).tiny); return st.expon.cdf(x, loc=loc, scale=sc)
    elif process == "Gamma":
        a, loc, th = P;                  return st.gamma.cdf(x, a=a, loc=loc, scale=th)
    elif process == "Rayleigh":
        loc, sig = P;                    return st.rayleigh.cdf(x, loc=loc, scale=sig)
    elif process == "Weibull":
        k, loc, lam = P;                 return st.weibull_min.cdf(x, c=k, loc=loc, scale=lam)

    # Extended:
    elif process == "Lognormal":
        s, loc, sc = P;                  return st.lognorm.cdf(x, s=s, loc=loc, scale=sc)
    elif process == "Loglogistic":
        c, loc, sc = P;                  return st.fisk.cdf(x, c=c, loc=loc, scale=sc)
    elif process == "Inverse Gaussian":
        mu, loc, sc = P;                 return st.invgauss.cdf(x, mu=mu, loc=loc, scale=sc)
    elif process == "Beta":
        a, b, loc, sc = P;               return st.beta.cdf(x, a=a, b=b, loc=loc, scale=sc)
    elif process == "GEV":
        xi, loc, sc = P;                 return st.genextreme.cdf(x, c=xi, loc=loc, scale=sc)
    elif process == "Logistic":
        loc_, s = P;                     return st.logistic.cdf(x, loc=loc_, scale=s)
    elif process == "Laplace":
        loc_, b = P;                     return st.laplace.cdf(x, loc=loc_, scale=b)

    # Previously added:
    elif process == "Chi-squared":
        nu, loc, sc = P;                 return st.chi2.cdf(x, df=nu, loc=loc, scale=sc)
    elif process == "Chi":
        nu, loc, sc = P;                 return st.chi.cdf(x, df=nu, loc=loc, scale=sc)
    elif process == "Nakagami":
        m, loc, sc = P;                  return st.nakagami.cdf(x, nu=m, loc=loc, scale=sc)
    elif process == "Rician":
        nu, loc, sig = P;                return st.rice.cdf(x, b=nu, loc=loc, scale=sig)
    elif process == "Cauchy":
        loc_, s = P;                     return st.cauchy.cdf(x, loc=loc_, scale=s)
    elif process == "Student-T":
        nu, loc_, s = P;                 return st.t.cdf(x, df=nu, loc=loc_, scale=s)
    else:
        raise ValueError("Unknown process.")

def theory_ppf(process, parameters, q):
    """Quantiles of `process` - the inverse of theory_cdf, used by the Q-Q plot."""
    q = np.asarray(q, dtype=float); P = parameters
    if process == "Normal":
        mu, sig = P;                     return st.norm.ppf(q, loc=mu, scale=sig)
    elif process == "Exponential":
        loc, rate = P; sc = 1.0 / max(rate, np.finfo(float).tiny); return st.expon.ppf(q, loc=loc, scale=sc)
    elif process == "Gamma":
        a, loc, th = P;                  return st.gamma.ppf(q, a=a, loc=loc, scale=th)
    elif process == "Rayleigh":
        loc, sig = P;                    return st.rayleigh.ppf(q, loc=loc, scale=sig)
    elif process == "Weibull":
        k, loc, lam = P;                 return st.weibull_min.ppf(q, c=k, loc=loc, scale=lam)
    elif process == "Lognormal":
        sh, loc, sc = P;                 return st.lognorm.ppf(q, s=sh, loc=loc, scale=sc)
    elif process == "Loglogistic":
        c, loc, sc = P;                  return st.fisk.ppf(q, c=c, loc=loc, scale=sc)
    elif process == "Inverse Gaussian":
        mu, loc, sc = P;                 return st.invgauss.ppf(q, mu=mu, loc=loc, scale=sc)
    elif process == "Beta":
        a, b, loc, sc = P;               return st.beta.ppf(q, a=a, b=b, loc=loc, scale=sc)
    elif process == "GEV":
        xi, loc, sc = P;                 return st.genextreme.ppf(q, c=xi, loc=loc, scale=sc)
    elif process == "Logistic":
        loc_, sh = P;                    return st.logistic.ppf(q, loc=loc_, scale=sh)
    elif process == "Laplace":
        loc_, b = P;                     return st.laplace.ppf(q, loc=loc_, scale=b)
    elif process == "Chi-squared":
        nu, loc, sc = P;                 return st.chi2.ppf(q, df=nu, loc=loc, scale=sc)
    elif process == "Chi":
        nu, loc, sc = P;                 return st.chi.ppf(q, df=nu, loc=loc, scale=sc)
    elif process == "Nakagami":
        m, loc, sc = P;                  return st.nakagami.ppf(q, nu=m, loc=loc, scale=sc)
    elif process == "Rician":
        nu, loc, sig = P;                return st.rice.ppf(q, b=nu, loc=loc, scale=sig)
    elif process == "Cauchy":
        loc_, sh = P;                    return st.cauchy.ppf(q, loc=loc_, scale=sh)
    elif process == "Student-T":
        nu, loc_, sh = P;                return st.t.ppf(q, df=nu, loc=loc_, scale=sh)
    else:
        raise ValueError("Unknown process.")

def theory_pdf(process, parameters, x):
    x = np.asarray(x, dtype=float); P = parameters
    if process == "Normal":
        mu, sig = P;                     return st.norm.pdf(x, loc=mu, scale=sig)
    elif process == "Exponential":
        loc, rate = P; sc = 1.0 / max(rate, np.finfo(float).tiny); return st.expon.pdf(x, loc=loc, scale=sc)
    elif process == "Gamma":
        a, loc, th = P;                  return st.gamma.pdf(x, a=a, loc=loc, scale=th)
    elif process == "Rayleigh":
        loc, sig = P;                    return st.rayleigh.pdf(x, loc=loc, scale=sig)
    elif process == "Weibull":
        k, loc, lam = P;                 return st.weibull_min.pdf(x, c=k, loc=loc, scale=lam)

    # Extended:
    elif process == "Lognormal":
        s, loc, sc = P;                  return st.lognorm.pdf(x, s=s, loc=loc, scale=sc)
    elif process == "Loglogistic":
        c, loc, sc = P;                  return st.fisk.pdf(x, c=c, loc=loc, scale=sc)
    elif process == "Inverse Gaussian":
        mu, loc, sc = P;                 return st.invgauss.pdf(x, mu=mu, loc=loc, scale=sc)
    elif process == "Beta":
        a, b, loc, sc = P;               return st.beta.pdf(x, a=a, b=b, loc=loc, scale=sc)
    elif process == "GEV":
        xi, loc, sc = P;                 return st.genextreme.pdf(x, c=xi, loc=loc, scale=sc)
    elif process == "Logistic":
        loc_, s = P;                     return st.logistic.pdf(x, loc=loc_, scale=s)
    elif process == "Laplace":
        loc_, b = P;                     return st.laplace.pdf(x, loc=loc_, scale=b)

    # Previously added:
    elif process == "Chi-squared":
        nu, loc, sc = P;                 return st.chi2.pdf(x, df=nu, loc=loc, scale=sc)
    elif process == "Chi":
        nu, loc, sc = P;                 return st.chi.pdf(x, df=nu, loc=loc, scale=sc)
    elif process == "Nakagami":
        m, loc, sc = P;                  return st.nakagami.pdf(x, nu=m, loc=loc, scale=sc)
    elif process == "Rician":
        nu, loc, sig = P;                return st.rice.pdf(x, b=nu, loc=loc, scale=sig)
    elif process == "Cauchy":
        loc_, s = P;                     return st.cauchy.pdf(x, loc=loc_, scale=s)
    elif process == "Student-T":
        nu, loc_, s = P;                 return st.t.pdf(x, df=nu, loc=loc_, scale=s)
    else:
        raise ValueError("Unknown process.")

# ---------- fit / summarize ----------
ALL_DISTS = ("Normal","Exponential","Gamma","Rayleigh","Weibull",
             "Lognormal","Loglogistic","Inverse Gaussian","Beta","GEV","Logistic","Laplace",
             "Chi-squared","Chi","Nakagami","Rician","Cauchy","Student-T")

def fit_subset_and_summarize(r, bins=100, subset=ALL_DISTS, fix_loc=False):
    """Fit requested subset; compute max-CDF error; build display table."""
    r = np.asarray(r, dtype=float); subset = list(subset)
    A = {}
    # loc is either fitted by scipy or held at 0 (floc=0). A fit that fails, e.g. data <= 0 with
    # loc = 0 for a positive-support distribution, is skipped and reported instead of aborting Fit All.
    kw = {"floc": 0.0} if fix_loc else {}
    skipped = []
    def _fit(name, dist):
        if name not in subset: return None
        try:
            p = dist.fit(r, **kw)
        except Exception:
            p = None
        if p is None or not np.all(np.isfinite(p)):
            skipped.append(name); return None
        return p
    if (p := _fit("Normal", st.norm)) is not None: A["Normal"] = [p[0], p[1]]
    if (p := _fit("Exponential", st.expon)) is not None: A["Exponential"] = [p[0], 1.0 / p[1]]
    if (p := _fit("Gamma", st.gamma)) is not None: A["Gamma"] = [p[0], p[1], p[2]]
    if (p := _fit("Rayleigh", st.rayleigh)) is not None: A["Rayleigh"] = [p[0], p[1]]
    if (p := _fit("Weibull", st.weibull_min)) is not None: A["Weibull"] = [p[0], p[1], p[2]]

    if (p := _fit("Lognormal", st.lognorm)) is not None: A["Lognormal"] = [p[0], p[1], p[2]]  # [s, loc, scale]
    if (p := _fit("Loglogistic", st.fisk)) is not None: A["Loglogistic"] = [p[0], p[1], p[2]]# [c, loc, scale]
    if (p := _fit("Inverse Gaussian", st.invgauss)) is not None: A["Inverse Gaussian"] = [p[0], p[1], p[2]] # [mu, loc, scale]
    if (p := _fit("Beta", st.beta)) is not None: A["Beta"] = [p[0], p[1], p[2], p[3]] # [a, b, loc, scale]
    if (p := _fit("GEV", st.genextreme)) is not None: A["GEV"] = [p[0], p[1], p[2]]        # [xi, loc, scale]
    if (p := _fit("Logistic", st.logistic)) is not None: A["Logistic"] = [p[0], p[1]]         # [loc, scale]
    if (p := _fit("Laplace", st.laplace)) is not None: A["Laplace"] = [p[0], p[1]]          # [loc, scale]

    if (p := _fit("Chi-squared", st.chi2)) is not None: A["Chi-squared"] = [p[0], p[1], p[2]]
    if (p := _fit("Chi", st.chi)) is not None: A["Chi"] = [p[0], p[1], p[2]]
    if (p := _fit("Nakagami", st.nakagami)) is not None: A["Nakagami"] = [p[0], p[1], p[2]]
    if (p := _fit("Rician", st.rice)) is not None: A["Rician"] = [p[0], p[1], p[2]]
    if (p := _fit("Cauchy", st.cauchy)) is not None: A["Cauchy"] = [p[0], p[1]]
    if (p := _fit("Student-T", st.t)) is not None: A["Student-T"] = [p[0], p[1], p[2]]

    if not A:
        raise ValueError("no distribution could be fitted" +
                         (" with loc = 0; most need every value > 0" if fix_loc else ""))

    rangex, emp_cdf = pdf_cdf(r, bins, PDF=False)
    rangex = np.asarray(rangex); emp_cdf = np.asarray(emp_cdf)

    test_dist, errors = [], []
    for d in A.keys():
        th = np.asarray(theory_cdf(d, A[d], rangex))
        err = float(np.max(np.abs(th - emp_cdf)))
        test_dist.append(d)
        errors.append(min(max(err, 0.0), 1.0))
    errors = np.array(errors); imin = int(np.nanargmin(errors))
    best_name, best_error = test_dist[imin], float(errors[imin])

    rows = []
    def add_row(name, p1, p2="", p3=""): rows.append({"Distribution":name, "Parameter 1":p1, "Parameter 2":p2, "Parameter 3":p3, "Fitting Error":f"{errors[test_dist.index(name)]:.3f}"})

    if "Normal" in A:      add_row("Normal",      f"loc (μ) = {A['Normal'][0]:.6f}",          f"σ = {A['Normal'][1]:.6f}")
    if "Exponential" in A: add_row("Exponential", f"loc = {A['Exponential'][0]:.6f}",        f"λ = {A['Exponential'][1]:.6f}")
    if "Gamma" in A:       add_row("Gamma",       f"loc = {A['Gamma'][1]:.6f}",              f"α = {A['Gamma'][0]:.6f}", f"θ = {A['Gamma'][2]:.6f}")
    if "Rayleigh" in A:    add_row("Rayleigh",    f"loc = {A['Rayleigh'][0]:.6f}",           f"σ = {A['Rayleigh'][1]:.6f}")
    if "Weibull" in A:     add_row("Weibull",     f"loc = {A['Weibull'][1]:.6f}",            f"k = {A['Weibull'][0]:.6f}", f"λ = {A['Weibull'][2]:.6f}")

    if "Lognormal" in A:   add_row("Lognormal",   f"loc = {A['Lognormal'][1]:.6f}",          f"s = {A['Lognormal'][0]:.6f}", f"scale = {A['Lognormal'][2]:.6f}")
    if "Loglogistic" in A: add_row("Loglogistic", f"loc = {A['Loglogistic'][1]:.6f}",        f"c = {A['Loglogistic'][0]:.6f}", f"scale = {A['Loglogistic'][2]:.6f}")
    if "Inverse Gaussian" in A: add_row("Inverse Gaussian", f"loc = {A['Inverse Gaussian'][1]:.6f}", f"μ = {A['Inverse Gaussian'][0]:.6f}", f"scale = {A['Inverse Gaussian'][2]:.6f}")
    if "Beta" in A:        add_row("Beta",        f"loc = {A['Beta'][2]:.6f}",               f"α = {A['Beta'][0]:.6f}, β = {A['Beta'][1]:.6f}", f"scale = {A['Beta'][3]:.6f}")
    if "GEV" in A:         add_row("GEV",         f"loc = {A['GEV'][1]:.6f}",                f"ξ = {A['GEV'][0]:.6f}", f"scale = {A['GEV'][2]:.6f}")
    if "Logistic" in A:    add_row("Logistic",    f"loc = {A['Logistic'][0]:.6f}",           f"scale = {A['Logistic'][1]:.6f}")
    if "Laplace" in A:     add_row("Laplace",     f"loc = {A['Laplace'][0]:.6f}",            f"b = {A['Laplace'][1]:.6f}")

    if "Chi-squared" in A: add_row("Chi-squared", f"loc = {A['Chi-squared'][1]:.6f}",        f"ν (df) = {A['Chi-squared'][0]:.6f}", f"scale = {A['Chi-squared'][2]:.6f}")
    if "Chi" in A:         add_row("Chi",         f"loc = {A['Chi'][1]:.6f}",                f"ν (df) = {A['Chi'][0]:.6f}", f"scale = {A['Chi'][2]:.6f}")
    if "Nakagami" in A:    add_row("Nakagami",    f"loc = {A['Nakagami'][1]:.6f}",           f"m = {A['Nakagami'][0]:.6f}", f"scale = {A['Nakagami'][2]:.6f}")
    if "Rician" in A:      add_row("Rician",      f"loc = {A['Rician'][1]:.6f}",             f"ν (noncentral) = {A['Rician'][0]:.6f}", f"σ = {A['Rician'][2]:.6f}")
    if "Cauchy" in A:      add_row("Cauchy",      f"loc = {A['Cauchy'][0]:.6f}",             f"s = {A['Cauchy'][1]:.6f}")
    if "Student-T" in A:   add_row("Student-T",   f"loc = {A['Student-T'][1]:.6f}",          f"ν (df) = {A['Student-T'][0]:.6f}", f"s = {A['Student-T'][2]:.6f}")

    # Best fit first: order the table, the error chart and the saved CSV by ascending error (NaN last)
    order = sorted(range(len(test_dist)), key=lambda i: (np.isnan(errors[i]), errors[i]))
    test_dist, errors = [test_dist[i] for i in order], errors[order]
    rank = {name: i for i, name in enumerate(test_dist)}
    df = pd.DataFrame(rows).sort_values("Distribution", key=lambda s: s.map(rank), kind="stable", ignore_index=True)
    if fix_loc:   # Parameter 1 is always loc; say it was held, not fitted
        df["Parameter 1"] = df["Parameter 1"].str.replace(r"= -?0\.0+$", "= 0 (fixed)", regex=True)
    return {"params": A, "rangex": rangex, "emp_cdf": emp_cdf,
            "errors": errors, "table": df, "best_name": best_name,
            "best_error": best_error, "test_dist": test_dist, "skipped": skipped}

# ---------------- Widgets & UI ----------------
# Global controls
# Where the data comes from; the distribution list is only used for simulated data
mode_dd    = widgets.Dropdown(options=[("Simulated Random Data", "sim"), ("From file", "file")], value="sim",
                              description="Data Source:", layout=widgets.Layout(width="340px"))
source_dd  = widgets.Dropdown(
    options=["Normal","Exponential","Gamma","Rayleigh","Weibull",
             "Lognormal","Loglogistic","Inverse Gaussian","Beta","GEV","Logistic","Laplace",
             "Chi-squared","Chi","Nakagami","Rician","Cauchy","Student-T"],
    value="Normal", description="Distribution:", layout=widgets.Layout(width="280px")
)

def _is_external():
    return mode_dd.value == "file"
n_int      = widgets.BoundedIntText(value=10000, min=50, max=2_000_000, step=50,
                                    description="N:", layout=widgets.Layout(width="160px"))
# Resolution of the empirical CDF every fit is scored against - applies to file data too
bins_int   = widgets.BoundedIntText(value=100, min=20, max=5000, step=10,
                                    description="CDF bins:", style={"description_width": "initial"},
                                    tooltip="Resolution of the empirical CDF: sets the histogram detail here "
                                            "and the empirical curve that Fit and Fit All are scored against",
                                    layout=widgets.Layout(width="190px"))
seed_txt   = widgets.Text(value="", description="Seed:", layout=widgets.Layout(width="160px"))

# Per-process parameter inputs (widgets; labels are drawn in the grid)
normal_loc = widgets.FloatText(value=0.0, layout=widgets.Layout(width="140px"))
sigma_f    = widgets.FloatText(value=1.0, layout=widgets.Layout(width="140px"))
exp_loc    = widgets.FloatText(value=0.0, layout=widgets.Layout(width="140px"))
exp_scale  = widgets.FloatText(value=2.0, layout=widgets.Layout(width="140px"))
ray_loc    = widgets.FloatText(value=0.0, layout=widgets.Layout(width="140px"))
ray_sigma  = widgets.FloatText(value=1.0, layout=widgets.Layout(width="140px"))
gam_loc    = widgets.FloatText(value=0.0, layout=widgets.Layout(width="140px"))
gam_k      = widgets.FloatText(value=1.0, layout=widgets.Layout(width="140px"))
gam_theta  = widgets.FloatText(value=5.0, layout=widgets.Layout(width="140px"))
wei_loc    = widgets.FloatText(value=0.0, layout=widgets.Layout(width="140px"))
wei_k      = widgets.FloatText(value=1.0, layout=widgets.Layout(width="140px"))
wei_lmbda  = widgets.FloatText(value=2.0, layout=widgets.Layout(width="140px"))

# Extended widgets
logn_loc   = widgets.FloatText(value=0.0, layout=widgets.Layout(width="140px"))
logn_s     = widgets.FloatText(value=0.5, layout=widgets.Layout(width="140px"))
logn_scale = widgets.FloatText(value=1.0, layout=widgets.Layout(width="140px"))
fisk_loc   = widgets.FloatText(value=0.0, layout=widgets.Layout(width="140px"))
fisk_c     = widgets.FloatText(value=2.0, layout=widgets.Layout(width="140px"))
fisk_scale = widgets.FloatText(value=1.0, layout=widgets.Layout(width="140px"))
invg_loc   = widgets.FloatText(value=0.0, layout=widgets.Layout(width="140px"))
invg_mu    = widgets.FloatText(value=1.0, layout=widgets.Layout(width="140px"))
invg_scale = widgets.FloatText(value=1.0, layout=widgets.Layout(width="140px"))
beta_loc   = widgets.FloatText(value=0.0, layout=widgets.Layout(width="140px"))
beta_a     = widgets.FloatText(value=2.0, layout=widgets.Layout(width="140px"))
beta_b     = widgets.FloatText(value=5.0, layout=widgets.Layout(width="140px"))
beta_scale = widgets.FloatText(value=1.0, layout=widgets.Layout(width="140px"))
gev_loc    = widgets.FloatText(value=0.0, layout=widgets.Layout(width="140px"))
gev_xi     = widgets.FloatText(value=0.1, layout=widgets.Layout(width="140px"))
gev_scale  = widgets.FloatText(value=1.0, layout=widgets.Layout(width="140px"))
logistic_loc = widgets.FloatText(value=0.0, layout=widgets.Layout(width="140px"))
logistic_s   = widgets.FloatText(value=1.0, layout=widgets.Layout(width="140px"))
laplace_loc  = widgets.FloatText(value=0.0, layout=widgets.Layout(width="140px"))
laplace_b    = widgets.FloatText(value=1.0, layout=widgets.Layout(width="140px"))
chi2_loc   = widgets.FloatText(value=0.0, layout=widgets.Layout(width="140px"))
chi2_nu    = widgets.FloatText(value=4.0, layout=widgets.Layout(width="140px"))
chi2_scale = widgets.FloatText(value=1.0, layout=widgets.Layout(width="140px"))
chi_loc    = widgets.FloatText(value=0.0, layout=widgets.Layout(width="140px"))
chi_nu     = widgets.FloatText(value=4.0, layout=widgets.Layout(width="140px"))
chi_scale  = widgets.FloatText(value=1.0, layout=widgets.Layout(width="140px"))
naka_loc   = widgets.FloatText(value=0.0, layout=widgets.Layout(width="140px"))
naka_m     = widgets.FloatText(value=1.0, layout=widgets.Layout(width="140px"))
naka_scale = widgets.FloatText(value=1.0, layout=widgets.Layout(width="140px"))
rice_loc   = widgets.FloatText(value=0.0, layout=widgets.Layout(width="140px"))
rice_nu    = widgets.FloatText(value=1.0, layout=widgets.Layout(width="140px"))
rice_sigma = widgets.FloatText(value=1.0, layout=widgets.Layout(width="140px"))
cauchy_loc = widgets.FloatText(value=0.0, layout=widgets.Layout(width="140px"))
cauchy_s   = widgets.FloatText(value=1.0, layout=widgets.Layout(width="140px"))
t_loc      = widgets.FloatText(value=0.0, layout=widgets.Layout(width="140px"))
t_nu       = widgets.FloatText(value=5.0, layout=widgets.Layout(width="140px"))
t_s        = widgets.FloatText(value=1.0, layout=widgets.Layout(width="140px"))

# --- Multi-format uploader (styled big button), plus "load from path" ---
uploader = widgets.FileUpload(
    accept=".csv,.tsv,.psv,"
           ".xlsx,.xls,"
           ".json,.jsonl,.ndjson,"
           ".parquet,.feather,.arrow,"
           ".sav,.dta,.sas7bdat,.xpt,"
           ".h5,.hdf5,"
           ".txt,.mat",          # ← add these
    multiple=False,
    description="Upload CSV/XLSX/Parquet/...",
    tooltip="CSV, TSV, PSV, TXT, Excel, JSON/JSONL, Parquet, Feather/Arrow, "
            "SPSS, Stata, SAS, HDF5, MATLAB"
)
uploader.layout.width = "340px"           # same plain button as the discrete app

# Load-from-path controls
path_txt   = widgets.Text(placeholder="Or paste server-local path (e.g., /home/jovyan/data.parquet)",
                          layout=widgets.Layout(width="420px"))
path_btn   = widgets.Button(description="Load path", icon="folder-open", layout=widgets.Layout(width="120px"))

col_dd = widgets.Dropdown(options=[], layout=widgets.Layout(width="300px"))   # labelled by "Use column:" in the row

# File-data shaping (file mode only): a stack of conditions, cleaning, group-by
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
filter_list  = widgets.VBox([])     # one row per active condition
filter_note  = widgets.HTML("")
filter_state = widgets.HTML("")     # rows kept vs total

# Cleaning options, applied to the fitted column
clean_dropna = widgets.Checkbox(value=True,  description="drop NaN / non-numeric", indent=False, layout=widgets.Layout(width="195px"))
clean_pos    = widgets.Checkbox(value=False, description="drop values <= 0", indent=False, layout=widgets.Layout(width="155px"))
clean_dedup  = widgets.Checkbox(value=False, description="drop duplicate rows", indent=False, layout=widgets.Layout(width="175px"))
clean_trim   = widgets.Checkbox(value=False, description="trim percentiles", indent=False, layout=widgets.Layout(width="145px"))
trim_lo      = widgets.BoundedFloatText(value=1.0, min=0.0, max=49.0, step=0.5, description="low %:",
                                        style={"description_width": "initial"}, layout=widgets.Layout(width="130px"))
trim_hi      = widgets.BoundedFloatText(value=99.0, min=51.0, max=100.0, step=0.5, description="high %:",
                                        style={"description_width": "initial"}, layout=widgets.Layout(width="135px"))

# Group-by: fit each group separately, or narrow every fit to one group
group_col = widgets.Dropdown(options=["(none)"], value="(none)", description="Group by:",
                             style={"description_width": "initial"}, layout=widgets.Layout(width="250px"))
group_val = widgets.Dropdown(options=["(all groups)"], value="(all groups)", description="Group:",
                             style={"description_width": "initial"}, layout=widgets.Layout(width="230px"))
group_val.layout.display = "none"

# Buttons
fit_all_btn = widgets.Button(description="Fit All", icon="play", button_style="primary", layout=widgets.Layout(width="120px"))
# Fit All option: let scipy fit loc, or hold it at 0
loc_mode = widgets.RadioButtons(options=[("loc free", False), ("loc = 0", True)], value=False,
                                orientation="horizontal", layout=widgets.Layout(width="auto"))
save_btn    = widgets.Button(description="Save results", icon="download", disabled=True, layout=widgets.Layout(width="140px"))

status_html = widgets.HTML("")
out         = widgets.Output()

def label_right(text):
    return widgets.HTML(f"<div style='text-align:right; white-space:nowrap; padding-right:6px;'>{text}</div>")

# --- Find error controls (up to 4 params) ---
find_proc_dd= widgets.Dropdown(options=list(ALL_DISTS), value="Normal", layout=widgets.Layout(width="200px"))
p1_lbl, p2_lbl, p3_lbl, p4_lbl = label_right("loc (μ)"), label_right("σ"), label_right("—"), label_right("—")
p1_in  = widgets.FloatText(value=0.0, layout=widgets.Layout(width="120px"))
p2_in  = widgets.FloatText(value=1.0, layout=widgets.Layout(width="120px"))
p3_in  = widgets.FloatText(value=0.0, layout=widgets.Layout(width="120px"), disabled=True)
p4_in  = widgets.FloatText(value=0.0, layout=widgets.Layout(width="120px"), disabled=True)
# flex "0 0 auto": never shrink below its label; buttons clip overflowing text
find_btn       = widgets.Button(description="Fit", icon="play", button_style="primary", layout=widgets.Layout(width="auto", flex="0 0 auto"))
find_status    = widgets.HTML("")
find_out       = widgets.Output()
# Shown only while Exponential is selected in the Fit row (toggled in _update_find_labels)
exp_note = widgets.HTML("<em>Exponential note: enter λ (rate); if you have scale s, use λ = 1/s.</em>")

def _update_find_labels(*_):
    proc = find_proc_dd.value
    exp_note.layout.display = "" if proc == "Exponential" else "none"
    p1_in.disabled = p2_in.disabled = False
    p3_in.disabled = True; p4_in.disabled = True
    p3_in.value = 0.0; p4_in.value = 0.0

    mapping = {
        "Normal":           ("loc (μ)","σ",None,None),
        "Exponential":      ("loc","λ (rate)",None,None),
        "Gamma":            ("loc","α (shape)","θ (scale)",None),
        "Rayleigh":         ("loc","σ",None,None),
        "Weibull":          ("loc","k (shape)","λ (scale)",None),
        "Lognormal":        ("loc","s (shape)","scale",None),
        "Loglogistic":      ("loc","c (shape)","scale",None),
        "Inverse Gaussian": ("loc","μ (shape)","scale",None),
        "Beta":             ("loc","α (shape)","β (shape)","scale"),
        "GEV":              ("loc","ξ (shape)","scale",None),
        "Logistic":         ("loc","scale",None,None),
        "Laplace":          ("loc","b (scale)",None,None),
        "Chi-squared":      ("loc","ν (df)","scale",None),
        "Chi":              ("loc","ν (df)","scale",None),
        "Nakagami":         ("loc","m (shape)","scale",None),
        "Rician":           ("loc","ν (noncentral)","σ",None),
        "Cauchy":           ("loc","s (scale)",None,None),
        "Student-T":        ("loc","ν (df)","s (scale)",None),
    }
    a,b,c,d = mapping[proc]
    p1_lbl.value = label_right(a).value
    p2_lbl.value = label_right(b).value
    p3_lbl.value = label_right(c or "—").value
    p4_lbl.value = label_right(d or "—").value
    if c: p3_in.disabled = False
    if d: p4_in.disabled = False

def _validate_proc_params(proc, p1, p2, p3, p4):
    def pos(name, v):
        if v <= 0:
            raise ValueError(f"{name} must be > 0")
    if proc == "Normal":         pos("σ", p2)
    elif proc == "Exponential":  pos("λ (rate)", p2)
    elif proc == "Gamma":        pos("α (shape)", p2); pos("θ (scale)", p3)
    elif proc == "Rayleigh":     pos("σ", p2)
    elif proc == "Weibull":      pos("k (shape)", p2); pos("λ (scale)", p3)
    elif proc == "Lognormal":    pos("s (shape)", p2); pos("scale", p3)
    elif proc == "Loglogistic":  pos("c (shape)", p2); pos("scale", p3)
    elif proc == "Inverse Gaussian": pos("μ (shape)", p2); pos("scale", p3)
    elif proc == "Beta":         pos("α (shape)", p2); pos("β (shape)", p3); pos("scale", p4)
    elif proc == "GEV":          pos("scale", p3)
    elif proc == "Logistic":     pos("scale", p2)
    elif proc == "Laplace":      pos("b (scale)", p2)
    elif proc == "Chi-squared":  pos("ν (df)", p2); pos("scale", p3)
    elif proc == "Chi":          pos("ν (df)", p2); pos("scale", p3)
    elif proc == "Nakagami":     pos("m (shape)", p2); pos("scale", p3)
    elif proc == "Rician":       pos("ν (noncentral)", p2); pos("σ", p3)
    elif proc == "Cauchy":       pos("s (scale)", p2)
    elif proc == "Student-T":    pos("ν (df)", p2); pos("s (scale)", p3)

def _params_for_theory(proc, p1, p2, p3, p4):
    """Map UI order [loc, ...] -> theory_* order used above."""
    if proc == "Normal":          return [p1, p2]
    if proc == "Exponential":     return [p1, p2]                 # [loc, rate]
    if proc == "Gamma":           return [p2, p1, p3]             # [a, loc, th]
    if proc == "Rayleigh":        return [p1, p2]
    if proc == "Weibull":         return [p2, p1, p3]             # [k, loc, lam]
    if proc == "Lognormal":       return [p2, p1, p3]             # [s, loc, scale]
    if proc == "Loglogistic":     return [p2, p1, p3]             # [c, loc, scale]
    if proc == "Inverse Gaussian":return [p2, p1, p3]             # [mu, loc, scale]
    if proc == "Beta":            return [p2, p3, p1, p4]         # [a, b, loc, scale]
    if proc == "GEV":             return [p2, p1, p3]             # [xi, loc, scale]
    if proc == "Logistic":        return [p1, p2]                 # [loc, scale]
    if proc == "Laplace":         return [p1, p2]                 # [loc, b]
    if proc == "Chi-squared":     return [p2, p1, p3]
    if proc == "Chi":             return [p2, p1, p3]
    if proc == "Nakagami":        return [p2, p1, p3]
    if proc == "Rician":          return [p2, p1, p3]
    if proc == "Cauchy":          return [p1, p2]
    if proc == "Student-T":       return [p2, p1, p3]
    raise ValueError("Unknown process.")

# ---- baseline cache so Find error matches Fit all exactly ----
_data_cache = {"key": None, "r": None, "bins": None, "rangex": None, "emp_cdf": None}

def _synthetic_params_for(src):
    if src == "Normal":        return [normal_loc.value, sigma_f.value]
    if src == "Exponential":   return [exp_loc.value, exp_scale.value]
    if src == "Rayleigh":      return [ray_loc.value, ray_sigma.value]
    if src == "Gamma":         return [gam_loc.value, gam_k.value, gam_theta.value]
    if src == "Weibull":       return [wei_loc.value, wei_k.value, wei_lmbda.value]
    if src == "Lognormal":     return [logn_loc.value, logn_s.value, logn_scale.value]
    if src == "Loglogistic":   return [fisk_loc.value, fisk_c.value, fisk_scale.value]
    if src == "Inverse Gaussian": return [invg_loc.value, invg_mu.value, invg_scale.value]
    if src == "Beta":          return [beta_loc.value, beta_a.value, beta_b.value, beta_scale.value]
    if src == "GEV":           return [gev_loc.value, gev_xi.value, gev_scale.value]
    if src == "Logistic":      return [logistic_loc.value, logistic_s.value]
    if src == "Laplace":       return [laplace_loc.value, laplace_b.value]
    if src == "Chi-squared":   return [chi2_loc.value, chi2_nu.value, chi2_scale.value]
    if src == "Chi":           return [chi_loc.value, chi_nu.value, chi_scale.value]
    if src == "Nakagami":      return [naka_loc.value, naka_m.value, naka_scale.value]
    if src == "Rician":        return [rice_loc.value, rice_nu.value, rice_sigma.value]
    if src == "Cauchy":        return [cauchy_loc.value, cauchy_s.value]
    if src == "Student-T":     return [t_loc.value, t_nu.value, t_s.value]
    return []

def _current_data_key(bins):
    src = source_dd.value
    if _is_external():
        return ("External", col_dd.value, _filter_signature(), bins)
    else:
        P = tuple(_synthetic_params_for(src))
        return (src, int(n_int.value), P, seed_txt.value.strip(), bins)

def _validate_params(src, P):
    def pos(name, v):
        if v <= 0: raise ValueError(f"{name} must be > 0")
    if src == "Normal":         pos("σ", P[1])
    elif src == "Exponential":  pos("scale", P[1])
    elif src == "Rayleigh":     pos("σ", P[1])
    elif src == "Gamma":        pos("k (shape)", P[1]); pos("θ (scale)", P[2])
    elif src == "Weibull":      pos("k (shape)", P[1]); pos("λ (scale)", P[2])
    elif src == "Lognormal":    pos("s (shape)", P[1]); pos("scale", P[2])
    elif src == "Loglogistic":  pos("c (shape)", P[1]); pos("scale", P[2])
    elif src == "Inverse Gaussian": pos("μ (shape)", P[1]); pos("scale", P[2])
    elif src == "Beta":         pos("α (shape)", P[1]); pos("β (shape)", P[2]); pos("scale", P[3])
    elif src == "GEV":          pos("scale", P[2])
    elif src == "Logistic":     pos("scale", P[1])
    elif src == "Laplace":      pos("b (scale)", P[1])
    elif src == "Chi-squared":  pos("ν (df)", P[1]); pos("scale", P[2])
    elif src == "Chi":          pos("ν (df)", P[1]); pos("scale", P[2])
    elif src == "Nakagami":     pos("m (shape)", P[1]); pos("scale", P[2])
    elif src == "Rician":       pos("ν (noncentral)", P[1]); pos("σ", P[2])
    elif src == "Cauchy":       pos("s (scale)", P[1])
    elif src == "Student-T":    pos("ν (df)", P[1]); pos("s (scale)", P[2])

_last = {"table": None, "errors_fig": None, "cdf_fig": None, "pdf_fig": None, "samples": None}

# -------- Upload & read helpers --------
def _get_uploaded_bytes_and_name(upl: widgets.FileUpload):
    """Return (content_bytes, filename) for ipywidgets v7 (dict) or v8 (tuple of UploadedFile)."""
    v = upl.value
    if not v:
        return None, None
    if isinstance(v, dict):  # v7
        first = next(iter(v.items()))[1]
        content = first.get("content", None)
        name = first.get("name", None) or "upload"
        return content, name
    # v8
    try:
        uf = v[0]
    except Exception:
        return None, None
    content = getattr(uf, "content", None)
    name = getattr(uf, "name", None) or (uf.get("name") if isinstance(uf, dict) else "upload")
    if content is None and isinstance(uf, dict):
        content = uf.get("content", None)
    return content, name

def _use_dataframe(df, origin="upload"):
    """After loading (upload or path), wire dataframe into UI & state."""
    uploader.df_raw = df.copy()
    uploader.df = df
    rows, cols = df.shape
    # Populate column pickers; a new file starts with no filters
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    col_dd.options = numeric_cols if numeric_cols else list(df.columns)
    if col_dd.options: col_dd.value = col_dd.options[0]
    _filters.clear()
    filter_col.options = list(df.columns)
    if filter_col.options: filter_col.value = filter_col.options[0]
    group_col.options = ["(none)"] + list(df.columns)
    group_col.value = "(none)"
    _render_filters(); _refresh_group_values(df)
    # Update messages
    if origin == "upload" and 'status_html' in globals():
        status_html.value = f"<span>Loaded <b>uploaded file</b> ({rows} rows × {cols} cols).</span>"
    elif origin == "path" and 'status_html' in globals():
        status_html.value = f"<span>Loaded <b>path file</b> ({rows} rows × {cols} cols).</span>"
    filter_state.value = f"<span>Rows: <b>{rows}</b> (no filter)</span>"
    # Invalidate baseline cache
    _data_cache.update({"key": None})

def _on_upload(change):
    content, fname = _get_uploaded_bytes_and_name(uploader)
    if content is None:
        return
    try:
        df = _read_uploaded_dataframe(content, fname or "upload")
    except Exception as e:
        status_html.value = f"<span style='color:#b91c1c'>Load error: {e}</span>"
        return
    _use_dataframe(df, origin="upload")

def _on_load_path(_):
    p = path_txt.value.strip()
    if not p:
        status_html.value = "<span style='color:#b91c1c'>Enter a file path first.</span>"
        return
    if not os.path.exists(p):
        status_html.value = f"<span style='color:#b91c1c'>Path not found: {p}</span>"
        return
    try:
        with open(p, "rb") as fh:
            content = fh.read()
        df = _read_uploaded_dataframe(content, os.path.basename(p))
    except Exception as e:
        status_html.value = f"<span style='color:#b91c1c'>Load error: {e}</span>"
        return
    _use_dataframe(df, origin="path")

uploader.observe(_on_upload, names="value")
path_btn.on_click(_on_load_path)

# -------- File data: filter stack, cleaning, grouping (file mode only) --------
_filters = []        # [{"col","op","v1","v2","ci","on"}] - all enabled ones must match
_clean_note = {"text": ""}

def _cond_text(c):
    if c["op"] == "between":
        return f"{c['col']}  between  {c['v1']} .. {c['v2']}"
    return f"{c['col']}  {c['op']}  {c['v1']}"

def _mask_for(df, c):
    """Boolean mask for one condition."""
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
        rx = re.compile(v1, re.IGNORECASE if ci else 0)   # re.error on a bad pattern
        return series.astype(str).str.match(rx, na=False)
    return pd.Series([True] * len(df), index=df.index)

def _current_df():
    """Raw file rows with every enabled condition applied."""
    if not hasattr(uploader, "df_raw"):
        raise RuntimeError("Please upload or load a data file.")
    df = uploader.df_raw
    if clean_dedup.value:
        df = df.drop_duplicates()
    for c in _filters:
        if c["on"] and c["col"] in df.columns:
            df = df.loc[_mask_for(df, c)]
    return df

def _clean_series(series):
    """Numeric values ready to fit, plus a note about what was dropped."""
    n0 = len(series)
    s  = pd.to_numeric(series, errors="coerce")
    bad = int(s.isna().sum())
    if bad and not clean_dropna.value:
        raise RuntimeError(f"{bad} of {n0} values are NaN or non-numeric - "
                           "tick 'drop NaN / non-numeric' to ignore them.")
    s = s.dropna()
    dropped_pos = dropped_trim = 0
    if clean_pos.value:
        before = len(s); s = s[s > 0]; dropped_pos = before - len(s)
    if clean_trim.value and len(s):
        lo, hi = np.percentile(s, [float(trim_lo.value), float(trim_hi.value)])
        before = len(s); s = s[(s >= lo) & (s <= hi)]; dropped_trim = before - len(s)
    bits = []
    if bad:          bits.append(f"{bad} NaN/non-numeric")
    if dropped_pos:  bits.append(f"{dropped_pos} <= 0")
    if dropped_trim: bits.append(f"{dropped_trim} outside percentiles")
    note = f" Dropped {', '.join(bits)} of {n0} values." if bits else ""
    return s.to_numpy(dtype=float), note

def _filter_signature():
    """Everything that changes the fitted sample, for the baseline cache key."""
    return (tuple((c["col"], c["op"], c["v1"], c["v2"], c["ci"]) for c in _filters if c["on"]),
            bool(clean_dropna.value), bool(clean_pos.value), bool(clean_dedup.value),
            bool(clean_trim.value), float(trim_lo.value), float(trim_hi.value),
            group_col.value, group_val.value)

def _refresh_group_values(df=None):
    if group_col.value == "(none)":
        group_val.options = ["(all groups)"]; group_val.value = "(all groups)"
        group_val.layout.display = "none"
        return
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
    """Re-read the filtered frame: column choices, group values, row count."""
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
    _render_filters(); _refresh_columns(); _data_cache.update({"key": None})

def _add_filter(_b=None):
    if not hasattr(uploader, "df_raw"):
        filter_state.value = "<span style='color:#b91c1c'>Load a file first.</span>"; return
    if not filter_col.value: return
    c = {"col": filter_col.value, "op": filter_op.value, "v1": filter_val1.value,
         "v2": filter_val2.value, "ci": bool(filter_ci.value), "on": True}
    try:
        _mask_for(uploader.df_raw, c)      # validate now (bad regex, bad number, ...)
    except Exception as e:
        filter_state.value = f"<span style='color:#b91c1c'>Filter error: {e}</span>"; return
    _filters.append(c)
    _after_filter_change()

def _clear_filters(_b=None):
    _filters.clear(); _after_filter_change()

filter_add.on_click(_add_filter)
filter_clear.on_click(_clear_filters)
group_col.observe(lambda ch: (_refresh_group_values(), _data_cache.update({"key": None})), names="value")
for _w in (group_val, clean_dropna, clean_pos, clean_trim, trim_lo, trim_hi):
    _w.observe(lambda ch: _data_cache.update({"key": None}), names="value")
clean_dedup.observe(lambda ch: _after_filter_change(), names="value")


# ---------- Source params table (compact 1 row) ----------
_process_grid = None
def _update_process_row(*_):
    if _process_grid is None: return
    grid = _process_grid
    dash = widgets.HTML("<div style='text-align:center;'>—</div>")
    src  = source_dd.value
    grid[1, 0] = widgets.Label(src)

    def set_cells(loc_label, loc_widget,
                  p1_label=None, p1_widget=None,
                  p2_label=None, p2_widget=None,
                  p3_label=None, p3_widget=None):
        grid[1, 1] = label_right(loc_label)
        grid[1, 2] = loc_widget if loc_widget is not None else dash
        grid[1, 3] = label_right(p1_label) if p1_label else dash
        grid[1, 4] = p1_widget if p1_widget is not None else dash
        grid[1, 5] = label_right(p2_label) if p2_label else dash
        grid[1, 6] = p2_widget if p2_widget is not None else dash
        grid[1, 7] = label_right(p3_label) if p3_label else dash
        grid[1, 8] = p3_widget if p3_widget is not None else dash

    if src == "Normal":          set_cells("loc (μ)", normal_loc, "σ", sigma_f)
    elif src == "Exponential":   set_cells("loc", exp_loc, "scale", exp_scale)
    elif src == "Gamma":         set_cells("loc", gam_loc, "k (shape)", gam_k, "θ (scale)", gam_theta)
    elif src == "Rayleigh":      set_cells("loc", ray_loc, "σ", ray_sigma)
    elif src == "Weibull":       set_cells("loc", wei_loc, "k (shape)", wei_k, "λ (scale)", wei_lmbda)
    elif src == "Lognormal":     set_cells("loc", logn_loc, "s (shape)", logn_s, "scale", logn_scale)
    elif src == "Loglogistic":   set_cells("loc", fisk_loc, "c (shape)", fisk_c, "scale", fisk_scale)
    elif src == "Inverse Gaussian": set_cells("loc", invg_loc, "μ (shape)", invg_mu, "scale", invg_scale)
    elif src == "Beta":          set_cells("loc", beta_loc, "α (shape)", beta_a, "β (shape)", beta_b); grid[1,7]=label_right("scale"); grid[1,8]=beta_scale
    elif src == "GEV":           set_cells("loc", gev_loc, "ξ (shape)", gev_xi, "scale", gev_scale)
    elif src == "Logistic":      set_cells("loc", logistic_loc, "scale", logistic_s)
    elif src == "Laplace":       set_cells("loc", laplace_loc, "b (scale)", laplace_b)
    elif src == "Chi-squared":   set_cells("loc", chi2_loc, "ν (df)", chi2_nu, "scale", chi2_scale)
    elif src == "Chi":           set_cells("loc", chi_loc, "ν (df)", chi_nu, "scale", chi_scale)
    elif src == "Nakagami":      set_cells("loc", naka_loc, "m (shape)", naka_m, "scale", naka_scale)
    elif src == "Rician":        set_cells("loc", rice_loc, "ν (noncentral)", rice_nu, "σ", rice_sigma)
    elif src == "Cauchy":        set_cells("loc", cauchy_loc, "s (scale)", cauchy_s)
    elif src == "Student-T":     set_cells("loc", t_loc, "ν (df)", t_nu, "s (scale)", t_s)
    else:                         set_cells("—", None, "—", None, "—", None)

def make_process_table():
    global _process_grid
    grid = GridspecLayout(2, 9, grid_gap="2px")
    hdr_right = "font-weight:bold; text-align:right; white-space:nowrap; padding-right:6px;"
    hdr_left  = "font-weight:bold; text-align:left;  white-space:nowrap;"
    grid[0, 0] = widgets.HTML(f"<div style='{hdr_left}'>Source Process</div>")
    grid[0, 1] = widgets.HTML(f"<div style='{hdr_right}'>loc</div>")
    grid[0, 2] = widgets.HTML(f"<div style='{hdr_left}'>value</div>")
    grid[0, 3] = widgets.HTML(f"<div style='{hdr_right}'>param 1</div>")
    grid[0, 4] = widgets.HTML(f"<div style='{hdr_left}'>value</div>")
    grid[0, 5] = widgets.HTML(f"<div style='{hdr_right}'>param 2</div>")
    grid[0, 6] = widgets.HTML(f"<div style='{hdr_left}'>value</div>")
    grid[0, 7] = widgets.HTML(f"<div style='{hdr_right}'>param 3</div>")
    grid[0, 8] = widgets.HTML(f"<div style='{hdr_left}'>value</div>")
    _process_grid = grid
    _update_process_row()
    return grid

# ---------- Find error (CDF+PDF) ----------
def _on_find_click(_):
    results_tabs.selected_index = 1
    find_status.value = ""
    find_out.clear_output()
    bins = int(bins_int.value)
    r = rangex = emp_cdf = None
    used_cache = False
    try:
        key_now = _current_data_key(bins)
        dc = _data_cache
        if dc.get("key") == key_now and dc.get("rangex") is not None:
            r, rangex, emp_cdf = dc["r"], dc["rangex"], dc["emp_cdf"]; used_cache = True
    except Exception: pass
    if not used_cache:
        try:
            r, bins = _prepare_data()
        except Exception as e:
            find_status.value = f"<span style='color:#b91c1c'>{e}</span>"; return
        rangex, emp_cdf = pdf_cdf(np.asarray(r, dtype=float), bins=bins, PDF=False)
        _data_cache.update({"key": _current_data_key(bins), "r": np.asarray(r,float), "bins": bins,
                            "rangex": np.asarray(rangex), "emp_cdf": np.asarray(emp_cdf)})

    proc = find_proc_dd.value
    p1, p2, p3, p4 = float(p1_in.value), float(p2_in.value), float(p3_in.value), float(p4_in.value)
    try:
        _validate_proc_params(proc, p1, p2, p3, p4)
        params = _params_for_theory(proc, p1, p2, p3, p4)
    except Exception as e:
        find_status.value = f"<span style='color:#b91c1c'>Parameter error: {e}</span>"; return

    try:
        th_cdf = theory_cdf(proc, params, rangex)
        err = float(np.nanmax(np.abs(np.asarray(th_cdf, float) - np.asarray(emp_cdf, float))))
    except Exception as e:
        find_status.value = f"<span style='color:#b91c1c'>Computation error: {e}</span>"; return

    param_list = [p1, p2]
    if not p3_in.disabled: param_list.append(p3)
    if not p4_in.disabled: param_list.append(p4)
    reuse_str = "<span style='color:green'>(reused baseline)</span>" if used_cache else "<span style='color:#555'>(rebuilt baseline)</span>"
    find_status.value = (f"<span>Max CDF error for <b>{proc}</b> with parameters "
                         f"<tt>{param_list}</tt> = <b>{err:.3f}</b>. </span>{reuse_str}"
                         + _data_context())

    pdf_x, emp_pdf = pdf_cdf(np.asarray(r, dtype=float), bins=bins, PDF=True)
    th_pdf = theory_pdf(proc, params, pdf_x)
    with find_out:
        clear_output()
        fig_cdf = plt.figure(num="fig_cdf", clear=True)   # clears axes if the figure exists
        fig_cdf.set_size_inches(6.6, 3.2, forward=True)
        #fig_cdf = plt.figure(figsize=(6.6, 3.2))
        plt.plot(rangex, emp_cdf, label="Empirical CDF")
        plt.plot(rangex, th_cdf,  label=f"Theory CDF ({proc})")
        plt.legend(); plt.xlabel("x"); plt.ylabel("CDF")
        plt.title(f"Find error: {proc}  (max CDF error = {err:.3f})")
        plt.tight_layout(); # display(fig_cdf)
        fig_pdf = plt.figure(num="fig_pdf", clear=True)   # clears axes if the figure exists
        fig_pdf.set_size_inches(6.6, 3.2, forward=True)
        #fig_pdf = plt.figure(figsize=(6.6, 3.2))
        plt.plot(pdf_x, emp_pdf, label="Empirical PDF")
        plt.plot(pdf_x, th_pdf,  label=f"Theory PDF ({proc})")
        plt.legend(); plt.xlabel("x"); plt.ylabel("PDF")
        plt.title(f"PDF overlay: {proc}")
        plt.tight_layout(); # display(fig_pdf)
        _emit(fig_cdf, fig_pdf)

find_proc_dd.observe(_update_find_labels, names="value")
_update_find_labels()
find_btn.on_click(_on_find_click)

# ---------- Prepare data / source change ----------
def _on_mode_change(change=None):
    ext = _is_external()
    n_int.disabled = ext
    for w in [
        normal_loc, sigma_f, exp_loc, exp_scale, ray_loc, ray_sigma,
        gam_loc, gam_k, gam_theta, wei_loc, wei_k, wei_lmbda,
        logn_loc, logn_s, logn_scale, fisk_loc, fisk_c, fisk_scale,
        invg_loc, invg_mu, invg_scale, beta_loc, beta_a, beta_b, beta_scale,
        gev_loc, gev_xi, gev_scale, logistic_loc, logistic_s, laplace_loc, laplace_b,
        chi2_loc, chi2_nu, chi2_scale, chi_loc, chi_nu, chi_scale,
        naka_loc, naka_m, naka_scale, rice_loc, rice_nu, rice_sigma,
        cauchy_loc, cauchy_s, t_loc, t_nu, t_s
    ]: w.disabled = ext
    # Show the file controls, hide what only applies to simulated data
    upload_area.layout.display = "block" if ext else "none"
    filter_panel.layout.display = "block" if ext else "none"
    source_dd.layout.display = "none" if ext else ""
    n_int.layout.display = "none" if ext else ""
    seed_txt.layout.display = "none" if ext else ""   # seed only affects simulated sampling
    process_table.layout.display = "none" if ext else ""
    _update_process_row()
    _data_cache.update({"key": None})

mode_dd.observe(_on_mode_change, names="value")

def _prepare_data():
    src   = "External" if _is_external() else source_dd.value
    bins  = int(bins_int.value)
    seed  = seed_txt.value.strip()
    if src == "External":
        col = col_dd.value
        if not col:
            raise RuntimeError("Please choose a column.")
        df = _current_df()
        if group_col.value != "(none)" and group_val.value != "(all groups)":
            df = df.loc[df[group_col.value].astype(str) == group_val.value]
        if df.empty:
            raise RuntimeError("No rows left after filtering.")
        r, note = _clean_series(df[col])
        if len(r) < 10:
            raise RuntimeError(f"Only {len(r)} usable values after filtering and cleaning.")
        _clean_note["text"] = note
        return r, bins
    N = int(n_int.value)
    P = _synthetic_params_for(src)
    _validate_params(src, P)
    _clean_note["text"] = ""
    r = datasource(src, N, P, None, None, seed=seed if seed else None)
    return r, bins

# ---------- Fit all ----------
def _run_fit_by_group():
    """Fit every group of the group-by column and compare their best fits in one table."""
    bins, col, gcol = int(bins_int.value), col_dd.value, group_col.value
    if not col:
        status_html.value = "<span style='color:#b91c1c'>Please choose a column.</span>"; return
    try:
        df = _current_df()
    except Exception as e:
        status_html.value = f"<span style='color:#b91c1c'>{e}</span>"; return
    rows, names, errs, frames, skipped = [], [], [], [], []
    for g, sub in df.groupby(df[gcol].astype(str), sort=True):
        try:
            r, _note = _clean_series(sub[col])
        except Exception as e:
            skipped.append(f"{g} ({e})"); continue
        if len(r) < 30:
            skipped.append(f"{g} (only {len(r)} values)"); continue
        try:
            res = fit_subset_and_summarize(r, bins=bins, subset=ALL_DISTS, fix_loc=loc_mode.value)
        except Exception as e:
            skipped.append(f"{g} ({e})"); continue
        best = res["best_name"]
        prow = res["table"].loc[res["table"]["Distribution"] == best].iloc[0]
        rows.append({gcol: g, "n": len(r), "Best fit": best,
                     "Parameter 1": prow["Parameter 1"], "Parameter 2": prow["Parameter 2"],
                     "Parameter 3": prow["Parameter 3"], "Fitting Error": f"{res['best_error']:.3f}"})
        names.append(g); errs.append(float(res["best_error"]))
        frames.append(pd.DataFrame({gcol: g, "value": r}))
    if not rows:
        status_html.value = ("<span style='color:#b91c1c'>No group had 30+ usable values. "
                             + ("Skipped: " + "; ".join(skipped[:6]) if skipped else "") + "</span>")
        return
    table = pd.DataFrame(rows).sort_values("Fitting Error", kind="stable", ignore_index=True)
    order = list(np.argsort(np.asarray(errs, float)))
    names = [names[i] for i in order]; errs = [errs[i] for i in order]
    with out:
        clear_output()
        html_style = """
        <style>
        .fitted-params { border-collapse: collapse; width: 100%;
            font-family: Arial,sans-serif; font-size: 14px; }
        .fitted-params th { background:#4CAF50; color:#fff; padding:10px; text-align:left; }
        .fitted-params td { border:1px solid #ddd; padding:8px; }
        .fitted-params tr:nth-child(even) { background:#f6f6f6; }
        .fitted-params td:last-child { background:#1e3a8a; color:#fff; font-weight:bold; text-align:center; }
        .fitted-params th:last-child { background:#1e3a8a; text-align:center; }
        </style>
        """
        display(HTML(f"<h4>Best fit per group of '{gcol}'</h4>" + _data_context() + html_style +
                     table.to_html(index=False, classes="fitted-params", escape=False)))
        figg = plt.figure(num="fig_groups", clear=True)
        figg.set_size_inches(max(8.0, 0.6 * len(names)), 4.6, forward=True)
        x = np.arange(len(names))
        plt.bar(x, errs, width=0.6)
        pad = max(0.02, 0.08 * (max(errs) if errs else 1.0))
        plt.ylim(0, (max(errs) if errs else 1.0) + pad * 3)
        for i, (nm, ev) in enumerate(zip(names, errs)):
            plt.text(i, ev + pad * 0.2, f"{ev:.3f}", ha="center", va="bottom", fontsize=9)
            plt.text(i, ev + pad * 1.2, str(nm), ha="center", va="bottom", rotation=90, fontsize=9)
        plt.xticks([]); plt.ylabel("Max |Empirical CDF - Theory CDF|")
        plt.xlabel(f"Best fit per group of '{gcol}' (lower is better)")
        plt.title("Best fit error by group")
        plt.tight_layout(); _emit(figg)
        _last.update({"table": table, "errors_fig": figg, "cdf_fig": None, "pdf_fig": None,
                      "samples": pd.concat(frames, ignore_index=True)})
    save_btn.disabled = False
    skip_note = (f" <span style='color:#b45309'>Skipped: {'; '.join(skipped[:6])}.</span>" if skipped else "")
    status_html.value = (f"<span>Done. Fitted <b>{len(rows)}</b> groups of '{gcol}'. "
                         f"Pick one in <b>Group</b> to see its charts.</span>{skip_note}")

def _run_fit_for_all(_=None):
    results_tabs.selected_index = 0
    status_html.value = ""
    save_btn.disabled = True
    out.clear_output()
    if _is_external() and group_col.value != "(none)" and group_val.value == "(all groups)":
        _run_fit_by_group(); return
    try:
        r, bins = _prepare_data()
    except Exception as e:
        status_html.value = f"<span style='color:#b91c1c'>{e}</span>"; return
    try:
        result = fit_subset_and_summarize(r, bins=bins, subset=ALL_DISTS, fix_loc=loc_mode.value)
    except Exception as e:
        status_html.value = f"<span style='color:#b91c1c'>Fit error: {e}</span>"; return
    _data_cache.update({"key": _current_data_key(bins), "r": np.asarray(r,float), "bins": bins,
                        "rangex": np.asarray(result["rangex"]), "emp_cdf": np.asarray(result["emp_cdf"])})
    with out:
        clear_output()
        html_style = """
        <style>
        .fitted-params { border-collapse: collapse; width: 100%;
            font-family: Arial,sans-serif; font-size: 14px; }
        .fitted-params th { background:#4CAF50; color:#fff; padding:10px; text-align:left; }
        .fitted-params td { border:1px solid #ddd; padding:8px; }
        .fitted-params tr:nth-child(even) { background:#f6f6f6; }
        .fitted-params td:last-child { background:#1e3a8a; color:#fff; font-weight:bold; text-align:center; }
        .fitted-params th:last-child { background:#1e3a8a; text-align:center; }
        </style>
        """
        html_table = result["table"].to_html(index=False, classes="fitted-params", escape=False)
        display(HTML("<h4>Fitted Parameters</h4>" + _data_context() + html_style + html_table))
        # Figure 1: errors bar with vertical process names above bars
        names = result["test_dist"]; errs = np.asarray(result["errors"], float); x = np.arange(len(names))
        fig_width = max(8.0, 0.5 * len(names))
        fig1 = plt.figure(num="fig1", clear=True)   # clears axes if the figure exists
        fig1.set_size_inches(fig_width, 4.8, forward=True)
        #fig1 = plt.figure(figsize=(fig_width, 4.8))
        bars = plt.bar(x, errs, width=0.6)
        ymax = float(errs.max()) if errs.size else 1.0
        pad  = max(0.02, 0.08 * ymax)
        plt.ylim(0, ymax + pad * 3)
        for i, (name, err) in enumerate(zip(names, errs)):
            plt.text(i, err + pad * 0.2, f"{err:.3f}", ha="center", va="bottom", fontsize=9)
            plt.text(i, err + pad * 1.2, name, ha="center", va="bottom", rotation=90, fontsize=9)
        plt.xticks([]); plt.ylabel("Max |Empirical CDF − Theory CDF|")
        plt.xlabel(f"Best = {result['best_name']} (error = {result['best_error']:.3f})")
        plt.title("Fit errors (all distributions)")
        plt.tight_layout(); #display(fig1)

        # CDF
        fig2 = plt.figure(num="fig2", clear=True)   # clears axes if the figure exists
        fig2.set_size_inches(8, 4, forward=True)
        #fig2 = plt.figure(figsize=(8, 4))
        best_cdf = theory_cdf(result["best_name"], result["params"][result["best_name"]], result["rangex"])
        plt.plot(result["rangex"], result["emp_cdf"], label="Empirical CDF")
        plt.plot(result["rangex"], best_cdf, label=f"Theory CDF ({result['best_name']})")
        plt.legend(); plt.xlabel("x"); plt.ylabel("CDF"); plt.title("Empirical vs Best-fit Theoretical CDF")
        plt.tight_layout(); #display(fig2)
        # PDF
        fig3 = plt.figure(num="fig3", clear=True)   # clears axes if the figure exists
        fig3.set_size_inches(8, 4, forward=True)
        #fig3 = plt.figure(figsize=(8, 4))
        pdf_x, emp_pdf = pdf_cdf(np.asarray(r, dtype=float), bins=bins, PDF=True)
        best_pdf = theory_pdf(result["best_name"], result["params"][result["best_name"]], pdf_x)
        plt.plot(pdf_x, emp_pdf, label="Empirical PDF")
        plt.plot(pdf_x, best_pdf, label=f"Theory PDF ({result['best_name']})")
        plt.legend(); plt.xlabel("x"); plt.ylabel("PDF"); plt.title("Empirical vs Best-fit Theoretical PDF")
        plt.tight_layout(); #display(fig3)
        _emit(fig1, fig2, fig3)
        _last.update({"table": result["table"], "errors_fig": fig1, "cdf_fig": fig2, "pdf_fig": fig3,
                      "samples": pd.DataFrame({"samples": np.asarray(r, dtype=float)})})
    save_btn.disabled = False
    sel = ", ".join(result["test_dist"])
    loc_note = " with loc fixed at 0" if loc_mode.value else ""
    skip_note = (f" <span style='color:#b45309'>Could not fit{loc_note}: {', '.join(result['skipped'])}.</span>"
                 if result["skipped"] else "")
    status_html.value = (f"<span>Done{loc_note}. Best among {sel}: <b>{result['best_name']}</b> "
                         f"(error = {result['best_error']:.3f}).</span>{skip_note}")

fit_all_btn.on_click(_run_fit_for_all)

# ---------- Save results ----------
def save_results(_):
    if _last["table"] is None:
        status_html.value = "<span style='color:#b91c1c'>Run a fit first.</span>"; return
    import time
    ts = time.strftime("%Y%m%d_%H%M%S"); base = os.path.join("outputs", f"fit_outputs_{ts}"); os.makedirs(base, exist_ok=True)
    table_path, samples_path = f"{base}/fitted_parameters.csv", f"{base}/samples.csv"
    _last["table"].to_csv(table_path, index=False)
    _last["samples"].to_csv(samples_path, index=False)
    links = [FileLink(table_path, result_html_prefix="Fitted parameters: "),
             FileLink(samples_path, result_html_prefix="Samples: ")]
    for key, fname, label in (("errors_fig", "fit_errors.png", "Fit errors plot: "),
                              ("cdf_fig", "cdf_overlay.png", "CDF overlay plot: "),
                              ("pdf_fig", "pdf_overlay.png", "PDF overlay plot: ")):
        fig = _last.get(key)
        if fig is None:      # group mode has no single overlay
            continue
        path = f"{base}/{fname}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        links.append(FileLink(path, result_html_prefix=label))
    status_html.value = "<b>Saved.</b> Open: " + " | ".join([l._repr_html_() for l in links])
save_btn.on_click(save_results)

# --- Build compact process table (header + one dynamic row) ---
def make_process_table():
    global _process_grid
    grid = GridspecLayout(2, 9, grid_gap="2px")
    hdr_right = "font-weight:bold; text-align:right; white-space:nowrap; padding-right:6px;"
    hdr_left  = "font-weight:bold; text-align:left;  white-space:nowrap;"
    grid[0, 0] = widgets.HTML(f"<div style='{hdr_left}'>Source Process</div>")
    grid[0, 1] = widgets.HTML(f"<div style='{hdr_right}'>loc</div>")
    grid[0, 2] = widgets.HTML(f"<div style='{hdr_left}'>value</div>")
    grid[0, 3] = widgets.HTML(f"<div style='{hdr_right}'>param 1</div>")
    grid[0, 4] = widgets.HTML(f"<div style='{hdr_left}'>value</div>")
    grid[0, 5] = widgets.HTML(f"<div style='{hdr_right}'>param 2</div>")
    grid[0, 6] = widgets.HTML(f"<div style='{hdr_left}'>value</div>")
    grid[0, 7] = widgets.HTML(f"<div style='{hdr_right}'>param 3</div>")
    grid[0, 8] = widgets.HTML(f"<div style='{hdr_left}'>value</div>")
    _process_grid = grid
    _update_process_row()
    return grid

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
        text = f"Simulated {source_dd.value}, N = {int(n_int.value)}" + (f", seed {seed}" if seed else "")
    return f"<div style='color:#555; margin: 2px 0 6px 0;'>{esc(text)}</div>" if as_html else text

# ---------- Data visualisation ----------
viz_out    = widgets.Output()
viz_status = widgets.HTML("")
def _viz_btn(label):
    return widgets.Button(description=label, layout=widgets.Layout(width="auto", flex="0 0 auto"))
viz_hist  = _viz_btn("Histogram")
viz_ecdf  = _viz_btn("ECDF")
viz_box   = _viz_btn("Box plot")
viz_qq    = _viz_btn("Q-Q plot")
viz_run   = _viz_btn("Run chart")
viz_stats = _viz_btn("Summary stats")
viz_split = widgets.Checkbox(value=False, description="split by group", indent=False,
                             layout=widgets.Layout(width="140px"))
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

def _emit(*figs, live=False):
    """Show figures: a live canvas when `live`, otherwise a static PNG."""
    for fig in figs:
        if live and _interactive_backend():
            display(fig.canvas)          # the canvas is the widget - leave the figure open
        else:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=fig.get_dpi(), bbox_inches="tight")
            display(Image(data=buf.getvalue()))
            plt.close(fig)


def _data_by_group():
    """[(label, values)] - one entry per group when splitting, otherwise one entry."""
    if viz_split.value and _is_external() and group_col.value != "(none)":
        df = _current_df()
        groups = []
        for g, sub in df.groupby(df[group_col.value].astype(str), sort=True):
            try:
                v, _n = _clean_series(sub[col_dd.value])
            except Exception:
                continue
            if len(v) >= 5:
                groups.append((str(g), v))
        if groups:
            return groups[:10]
    r, _bins = _prepare_data()
    return [("data", r)]

def _viz_start(title):
    """Select the Data view tab, clear it, and report what is being shown."""
    results_tabs.selected_index = 2
    viz_out.clear_output()
    viz_status.value = f"<span>{title}</span>" + _data_context()

def _viz_guard(fn):
    """Run a plot handler, showing any problem in the panel instead of raising."""
    def wrapped(_b=None):
        try:
            fn()
        except Exception as e:
            viz_out.clear_output()
            viz_status.value = f"<span style='color:#b91c1c'>{e}</span>"
    return wrapped

def _viz_hist():
    sets = _data_by_group()
    _viz_start(f"Histogram of <b>{'the fitted sample' if len(sets) == 1 else str(len(sets)) + ' groups'}</b>")
    nbins = min(int(bins_int.value), 120)
    with viz_out:
        clear_output()
        fig = plt.figure(num="viz_hist", clear=True); fig.set_size_inches(8, 4.2, forward=True)
        for label, v in sets:
            plt.hist(v, bins=nbins, alpha=0.55 if len(sets) > 1 else 0.85,
                     density=True, label=f"{label} (n={len(v)})")
        plt.xlabel("value"); plt.ylabel("density"); plt.title("Histogram")
        plt.legend(fontsize=8)
        plt.tight_layout(); _emit(fig)

def _viz_ecdf():
    sets = _data_by_group()
    _viz_start("Empirical CDF")
    with viz_out:
        clear_output()
        fig = plt.figure(num="viz_ecdf", clear=True); fig.set_size_inches(8, 4.2, forward=True)
        for label, v in sets:
            xs = np.sort(v); ys = np.arange(1, len(xs) + 1) / len(xs)
            plt.step(xs, ys, where="post", label=f"{label} (n={len(xs)})")
        plt.xlabel("value"); plt.ylabel("F(x)"); plt.title("Empirical CDF")
        plt.legend(fontsize=8); plt.tight_layout(); _emit(fig)

def _viz_box():
    sets = _data_by_group()
    _viz_start("Box plot")
    with viz_out:
        clear_output()
        fig = plt.figure(num="viz_box", clear=True)
        fig.set_size_inches(max(6.0, 1.4 * len(sets) + 3), 4.2, forward=True)
        plt.boxplot([v for _l, v in sets], tick_labels=[l for l, _v in sets], showmeans=True)
        plt.ylabel("value"); plt.title("Box plot"); plt.grid(axis="y", alpha=0.3)
        plt.tight_layout(); _emit(fig)

def _viz_qq():
    proc = find_proc_dd.value
    p1, p2, p3, p4 = float(p1_in.value), float(p2_in.value), float(p3_in.value), float(p4_in.value)
    _validate_proc_params(proc, p1, p2, p3, p4)
    params = _params_for_theory(proc, p1, p2, p3, p4)
    sets = _data_by_group()
    _viz_start(f"Q-Q plot against <b>{proc}</b> with the parameters in the Fit row")
    with viz_out:
        clear_output()
        fig = plt.figure(num="viz_qq", clear=True); fig.set_size_inches(5.6, 5.4, forward=True)
        lo = hi = None
        for label, v in sets:
            xs = np.sort(v); n = len(xs)
            q  = (np.arange(1, n + 1) - 0.5) / n
            th = np.asarray(theory_ppf(proc, params, q), dtype=float)
            ok = np.isfinite(th) & np.isfinite(xs)
            plt.plot(th[ok], xs[ok], ".", ms=3, label=f"{label} (n={n})")
            if ok.any():
                a, b = float(min(th[ok].min(), xs[ok].min())), float(max(th[ok].max(), xs[ok].max()))
                lo = a if lo is None else min(lo, a); hi = b if hi is None else max(hi, b)
        if lo is not None:
            plt.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
        plt.xlabel(f"{proc} quantiles"); plt.ylabel("sample quantiles")
        plt.title("Q-Q plot"); plt.legend(fontsize=8); plt.tight_layout(); _emit(fig)

def _viz_run():
    r, _bins = _prepare_data()
    _viz_start("Run chart - values in row order, to show drift or steps")
    with viz_out:
        clear_output()
        fig = plt.figure(num="viz_run", clear=True); fig.set_size_inches(9, 3.6, forward=True)
        plt.plot(np.arange(len(r)), r, lw=0.7)
        plt.axhline(float(np.mean(r)), color="orange", lw=1, label=f"mean = {np.mean(r):.3f}")
        plt.xlabel("row order"); plt.ylabel("value"); plt.title("Run chart")
        plt.legend(fontsize=8); plt.tight_layout(); _emit(fig, live=True)

def _viz_stats():
    sets = _data_by_group()
    _viz_start("Summary statistics")
    rows = []
    for label, v in sets:
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        rows.append({"group": label, "n": len(v), "mean": f"{np.mean(v):.4f}", "std": f"{np.std(v, ddof=1):.4f}",
                     "min": f"{np.min(v):.4f}", "25%": f"{q1:.4f}", "median": f"{med:.4f}",
                     "75%": f"{q3:.4f}", "max": f"{np.max(v):.4f}",
                     "skew": f"{float(st.skew(v)):.3f}", "kurtosis": f"{float(st.kurtosis(v)):.3f}"})
    with viz_out:
        clear_output()
        style = """<style>.viz-stats{border-collapse:collapse;font-family:Arial,sans-serif;font-size:14px;}
        .viz-stats th{background:#4CAF50;color:#fff;padding:8px;text-align:left;}
        .viz-stats td{border:1px solid #ddd;padding:6px;}
        .viz-stats tr:nth-child(even){background:#f6f6f6;}</style>"""
        display(HTML(style + pd.DataFrame(rows).to_html(index=False, classes="viz-stats", escape=False)))

viz_hist.on_click(_viz_guard(_viz_hist));   viz_ecdf.on_click(_viz_guard(_viz_ecdf))
viz_box.on_click(_viz_guard(_viz_box));     viz_qq.on_click(_viz_guard(_viz_qq))
viz_run.on_click(_viz_guard(_viz_run));     viz_stats.on_click(_viz_guard(_viz_stats))

def _sep(label):
    """Captioned rule marking the start of a block of the UI."""
    return widgets.HTML(
        "<div style='border-top:1px solid #d9d9d9; margin:10px 0 4px 0; padding-top:4px;"
        " font-size:11px; letter-spacing:.06em; text-transform:uppercase; color:#8a8a8a;'>"
        f"{label}</div>")

# ---------- Layout blocks ----------
top_row    = widgets.HBox([source_dd, n_int, seed_txt])

# Upload & path area
upload_area = widgets.VBox([uploader,
                            widgets.HBox([path_txt, path_btn]),
                            widgets.HBox([widgets.HTML("<b>Use column:</b>"), col_dd])])
upload_area.layout.display = "none"  # toggled by the Data Source dropdown

# Filter area (External)
filter_row1  = widgets.HBox([filter_add, filter_col, filter_op, filter_val1, filter_val2, filter_ci])
filter_row2  = widgets.HBox([filter_clear, filter_state])
clean_row    = widgets.HBox([widgets.HTML("<b style='white-space:nowrap'>Clean:</b>", layout=widgets.Layout(width="58px")),
                             clean_dropna, clean_pos, clean_dedup, clean_trim, trim_lo, trim_hi])
group_row    = widgets.HBox([group_col, group_val])
filter_panel = widgets.VBox([filter_row1, filter_list, filter_row2, clean_row, group_row])
filter_panel.layout.display = "none"

process_table = make_process_table()

find_row = widgets.HBox([find_btn, find_proc_dd,
                         p1_lbl, p1_in,
                         p2_lbl, p2_in,
                         p3_lbl, p3_in,
                         p4_lbl, p4_in])

controls_row = widgets.HBox([fit_all_btn, loc_mode, save_btn])

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
viz_row = widgets.HBox([bins_int, viz_hist, viz_ecdf, viz_box, viz_qq, viz_run, viz_stats, viz_split])

results_tabs = widgets.Tab(children=[out, widgets.VBox([find_status, find_out]),
                                     widgets.VBox([viz_status, viz_out])])
results_tabs.set_title(0, "All-distribution results")   # filled by Fit All
results_tabs.set_title(1, "Single-distribution results")  # filled by Fit
results_tabs.set_title(2, "Data view")                    # filled by the Visualize buttons

_ui = widgets.VBox([
    _tab_css,                 # widen the result tab headers
    _sep("Data source"),
    mode_dd,                  # Simulated | From file
    top_row,
    upload_area,              # file mode only
    filter_panel,             # file mode only
    process_table,            # compact single-row param table
    _sep("Fit one distribution"),
    find_row,
    exp_note,                 # only while Exponential is selected
    _sep("Fit all distributions"),
    controls_row,             # Fit All + loc option + Save results
    _sep("Visualize data"),
    viz_row,                  # plots of the data itself
    _sep("Results"),
    status_html,              # data loading, Fit All summary, Save links
    results_tabs,             # three result tabs
])

# Hook observers & show UI
source_dd.observe(_update_process_row, names="value")
_update_find_labels()
_update_process_row()
_on_mode_change()
def main():
    """Display the app. The run chart is interactive; other figures are images."""
    setup_figures()
    display(_ui)

if "__main__" == __name__:
    main()