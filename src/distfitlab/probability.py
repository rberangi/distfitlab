"""Probability of an event on the fitted column, conditional on the rows the filters keep.

Shared by both apps so they answer the same way. Pure: arrays and numbers in, numbers and
an HTML string out - no widgets and no app state.

    P(E | condition)   share of the kept rows where the event holds, with a Wilson 95% interval
    P(E)               the same share over every row, for comparison
    P(condition)       share of all rows the condition keeps - the missing term in Bayes' rule
    P(condition | E)   of the rows where the event holds, the share the condition keeps (Bayes)
    model P(E)         the event's probability under a fitted distribution

All four data quantities are plug-in counts on the same rows, so Bayes' rule closes exactly:
P(condition | E) P(E) = P(E | condition) P(condition) = k / N. That holds only while the kept
rows are a subset of all rows, which is what `subset` records.
"""
import html
import math
import numpy as np

EVENT_OPS = ("<=", "<", ">", ">=", "between", "==")

# The model row and the button that sets the parameters it reads share one cyan, so the table
# points back at the control that produced the number. Both apps import these.
FIT_COLOR = "#0097A7"        # cyan 700 - dark enough for white button text
FIT_TINT  = "#e0f7fa"        # the same cyan, washed out, for the rest of the model row
FIT_LABEL = "Check my parameters"   # the button's description, referred to in the model row's note


def event_mask(x, op, a, b=None):
    """Boolean array: where the event `x op a` (or a <= x <= b for "between") holds."""
    x = np.asarray(x, dtype=float)
    if op == "<=": return x <= a
    if op == "<":  return x < a
    if op == ">":  return x > a
    if op == ">=": return x >= a
    if op == "==": return np.isclose(x, a)
    if op == "between":
        lo, hi = sorted((a, b))
        return (x >= lo) & (x <= hi)
    raise ValueError(f"Unknown event operator {op!r}.")


def event_text(col, op, a, b=None):
    """Human-readable event, e.g. "count > 500" or "2 <= hours <= 8"."""
    if op == "between":
        lo, hi = sorted((a, b))
        return f"{lo:g} ≤ {col} ≤ {hi:g}"
    return f"{col} {op.replace('<=', '≤').replace('>=', '≥')} {a:g}"


def wilson_interval(k, n, z=1.959964):
    """95% Wilson score interval for a proportion k/n - stays inside [0, 1] even near 0 or 1."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def model_probability(cdf, op, a, b=None, discrete=False):
    """P(event) under a distribution given by its CDF, a callable on arrays.

    For counts the CDF is a step function, so strict and non-strict bounds differ:
    P(X < a) = F(ceil(a) - 1), P(X <= a) = F(floor(a)). For a continuous distribution
    they coincide and P(X == a) is 0.
    """
    def F(t):
        if discrete and t < 0:
            return 0.0                       # below every count's support
        return float(np.asarray(cdf(np.asarray([t])), dtype=float)[0])

    if discrete:
        below_or_at = lambda t: F(math.floor(t))           # P(X <= t)
        below = lambda t: F(math.ceil(t) - 1)              # P(X < t)
        if op == "<=": p = below_or_at(a)
        elif op == "<": p = below(a)
        elif op == ">": p = 1.0 - below_or_at(a)
        elif op == ">=": p = 1.0 - below(a)
        elif op == "==": p = F(a) - F(a - 1) if float(a).is_integer() else 0.0
        elif op == "between":
            lo, hi = sorted((a, b)); p = below_or_at(hi) - below(lo)
        else:
            raise ValueError(f"Unknown event operator {op!r}.")
    else:
        if op in ("<=", "<"): p = F(a)
        elif op in (">", ">="): p = 1.0 - F(a)
        elif op == "==": p = 0.0
        elif op == "between":
            lo, hi = sorted((a, b)); p = F(hi) - F(lo)
        else:
            raise ValueError(f"Unknown event operator {op!r}.")
    return min(max(p, 0.0), 1.0)


def conditional_summary(x_cond, x_all, op, a, b=None, subset=True):
    """Counts and probabilities for the event within the kept rows and, if given, over all rows.

    `x_all` is None when nothing narrows the data (no filter, no single group): there is no
    condition to compare against. `subset` says whether the kept rows are a subset of all
    rows after cleaning - False when percentile trimming ran separately on each - and only
    then are P(condition) and P(condition | E) reported.
    """
    m = event_mask(x_cond, op, a, b)
    k, n = int(m.sum()), int(m.size)
    res = {"k": k, "n": n, "p": k / n if n else math.nan, "ci": wilson_interval(k, n),
           "k_all": None, "n_all": None, "p_all": None, "ratio": None,
           "p_cond": None, "p_cond_given_e": None}
    if x_all is not None:
        ma = event_mask(x_all, op, a, b)
        K, N = int(ma.sum()), int(ma.size)
        res.update(k_all=K, n_all=N, p_all=K / N if N else math.nan)
        res["ratio"] = res["p"] / res["p_all"] if res["p_all"] else math.nan
        res["p_cond"] = (n / N) if (subset and N) else math.nan
        res["p_cond_given_e"] = (k / K) if (subset and K) else math.nan
    return res


def _p(v):
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.3f}"


def probability_table_html(event, condition, res, model_label=None, model_p=None, note=""):
    """The result as a small HTML table, styled like the apps' other result tables."""
    esc = lambda t: html.escape(str(t))
    cond = condition or "none"
    rows = []          # (quantity, value, based-on, row css class)
    lo, hi = res["ci"]
    if res["n_all"] is not None:
        rows.append((f"P(E | condition)", _p(res["p"]),
                     f"{res['k']:,} of {res['n']:,} kept rows · 95% interval {_p(lo)}–{_p(hi)}"))
        rows.append(("P(E) over all rows", _p(res["p_all"]), f"{res['k_all']:,} of {res['n_all']:,} rows"))
        pc = res.get("p_cond")
        pc = math.nan if pc is None else pc
        rows.append(("P(condition)", _p(pc),
                     f"{res['n']:,} of {res['n_all']:,} rows kept by the condition" if not math.isnan(pc)
                     else "not shown while trim percentiles is on: kept rows are no longer a subset"))
        ratio = res["ratio"]
        rows.append(("Ratio P(E | condition) / P(E)",
                     "—" if ratio is None or math.isnan(ratio) else f"{ratio:.2f}×",
                     "above 1: the condition makes the event more likely"))
        pce = res["p_cond_given_e"]
        rows.append(("P(condition | E)", _p(pce),
                     f"{res['k']:,} of the {res['k_all']:,} rows where E holds" if not math.isnan(pce)
                     else "not shown while trim percentiles is on: kept rows are no longer a subset"))
    else:
        rows.append(("P(E)", _p(res["p"]),
                     f"{res['k']:,} of {res['n']:,} rows · 95% interval {_p(lo)}–{_p(hi)}"))
    if model_label:
        rows.append(("P(E) under the model", _p(model_p), model_label, "model-row"))
    body = "".join(f"<tr class='{r[3] if len(r) > 3 else ''}'>"
                   f"<td>{esc(r[0])}</td><td class='pv'>{esc(r[1])}</td><td>{esc(r[2])}</td></tr>"
                   for r in rows)
    title = (f"P( {esc(event)} | {esc(cond)} )" if res["n_all"] is not None else f"P( {esc(event)} )")
    style = """<style>
    .cond-prob { border-collapse: collapse; font-family: Arial, sans-serif; font-size: 14px; min-width: 60%; }
    .cond-prob th { background:#4CAF50; color:#fff; padding:8px 10px; text-align:left; }
    .cond-prob td { border:1px solid #ddd; padding:7px 10px; }
    .cond-prob tr:nth-child(even) { background:#f6f6f6; }
    .cond-prob td.pv { font-weight:bold; text-align:right; font-variant-numeric: tabular-nums; }
    /* the model row carries the cyan of the button that sets the parameters it reads */
    .cond-prob tr.model-row td { background:""" + FIT_TINT + """; }
    .cond-prob tr.model-row td:first-child { background:""" + FIT_COLOR + """; color:#fff; font-weight:bold; }
    .cond-prob tr.model-row td.pv { color:""" + FIT_COLOR + """; }
    </style>"""
    extra = f"<div style='color:#555; margin-top:6px;'>{esc(note)}</div>" if note else ""
    return (style + f"<h4 style='margin:4px 0 8px 0;'>{title}</h4>"
            + "<table class='cond-prob'><tr><th>Quantity</th><th>Value</th><th>Based on</th></tr>"
            + body + "</table>" + extra)


def formula_html(event, condition):
    """The probability the current settings define, and how it is computed, as a small box.

    P( E | C ) = P( E and C ) / P( C ) when a condition is set, otherwise just P( E ).
    """
    esc = lambda t: html.escape(str(t))
    box = ("<div style='font-family: Menlo, Consolas, monospace; font-size: 13.5px; line-height: 1.6;"
           " background:#f6f8fa; border:1px solid #d0d7de; border-left:3px solid #2a78d6;"
           " border-radius:4px; padding:8px 12px; margin:4px 0; max-width: 100%; overflow-x:auto;'>")
    if condition:
        top = f"<b>P( {esc(event)} | {esc(condition)} )</b>"
        how = (f"= P( {esc(event)} and {esc(condition)} ) / P( {esc(condition)} )")
    else:
        top = f"<b>P( {esc(event)} )</b>"
        how = "no filter or single group is set, so this is not conditional yet"
    return box + top + f"<br><span style='color:#57606a'>{how}</span></div>"
