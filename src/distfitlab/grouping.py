"""Group values in the order people expect: 1, 2, ..., 10 - not the text order 1, 10, 2.

Shared by both apps. Groups are labelled by their text form (that is what the Group dropdown
shows and matches on), so sorting those labels as text puts month 10 before month 2. The key
below reads a label as a number when it is one, and otherwise compares text with its digit
runs as numbers, so "press-2" comes before "press-10".
"""
import re


def natural_key(label):
    """Sort key: numbers by value first, then text with embedded numbers compared as numbers."""
    s = str(label)
    try:
        return (0, float(s), ())
    except ValueError:
        parts = re.split(r"(\d+)", s.lower())            # alternates text, digits, text, ...
        return (1, 0.0, tuple(int(p) if p.isdigit() else p for p in parts))


def groups_in_order(df, col):
    """[(label, rows)] for each non-missing value of `col`, labels as text, in natural order."""
    labels = df[col].astype(str).where(df[col].notna())
    return sorted(df.groupby(labels, sort=False), key=lambda kv: natural_key(kv[0]))


def values_in_order(series, limit=200):
    """Distinct non-missing values of `series` as text, in natural order - for the Group dropdown."""
    vals = {str(v) for v in series.dropna().unique().tolist()}
    return sorted(vals, key=natural_key)[:limit]
