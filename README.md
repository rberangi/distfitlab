# distfitlab

Interactive distribution fitting in Jupyter. Simulate or load data, fit many
probability distributions at once, see which one fits best, and inspect the data —
from a single widget UI.

Two apps behind one switch:

- **Continuous** — 18 distributions: Normal, Exponential, Gamma, Rayleigh, Weibull,
  Lognormal, Loglogistic, Inverse Gaussian, Beta, GEV, Logistic, Laplace, Chi-squared,
  Chi, Nakagami, Rician, Cauchy, Student-T. Fitted with `scipy.stats`.
- **Discrete** — 5 count distributions: Poisson, Binomial, Geometric, Negative Binomial,
  Zero-Inflated Poisson. SciPy has no `.fit` for these, so the fitters are written here.

## Install

```bash
pip install distfitlab
```

Optional readers and the interactive run charts are extras:

```bash
pip install "distfitlab[all]"
```

For the latest development version, install from GitHub instead:
`pip install "distfitlab[all] @ git+https://github.com/rberangi/distfitlab.git"`.

| Extra | Adds |
|---|---|
| `excel` | `.xlsx`, `.xls` |
| `parquet` | `.parquet`, `.feather`, `.arrow` |
| `spss` | `.sav`, `.dta`, `.sas7bdat`, `.xpt` |
| `hdf5` | `.h5`, `.hdf5` |
| `zoom` | pan/zoom toolbar on the run charts |
| `all` | all of the above |

## After installing

distfitlab is a widget app, so it runs inside a notebook — not in a plain Python script
or terminal.

**1. Open a notebook** in the same Python environment you installed into:

```bash
pip install jupyterlab      # skip if you already have Jupyter
jupyter lab
```

VS Code works too: create a `.ipynb` file and select that Python as the kernel.
Notebook 7 is also fine — anywhere `ipywidgets` renders.

**2. Start the app** in a notebook cell:

```python
from distfitlab import main

main()                # opens on the continuous app
# main("Discrete")    # opens on the discrete app
```

**3. Use the UI** that appears below the cell:

1. **Data source** — simulate data, or choose **From file** to upload a file (or enter a
   path) and pick the column.
2. **Fit All** — fit every distribution and get a table ranked best-first.
3. **Fit** — try one distribution with parameters you type.
4. **Visualize data** — histogram, ECDF, box plot, Q-Q plot, run charts and summary stats.
5. **Save results** — write the table and figures to an `outputs/` folder next to the
   notebook.

A ready notebook is in [`examples/quickstart.ipynb`](https://github.com/rberangi/distfitlab/blob/main/examples/quickstart.ipynb)
(in the repository; a pip install does not include it), and every control is explained in
[`docs/guide.md`](https://github.com/rberangi/distfitlab/blob/main/docs/guide.md).

### If something doesn't work

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'distfitlab'` | The notebook kernel is a different Python from the one pip installed into. Run `%pip install "distfitlab[all]"` in a cell, then restart the kernel. |
| Text such as `VBox(children=...)` instead of the UI | The frontend can't render widgets. Use JupyterLab 3+, Notebook 7, or VS Code with the Jupyter extension. |
| No pan/zoom toolbar on the run charts | `ipympl` is missing. Install the `zoom` or `all` extra, then restart the kernel. |

## What you get

- **Fit All** fits every distribution and ranks them best-first by the largest gap
  between the empirical and theoretical CDF, with the parameters and AIC/BIC for each.
- **Fit** scores one distribution against parameters you type, with CDF/PDF overlays.
- **Data view** — histogram, ECDF, empirical PDF (counts: empirical PMF), box plot, Q-Q plot
  against the distribution you chose, run chart and sorted run chart, and summary statistics.
  ECDF, PDF/PMF and the run charts have a pan/zoom toolbar and optional grid lines.
- **File data** — CSV, TSV, Excel, JSON, Parquet, Feather, SPSS/Stata/SAS and HDF5; a
  stack of filter conditions across columns; cleaning (drop NaN, drop ≤ 0, percentile
  trim, drop duplicates); and **Group by**, which fits every group and ranks them.
- **loc = 0** option, for when a distribution should be anchored at the origin.
- **Conditional probability** — P(event | filters) for an event on the Use column, e.g.
  P(count > 500 | season == summer), with its all-rows baseline, the ratio, Bayes' reverse
  P(condition | event), and the model's value.
- **Save results** writes the table, the samples and the figures to a timestamped folder.

## Requirements

Python 3.9+, with `numpy`, `pandas`, `scipy`, `matplotlib` and `ipywidgets`. Developed
and tested on Python 3.12.

## Documentation

The full walkthrough — every control, the parameter conventions, supported file types,
troubleshooting and how the fitting works — is in [`docs/guide.md`](https://github.com/rberangi/distfitlab/blob/main/docs/guide.md).

## License

MIT — see [LICENSE](https://github.com/rberangi/distfitlab/blob/main/LICENSE).

## Citation

If this helps with published work, please cite the libraries it stands on: SciPy, NumPy,
pandas, Matplotlib and ipywidgets.
