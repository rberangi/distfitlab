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
pip install git+https://github.com/rberangi/distfitlab.git
```

Optional readers and the interactive run chart are extras:

```bash
pip install "distfitlab[all] @ git+https://github.com/rberangi/distfitlab.git"
```

| Extra | Adds |
|---|---|
| `excel` | `.xlsx`, `.xls` |
| `parquet` | `.parquet`, `.feather`, `.arrow` |
| `spss` | `.sav`, `.dta`, `.sas7bdat`, `.xpt` |
| `hdf5` | `.h5`, `.hdf5` |
| `zoom` | pan/zoom toolbar on the run chart |
| `all` | all of the above |

## Quick start

```python
from distfitlab import main

main()                # opens on the continuous app
# main("Discrete")    # opens on the discrete app
```

Run it in JupyterLab, Notebook 7 or VS Code — anywhere `ipywidgets` renders. A ready
notebook is in [`examples/quickstart.ipynb`](examples/quickstart.ipynb).

## What you get

- **Fit All** fits every distribution and ranks them best-first by the largest gap
  between the empirical and theoretical CDF, with the parameters for each.
- **Fit** scores one distribution against parameters you type, with CDF/PDF overlays.
- **Data view** — histogram (or counts bar chart), ECDF, box plot, Q-Q plot against the
  distribution you chose, run chart with a pan/zoom toolbar, and summary statistics.
- **File data** — CSV, TSV, Excel, JSON, Parquet, Feather, SPSS/Stata/SAS and HDF5; a
  stack of filter conditions across columns; cleaning (drop NaN, drop ≤ 0, percentile
  trim, drop duplicates); and **Group by**, which fits every group and ranks them.
- **loc = 0** option, for when a distribution should be anchored at the origin.
- **Save results** writes the table, the samples and the figures to a timestamped folder.

## Requirements

Python 3.9+, with `numpy`, `pandas`, `scipy`, `matplotlib` and `ipywidgets`. Developed
and tested on Python 3.12.

## Documentation

The full walkthrough — every control, the parameter conventions, supported file types,
troubleshooting and how the fitting works — is in [`docs/guide.md`](docs/guide.md).

## License

MIT — see [LICENSE](LICENSE).

## Citation

If this helps with published work, please cite the libraries it stands on: SciPy, NumPy,
pandas, Matplotlib and ipywidgets.
