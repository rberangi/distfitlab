# distfitlab — user guide

This is the full walkthrough. For installation and a one-minute start, see the
[README](../README.md).

## Using the UI
0) Pick Continuous or Discrete with the switch at the top.

1) Data source  (on-screen block: DATA SOURCE)

Pick **Data Source**, which has two options:

Simulated Random Data: choose a **Distribution** and N; a single compact row shows just its parameters.
UI convention: loc is always first.

From file: upload a file or enter a local path, then choose a Column. Use the panel below it to stack filters, clean the column and group rows (all optional). The Distribution list, N and Seed are hidden in this mode, since they only apply to simulated data. CDF bins lives in the **Visualize data** block and stays visible in both modes, because it sets the empirical CDF that every fit is scored against, file data included.

Reproducibility: for synthetic data, set Seed; for external data, “Fit” and “Fit All” share a cached empirical baseline so results match.

2) Fit all distributions  (on-screen block: FIT ALL DISTRIBUTIONS)

Continuous only: choose **loc free** (scipy fits loc) or **loc = 0** (loc held at zero) with the radio buttons next to Fit All. With loc = 0, distributions that need positive data cannot be fitted when the data has values ≤ 0; they are skipped and listed in the status line, and the table shows loc = 0 (fixed).

Click Fit All to:

Fit every supported distribution to the current data.

See a parameter table (with Parameter 1 = loc where applicable) and Fitting Error (max CDF deviation).

The table also has **AIC** and **BIC** columns. Both start from the log-likelihood of the data under the fitted distribution and add a penalty for each free parameter (AIC = 2k − 2 ln L, BIC = k ln n − 2 ln L), so a flexible distribution only scores better if it earns its extra parameters. BIC penalises parameters more strongly than AIC. Lower is better; the absolute values mean nothing, only differences within one table, and a gap under about 2 is too small to prefer one distribution over the other. A held loc = 0 is not counted as a parameter. A dash means some data falls outside that distribution's support, so its likelihood is zero. The table stays ranked by Fitting Error; when a different distribution has the lowest AIC, the status line names it.

View CDF/PDF overlays for the best fit.

Inspect a bar chart of errors (labels printed vertically).

3) Fit one distribution  (on-screen block: FIT ONE DISTRIBUTION)

Pick a distribution, enter parameters, and click Fit to:

Compute the max CDF error against the same empirical baseline.

Display CDF and PDF overlays for those parameters.

4) Visualize data  (on-screen block: VISUALIZE DATA, results in the Data view tab)

The **Visualize** buttons draw the data you are about to fit - after filters, cleaning and group selection - into the **Data view** tab. **Q-Q plot** uses the distribution and parameters currently in the Fit row, so it answers "do my parameters match this data?" before you fit. **Run chart** plots values in row order, which shows drift or steps that a histogram hides. **Sorted run chart** plots the same values in ascending order against their rank, with mean and median lines, so the range, gaps, ties and outliers stand out; like the run chart it has a pan/zoom toolbar. Tick **grid lines** to add a light grid to both run charts; it applies to a chart already on screen, keeping the current zoom. With a **Group by** column set, tick **split by group** to draw one series per group (up to 10).

5) Save results

Click Save results to write, into a timestamped folder under outputs/:

fitted_parameters.csv, samples.csv

fit_errors.png, cdf_overlay.png, pdf_overlay.png

Links appear inline in the notebook. Group mode produces no CDF/PDF overlay, so only the files that exist are written.

Parameter conventions

UI and synthetic generation always show loc first.

Fitted parameter table makes Parameter 1 the loc value (when defined).

Special notes:

Exponential

Simulated input uses scale (Distribution → Exponential has “scale”).

Fit expects λ (rate); it’s clearly labeled as “λ (rate)”.

Internally we convert rate ↔ scale as needed.

Beta expects (loc, α, β, scale); support for general location/scale form.

Weibull UI uses (loc, k(shape), λ(scale)).

Lognormal UI uses (loc, s(shape), scale).

GEV UI uses (loc, ξ(shape), scale).

All other distributions follow the labels shown in the UI and table.

Supported file types (and add-ons)
Type	Ext	Notes / Add-on
CSV/TSV/PSV/TXT	.csv, .tsv, .psv, .txt	TXT auto-sniffs delimiter; falls back to whitespace
Excel	.xlsx, .xls	openpyxl for .xlsx
JSON	.json	Standard JSON
JSON Lines	.jsonl, .ndjson	lines=True
Parquet	.parquet	pyarrow recommended
Feather / Arrow IPC	.feather, .arrow	pyarrow required
SPSS / Stata / SAS	.sav, .dta, .sas7bdat, .xpt	pyreadstat
HDF5	.h5, .hdf5	pandas.read_hdf
MATLAB	.mat	v5 via scipy.io; v7.3 via h5py

Column selection: the column dropdown prioritizes numeric columns; non-numeric are kept for filtering but are not usable for fitting until coerced to numeric.

Troubleshooting

“Load error: …requires pyarrow/pyreadstat/openpyxl/h5py”
Install the indicated package (see Quick start).

No numeric data in selected column
Choose a different column or clean/filter data first. The app coerces to numeric and drops non-numeric rows for modeling.

Fit vs Fit All mismatch

Ensure you didn’t change bins, filter, or seed between actions.

The app caches the empirical baseline so both use the same CDF/PDF reference.

Widget UI not rendering
Ensure ipywidgets is installed and enabled; reload the notebook kernel.

How it works (internals snapshot)

pdf_cdf(x, bins, PDF=True) builds a histogram-based empirical CDF (stable in the tails) and derives an empirical PDF by differencing.

datasource(...) generates synthetic arrays using SciPy/NumPy or returns a numeric column from the uploaded DataFrame.

theory_cdf/pdf(process, params, x) call SciPy CDF/PDF for each distribution (parameter order mapped from the UI).

fit_subset_and_summarize(r, bins, subset):

Fits each distribution with scipy.stats.<dist>.fit.

Computes max absolute deviation between empirical CDF and theoretical CDF.

Builds a table (with loc in Parameter 1 where applicable).

A small cache ensures the empirical baseline used in Fit All is reused by Fit.

Notes & tips

CDF bins (in the **Visualize data** block): affects empirical CDF/PDF smoothness and the histogram detail; 50–200 is a sensible range. It also sets the empirical curve that Fit and Fit All are scored against, for file data as well as simulated data.

Scale/shape positivity: the UI enforces positive values for these parameters.

Large catalogs: the error bar chart turns labels vertical to stay readable.

## Where files are written

`Save results` and the formula sheets write under your **current working directory**:

- `outputs/fit_outputs_<timestamp>/` — continuous fits
- `outputs/discrete_fit_outputs_<timestamp>/` — discrete fits
- `artifacts/` — the rendered formula sheets

Run the notebook from the folder you want those in.
