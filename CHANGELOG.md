# Changelog

## 0.5.0 — 2026-09-22

- **P(condition) is now its own row** in the Probability tab, so Bayes' rule can be read straight
  off the table: P(condition | E) x P(E) = P(E | condition) x P(condition). Its value is the share
  of all rows the condition keeps; previously the two counts were only available inside the
  "Based on" text. Like P(condition | E) it is left out while **trim percentiles** is on, because
  the kept rows are then not a subset of all rows.
- **"Fit" is renamed "Check my parameters"** in both apps. The button never fitted anything - it
  scores the distribution and parameters *you* typed against the data - so the old name suggested
  it would fill the boxes for you. The block above it is now
  *Check one distribution against your parameters*, and it carries a tooltip saying what it does.
  **Fit All**, which really does fit, is unchanged.
- **The model row is colour-coded to that button**: *P(E) under the model* is tinted the same cyan,
  because it is the one row computed from the parameter boxes rather than from your rows. The
  colour and the button label are single constants (`FIT_COLOR`, `FIT_LABEL`) in
  `distfitlab.probability`, shared by both apps so they cannot drift apart.

## 0.4.0 — 2026-09-19

- **Conditional probability** in both apps: set an event on the Use column (`<=`, `<`, `>`, `>=`,
  `between`, `==`) and click **P(event | filters)**. The new **Probability** tab shows
  P(event | the rows the filters and group keep) with a 95% Wilson interval, P(event) over all
  rows, their ratio, P(condition | event) by Bayes' rule, and P(event) under the distribution
  in the Fit row. A live formula box under the row spells out the probability the current
  settings define, e.g. `P( count > 500 | season == winter and year == 1 )`. The calculations
  live in a new shared module, `distfitlab.probability`. The row's label names the
  Use column (*Event on count:*).
- **PDF** button (empirical PDF) in the Continuous app's Data view: the histogram-based CDF on **CDF bins** bins,
  differenced - the same curve Fit All's PDF overlay is compared against. Split by group works.
- **PMF**: the Discrete app's "Counts bar chart" button is renamed PMF - it draws the empirical PMF.
  Button tooltips say "empirical" and how each chart is computed; chart titles are unchanged.
- **Zoom and grid** on ECDF, PDF and PMF, like the run charts: pan/zoom toolbar, and **grid lines** toggles the chart on screen.
- **Run charts split by group**: with *split by <column>* ticked, the Run chart shows one panel per
  group (shared axes, own mean line) and the Sorted run chart overlays the groups against
  percentile rank, so groups of different sizes compare directly. Both stay zoomable.
- **Group by is easier to follow**: the file panel has its own *Filter · Clean · Group* caption, and
  the Visualize row's split checkbox names the column (*split by month*) and is disabled while
  Group by is *(none)* or the data is simulated.
- **Group order**: groups are listed in natural order everywhere - the Group dropdown, the
  Data view charts and summary table, and grouped Fit All - so numeric groups such as month
  run 1, 2, ..., 12 instead of 1, 10, 11, 12, 2, and text such as press-2 comes before press-10.

## 0.3.0 — 2026-09-14

- **Fit All**: **AIC** and **BIC** columns in both apps, from the log-likelihood of each fit with
  free parameters counted (a held loc = 0 is not). The status line names the lowest-AIC
  distribution when it differs from the best by fitting error. Fits whose data falls outside
  the distribution's support show a dash. Saved CSVs include both columns.
- **Data view**: the Visualize buttons wrap onto a second line in narrow notebooks instead of
  being cut off on the right.

## 0.2.0 — 2026-09-13

- First release on PyPI: `pip install distfitlab`.
- **Data view**: new **Sorted run chart** — values in ascending order against rank, with
  mean and median lines and the same pan/zoom toolbar as the run chart.
- **Data view**: **Grid lines** option for both run charts, applied live to the chart on screen.
- **Data view**: run charts are narrower (7 in, was 9 in), since they can be zoomed.
- README: "After installing" walkthrough and troubleshooting; links now work on the PyPI page.

## 0.1.0 — 2026-09-13 (GitHub only)

- Two fitting apps behind one switch: 18 continuous distributions and 5 count distributions.
- **Fit All** ranks every distribution by max |empirical CDF − theoretical CDF|, best first.
- **Fit** scores a single distribution against parameters you type, with CDF/PDF overlays.
- **Data view**: histogram / counts bar chart, ECDF, box plot, Q-Q plot, run chart and
  summary statistics, optionally split by group. The run chart has a pan/zoom toolbar.
- File data: multi-format reader (CSV, Excel, JSON, Parquet, Feather, SPSS/Stata/SAS, HDF5),
  a stack of filter conditions, cleaning options, and group-by that fits every group.
- **Save results** writes the table, the samples and the figures to a timestamped folder.
