# Changelog

## 0.1.0 — first public release

- Two fitting apps behind one switch: 18 continuous distributions and 5 count distributions.
- **Fit All** ranks every distribution by max |empirical CDF − theoretical CDF|, best first.
- **Fit** scores a single distribution against parameters you type, with CDF/PDF overlays.
- **Data view**: histogram / counts bar chart, ECDF, box plot, Q-Q plot, run chart and
  summary statistics, optionally split by group. The run chart has a pan/zoom toolbar.
- File data: multi-format reader (CSV, Excel, JSON, Parquet, Feather, SPSS/Stata/SAS, HDF5),
  a stack of filter conditions, cleaning options, and group-by that fits every group.
- **Save results** writes the table, the samples and the figures to a timestamped folder.
