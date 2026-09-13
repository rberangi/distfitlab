"""Reading an uploaded data file into a DataFrame.

Shared by both apps so they accept the same formats. Pure: bytes and a filename in,
a DataFrame out - no widgets and no app state, so either app can call it.
"""
import io, os, tempfile
import pandas as pd


def read_uploaded_dataframe(content: bytes, filename: str) -> pd.DataFrame:
    """Best-effort reader for many formats. Requires optional deps for some types."""
    ext = os.path.splitext((filename or "").lower())[1]
    # Delimited text
    if ext in {".csv", ".tsv", ".psv"}:
        default_sep = {".csv": ",", ".tsv": "\t", ".psv": "|"}[ext]
        try:
            return pd.read_csv(io.BytesIO(content), sep=default_sep)
        except Exception:
            return pd.read_csv(io.BytesIO(content), sep=None, engine="python")
    # Excel
    if ext in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(io.BytesIO(content))
        except ImportError as e:
            raise RuntimeError("Reading Excel requires 'openpyxl' (for .xlsx). Install: pip install openpyxl") from e
    # JSON / NDJSON
    if ext == ".json":
        return pd.read_json(io.BytesIO(content), lines=False)
    if ext in {".jsonl", ".ndjson"}:
        return pd.read_json(io.BytesIO(content), lines=True)
    # Parquet
    if ext == ".parquet":
        try:
            return pd.read_parquet(io.BytesIO(content))
        except ImportError as e:
            raise RuntimeError("Reading Parquet requires 'pyarrow' or 'fastparquet'. Try: pip install pyarrow") from e
    # Feather / Arrow IPC
    if ext in {".feather", ".arrow"}:
        try:
            import pyarrow.feather as feather
            tbl = feather.read_table(io.BytesIO(content))
            return tbl.to_pandas()
        except ImportError as e:
            raise RuntimeError("Reading Feather/Arrow requires 'pyarrow'. Try: pip install pyarrow") from e
    # SPSS / Stata / SAS
    if ext in {".sav", ".dta", ".sas7bdat", ".xpt"}:
        try:
            import pyreadstat
        except ImportError as e:
            raise RuntimeError("Reading SPSS/Stata/SAS requires 'pyreadstat'. Try: pip install pyreadstat") from e
        with tempfile.NamedTemporaryFile(suffix=ext) as tmp:
            tmp.write(content); tmp.flush()
            if ext == ".sav":
                df, _ = pyreadstat.read_sav(tmp.name)
            elif ext == ".dta":
                df, _ = pyreadstat.read_dta(tmp.name)
            elif ext == ".sas7bdat":
                df, _ = pyreadstat.read_sas7bdat(tmp.name)
            else:  # .xpt
                df, _ = pyreadstat.read_xport(tmp.name)
        return df
    # HDF5
    if ext in {".h5", ".hdf5"}:
        try:
            with tempfile.NamedTemporaryFile(suffix=ext) as tmp:
                tmp.write(content); tmp.flush()
                return pd.read_hdf(tmp.name)
        except ImportError as e:
            raise RuntimeError("Reading HDF5 may require 'tables' and 'h5py'. Try: pip install tables h5py") from e
    # Fallback: CSV sniffing
    return pd.read_csv(io.BytesIO(content), sep=None, engine="python")
