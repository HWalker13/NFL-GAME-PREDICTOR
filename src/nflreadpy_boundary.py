"""polars -> pandas conversion at the ingest boundary (SPEC Section 13, Phase 9).

``nflreadpy`` returns polars DataFrames. Everything downstream of
``src/data_ingest.py`` is pandas, so the conversion happens here, once, and
nothing past ``data/raw/`` ever sees polars.

Why not ``polars.DataFrame.to_pandas()``: it requires ``pyarrow``, which is not
installed in this venv. Installing it would silently switch pandas'
``read_parquet(engine='auto')`` from fastparquet to pyarrow for EVERY parquet
read in the project (raw cache, game_features, predictions), which is a
pipeline change in its own right. This converter goes column by column through
numpy instead, so no new parquet engine is introduced.

Mapping (chosen to match what ``nfl_data_py`` + pandas produced before):

* float columns            -> float64 (nulls -> NaN)
* integer columns, no null -> numpy int of the same width
* integer columns, nulls   -> float64 (nulls -> NaN), pandas' own convention
* boolean, no null         -> bool; with nulls -> object (True/False/None)
* string / categorical     -> object (nulls -> None)
* date / datetime          -> datetime64[ns]
* anything else            -> object via ``to_list()``

``downcast_floats`` then reproduces ``nfl_data_py.import_pbp_data(downcast=True)``
(float64 -> float32), which is the contract the cached pbp files were written
under.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

_INT_NUMPY = {
    pl.Int8: np.int8, pl.Int16: np.int16, pl.Int32: np.int32, pl.Int64: np.int64,
    pl.UInt8: np.uint8, pl.UInt16: np.uint16, pl.UInt32: np.uint32, pl.UInt64: np.uint64,
}


def _series_to_numpy(s: pl.Series) -> np.ndarray:
    dt = s.dtype
    has_null = s.null_count() > 0
    if dt in (pl.Float32, pl.Float64):
        return s.cast(pl.Float64).to_numpy()
    if dt in _INT_NUMPY:
        if has_null:
            return s.cast(pl.Float64).to_numpy()
        return s.to_numpy().astype(_INT_NUMPY[dt], copy=False)
    if dt == pl.Boolean:
        if has_null:
            return np.array(s.to_list(), dtype=object)
        return s.to_numpy().astype(bool, copy=False)
    if dt in (pl.Date, pl.Datetime):
        return pd.to_datetime(pd.Series(s.to_list(), dtype=object)).to_numpy()
    # Utf8/String, Categorical, Enum, Null, nested types
    return np.array(s.to_list(), dtype=object)


def polars_to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    """Convert a polars DataFrame to pandas without pyarrow (see module doc)."""
    return pd.DataFrame({name: _series_to_numpy(df.get_column(name)) for name in df.columns},
                        columns=df.columns)


def downcast_floats(df: pd.DataFrame) -> pd.DataFrame:
    """float64 -> float32, exactly as ``nfl_data_py.import_pbp_data(downcast=True)``."""
    cols = df.select_dtypes(include=[np.float64]).columns
    df[cols] = df[cols].astype(np.float32)
    return df


# --------------------------------------------------------------------------- #
# Raw-file contract: reproduce the dtypes nfl_data_py 0.3.3 wrote to data/raw/
# --------------------------------------------------------------------------- #
# Found by the Phase 9 equivalence test (docs/PHASE9_DATA_LAYER.md): values
# are identical across libraries, but nflreadpy carries some integer columns
# as int32 where the cached files hold int64 -- and the unchanged
# features.prior_season_mean() merge_asof refuses mixed int32/int64 keys. The
# two functions below restore the old contract; they never change a value.

def pbp_to_raw_contract(df: pl.DataFrame) -> pd.DataFrame:
    """nflreadpy pbp -> the frame ``import_pbp_data(downcast=True)`` returned.

    * floats downcast to float32 (``downcast=True``);
    * ``season`` as int64 (nfl_data_py overwrote it with the Python int year);
    * other integer columns keep their parquet width, as nfl_data_py's
      ``pandas.read_parquet`` kept them.

    Not reproduced: nfl_data_py's left-merge of ``pbp_participation`` (2016+).
    Those ~20-26 columns are read by nothing in this project; if they are ever
    needed, load them explicitly with ``nflreadpy.load_participation``.
    """
    out = downcast_floats(polars_to_pandas(df))
    out["season"] = out["season"].astype(np.int64)
    return out


def schedules_to_raw_contract(df: pl.DataFrame) -> pd.DataFrame:
    """nflreadpy schedules -> the dtypes ``import_schedules`` (``read_csv``) gave.

    Every integer column becomes int64 (``read_csv``'s integer default).
    ``old_game_id``/``espn`` stay strings (nflverse stores them as strings;
    ``read_csv`` had parsed the digits as int64). Nothing in the project reads
    either column.
    """
    out = polars_to_pandas(df)
    for c in out.columns:
        if pd.api.types.is_integer_dtype(out[c]):
            out[c] = out[c].astype(np.int64)
    return out
