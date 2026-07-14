import numpy as np
import pandas as pd


def coerce_to_numeric(series: pd.Series) -> pd.Series:
    """Convert a column to numeric values, using category codes when needed."""
    numeric_series = pd.to_numeric(series, errors="coerce")
    if numeric_series.notna().any():
        return numeric_series.astype(float)

    if series.empty:
        return pd.Series(dtype=float)

    non_null = series.dropna().astype(str)
    if non_null.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)

    mapping = {value: idx for idx, value in enumerate(non_null.unique())}
    encoded = series.astype(str).map(mapping)
    return encoded.astype(float)


def build_numeric_view(df: pd.DataFrame) -> pd.DataFrame:
    """Create a numeric-friendly view of a dataframe for plotting and inference."""
    return df.apply(coerce_to_numeric, axis=0)


def get_plotable_columns(df: pd.DataFrame) -> list[str]:
    """Return columns that can be displayed after safe coercion."""
    return [col for col in df.columns if df[col].notna().any()]
