import awkward as ak
import numpy as np
import pandas as pd

from awkward_pandas.array import AwkwardExtensionArray


def merge(dataframe, name: str | None = None) -> pd.Series:
    """Create a single awkward series by merging the columns of a dataframe.

    Parameters
    ----------
    dataframe: pd.DataFrame
        Containing columns of simple numpy type, object type (e.g.,
        srtings, lists or dicts) or existing awkward columns.
    name: str or None
        Name of the output series.
    """
    out = {}
    for c in dataframe.columns:
        if dataframe[c].dtype == "awkward":
            out[c] = dataframe[c].values._data
        elif dataframe[c].dtype == "string[pyarrow]":
            out[c] = ak.from_arrow(dataframe[c].values._data)
        elif dataframe[c].dtype == np.dtype("O"):
            out[c] = ak.from_iter(dataframe[c])
        else:
            out[c] = dataframe[c].values
    return pd.Series(AwkwardExtensionArray(ak.Array(out)), name=name)
