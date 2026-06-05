# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import pandas as pd

from anemoi.transform.filter import Filter
from anemoi.transform.filters.tabular import filter_registry
from anemoi.transform.filters.tabular.support.utils import raise_if_df_missing_cols


def pivot_obs_df(df: pd.DataFrame, values: list, columns: list) -> pd.DataFrame:
    """Reshape the DataFrame, organized by the values in particular columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame to pivot.
    values : list
        List of column names. Values in these columns will be spread across
        different values of the columns, such as observed value or quality
        control values.
    columns : list
        List of column names. Values in these columns will be used to define
        the new columns after reshaping, such as channel number or varno.

    Returns
    -------
    pandas.DataFrame
        Pivoted DataFrame.

    Notes
    -----
    The function reshapes the DataFrame based on the specified `columns` and
    `values`. All columns not specified in `columns` and `values` are assumed
    to be "index" values (i.e., they remain the same within a given observation
    group).

    For example, given the following DataFrame:

    +---+---+
    | A | B |
    +---+---+
    | 1 | 0.5 |
    | 2 | 0.1 |
    | 3 | 0.6 |
    | 1 | 0.3 |
    | 3 | 0.7 |
    +---+---+

    Using `columns=["A"]` and `values=["B"]`, the reshaped DataFrame would be:

    +-----+-----+-----+
    | B_1 | B_2 | B_3 |
    +-----+-----+-----+
    | 0.5 | 0.1 | 0.6 |
    | 0.3 | NaN | 0.7 |
    +-----+-----+-----+

    The column names in the resulting DataFrame are flattened to include the
    original column name and the unique values from the `columns` parameter.

    The new columns are appended to the "index" columns.
    """
    # Calculate the index variables, based on all variables not in columns or values.
    indices = list(filter(lambda a: a not in values + columns, df.columns))
    # Perform the pivot
    pivoted = df.pivot(index=indices, columns=columns, values=values)
    # Flatten MultiIndex column names
    pivoted.columns = ["_".join(str(elem)[:6] for elem in col) for col in pivoted.columns]
    # Reset the dataframe index
    pivoted = pivoted.reset_index()
    return pivoted

@filter_registry.register("pivot_table")
class PivotTable(Filter):
    """Pivot the dataframe

    The configuration should be a dictionary with the following keys:

    - ``pivot_columns``: list[str], the columns to be pivoted
    - ``pivot_values``: list[str], the columns to preserve and add a pivot


    All columns not specified in "pivot_columns" and "pivot_values"
    will be assumed to be "index" values (i.e. are the same within a
    given observation group).

    Pivot values are named according to the unique values in the
    pivot columns. For instance, if
    `pivot_columns=["channel_number@body"]`
    with two unique channel numbers 1 and 2 that identify rows, and
    `pivot_values=["initial_obsvalue@body"]`, then the resulting columns
    will be named "observed_value_1" and "observed_value_2".


    Examples
    --------
    .. code-block:: yaml

      input:
        pipe:
          - source:
              ...
          - pivot-table:
              pivot_columns: ["channel_number@body"]
              pivot_values: ["initial_obsvalue@body"]

    """

    def __init__(self, *, pivot_columns=list[str], pivot_values=list[str]):
        self.pivot_columns = pivot_columns
        self.pivot_values = pivot_values

        assert len(pivot_columns)>0, "Empty list of columns for pivot"
        assert len(pivot_values)>0, "Empty list of values for pivot"

    def forward(self, obs_df: pd.DataFrame) -> pd.DataFrame:
        raise_if_df_missing_cols(obs_df, self.pivot_columns)
        raise_if_df_missing_cols(obs_df, self.pivot_values)

        logging.info(f"pivoting {self.pivot_columns} onto {self.pivot_values}")
        obs_df = pivot_obs_df(obs_df, self.pivot_values, self.pivot_columns)
        return obs_df

