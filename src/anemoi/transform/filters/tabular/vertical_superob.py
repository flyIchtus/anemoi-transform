# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pandas as pd

from anemoi.transform.filter import Filter
from anemoi.transform.filters.tabular import filter_registry
from anemoi.transform.filters.tabular.support.superob import assign_vertical_grid
from anemoi.transform.filters.tabular.support.superob import define_vertical_grid_from_template

@filter_registry.register("vertical-superob")
class VerticalSuperOb(Filter):
    """VerticalSuperOb filter for aggregating observations into vertical height cells

    The configuration should be a dictionary with the following keys:

    - ``vertical_grid``: str, the vertical list of levels to use for aggregation
    - ``height_column``: list[str], the columns to take the nearest value for
    - ``height_tol``: float, the +/- tolerance (in units of height column) to aggregate value on
    - ``columns_to_groupby``: list[str], the columns to group by for aggregation
    - ``agg_method``: str, the pandas-compatible aggregation method (default mean)

    Examples
    --------
    .. code-block:: yaml

      input:
        pipe:
          - source:
              ...
          - vertical-superob:
              vertical_grid: /path/to/vertical/list/of/levels.npz
              height_column: pressure
              columns_to_groupby: ["reporttype"]
              agg_method: median
             

    """

    def __init__(
        self,
        *,
        vertical_grid: str,
        height_column: list[str],
        height_tolerance: float,
        agg_method: str = 'mean'
    ):
        self.vertical_grid = vertical_grid
        self.agg_method = agg_method
        self.height_tolerance = height_tolerance
        self.height_column = height_column

    def forward(self, df: pd.DataFrame) -> pd.DataFrame:
        assert self.height_column in df.columns, f"No column named {self.height_column} in tabular data"
        
        if self.vertical_grid.endswith("npz"):
            vertical_grid = define_vertical_grid_from_template(self.vertical_grid, self.height_column)
        else:
            raise ValueError("Only npz files supported for vertical grid definition")

        if len(df) == 0:
            return df

        # Assign each observation to an output grid cell
        df = assign_vertical_grid(df, vertical_grid, self.height_column, self.height_tolerance, rejection=True)
        return df
