# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import pandas as pd

from anemoi.transform.filter import Filter
from anemoi.transform.filters.tabular import filter_registry

@filter_registry.register("time-grid")
class TimeGrid(Filter):
    """SuperOb filter for aggregating observation dates onto a time grid.

    The configuration should be a dictionary with the following keys:

    - ``timeslot_length``: int, the length of the timeslot in seconds
    - ``round``: str | None = None, a method to round dates to the closest time unit

    Examples
    --------
    .. code-block:: yaml

      input:
        pipe:
          - source:
              ...
          - time-grid:
              timeslot_length: 3600
              round: hour
    """

    def __init__(
        self,
        *,
        timeslot_length: int,
        rounding: str | None = None
    ):
        self.timeslot_length = timeslot_length
        self.rounding = rounding

    def forward(self, df: pd.DataFrame) -> pd.DataFrame:

      df = df.copy()

      time_start = df["date"].min()
      time_end = df["date"].max()

      # Create time grid
      time_grid = pd.date_range(time_start, time_end, freq=f"{self.timeslot_length}s")
      if self.rounding is not None:
          time_grid = time_grid.round(self.rounding)
      # Find nearest time slot for each data point
      # Use side='right' to handle exact matches correctly
      temporal_indices = np.searchsorted(time_grid, df["date"], side="right") - 1
    
      time_start = df["date"].min()
      time_end = df["date"].max()

      time_grid = time_grid.to_pydatetime()
      time_references = np.where((temporal_indices)==0, time_grid[0], time_grid[0])

      for ti in temporal_indices[1:]:
          time_references = np.where((temporal_indices)==ti, time_grid[ti], time_references)

      df = df.assign(date=time_references)

      return df