# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime as dt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

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
        rounding: str | None = None,
        subsample: str | None = None,
    ):
        self.timeslot_length = timeslot_length
        self.rounding = rounding
        self.subsample = subsample

    def forward(self, df: pd.DataFrame) -> pd.DataFrame:
      
      time_start = df["date"].min().to_pydatetime()
      time_end = df["date"].max()

      # rounding to next rounding slot (no backward in time)
      # shifting start to break tie
      time_start = time_start + dt.timedelta(microseconds=1)
      if self.subsample:
          
          delta = dt.timedelta(seconds=self.subsample)
          subsample_grid = pd.date_range(time_start, time_end, freq=delta)
          tree = cKDTree((subsample_grid.values.astype('int64')// 10**9).reshape(-1,1))
          distances, _ = tree.query((df['date'].values.astype('int64') // 10**9).reshape(-1,1))
          df = df.assign(temporal_distance=distances)
          df = df.loc[df["temporal_distance"]<=1]
          df = df.drop(columns=['temporal_distance'])
      # Create time grid
      time_start = df["date"].min().to_pydatetime()
      time_end = df["date"].max()
      
      # rounding to next rounding slot (no backward in time)
      # shifting start to break tie
      time_start = time_start + dt.timedelta(microseconds=1)
      time_grid = pd.date_range(time_start, time_end, freq=f"{self.timeslot_length}s")
      if self.rounding is not None:
          time_grid = time_grid.round(self.rounding)

      tree = cKDTree((time_grid.values.astype('int64')// 10**9).reshape(-1,1))
      distances, indices = tree.query((df['date'].values.astype('int64') // 10**9).reshape(-1,1))
      df = df.assign(date=time_grid[indices])
      
      return df