
# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

from anemoi.transform.filter import Filter
from anemoi.transform.filters.tabular import filter_registry


@filter_registry.register("whitelist")
class WhiteList(Filter):
    """Observation spatial Whitelist filter (keeping only reliable sources)

    The configuration should be a dictionary with the following keys:

    - ``format``: str, the format of whitelist ('from_lists'/'from_file', default from_list)
    - ``drop-whitelist-id``: bool = False, whether to drop the column identifying whitelist
    - in case ``format`` is `from_lists` : lists of keys/values to keep
    - in case ``format` is `from_file`, ``path`` :a yaml file containing period-separated whitelists

    Examples
    --------
    .. code-block:: yaml

      input:
        pipe:
          - source:
              ...
          - whitelist:
              format: from_file
              path: path/to/yaml
              drop-whitelist-id: false

    .. code-block:: yaml

      input:
        pipe:
          - source:
              ...
          - whitelist:
              format: from_list
              lists: 
                aircraftId:
                - M****
                - M////
                - M----
              drop-whitelist-id: true
    """

    def __init__(
        self,
        *args,
        **kwargs
    ):

        self.whitelist = kwargs

        self.filter_format = self.whitelist.get('format', 'from_lists')
        self.drop = self.whitelist.get('drop-whitelist-id',False)

    def get_whitelist_filter(self, start: np.datetime64, end: np.datetime64) -> tuple[list, list]:
        """Create the keys and values to filter dataframe from config 
        Special case is whitelist is taken from file : refine whitelist with date-dependent filter
        """

        match self.filter_format:
            case 'from_lists':
                whitedict = self.whitelist.get('lists', None)

                assert whitedict is not None, ("Whitelist format is from_lists"
                                        "(default), got no information")
            case 'from_file':
                filename = self.whitelist.get('path')
                with open(filename,'r') as f:
                    whitelist_data = yaml.safe_load(f)

                start_dt, end_dt = (start.to_pydatetime()), (end.to_pydatetime())
                periods = whitelist_data.keys()
                found_period = False
                for period in periods:
                    period_start, period_end = period.split('/')
                    period_start, period_end = iso8601_to_datetime(period_start), iso8601_to_datetime(period_end)
                    if start_dt >= period_start and end_dt <= period_end:
                        whitedict = whitelist_data.get(period,None)

                        assert whitedict is not None, ("Whitelist format is from_file,"
                                        f"but got no dict for {period}")
                        found_period = True
                        break
                if not found_period:
                    raise ValueError(f"No period found for whitelist {filename}, start {start}, end {end}")
            
            case _:
                raise ValueError("Unknown whitelist format")
            
        return whitedict

    def forward(self, df: pd.DataFrame) -> pd.DataFrame:

        start = df["date"].min()
        end = df["date"].max()
        whitedict = self.get_whitelist_filter(start, end)
        wl_keys = list(whitedict.keys())
        
        for key, val in whitedict.items():
            df = df.loc[df[key].isin(val)]
        
        if self.drop:
            df = df.drop(columns=wl_keys)

        return df

def iso8601_to_datetime(iso8601_str: str) -> str:
    """Convert ISO8601 datetime string to YYYYMMDDHHMMSS string.

    Parameters
    ----------
    iso8601_str : str
        ISO8601 datetime string.

    Returns
    -------
    str
        Datetime string in YYYYMMDDHHMMSS format.
    """
    dt = datetime.fromisoformat(iso8601_str)
    return dt