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


@filter_registry.register("height_to_pressure_icao_tabular")
class HeightToPressureICAO(Filter):
    """Converts height (in meters) to pressure using the ICAO convention.

    The ``height`` config key defines the name of the column containing
    meter height (must exist in the DataFrame). The ``pressure`` config key
    defines the name of the column containing height following conversion
    (i.e. the result) - if not passed in (or None), this will overwrite the
    pressure column.

    Examples
    --------
    .. code-block:: yaml

      input:
        pipe:
          - source:
              ...
          - pressure_to_height:
              pressure: p
              height: z

    """

    def __init__(self, *, height, pressure=None, drop_height=False):
        self.height = height
        self.pressure = pressure if pressure else height
        self.drop_height = drop_height

    def forward(self, obs_df: pd.DataFrame) -> pd.DataFrame:
        raise_if_df_missing_cols(obs_df, [self.height])
        logging.info("Converting height to pressure")
        obs_df[self.pressure] = obs_df[self.height].apply(
            lambda x: round(1013.25 * (1 - 0.0065 * x / 288.15) ** 5.255)
        )
        if self.drop_height:
            obs_df = obs_df.drop(columns=[self.height])
        return obs_df

