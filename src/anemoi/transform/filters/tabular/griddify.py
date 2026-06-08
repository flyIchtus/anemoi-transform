# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import sys

import earthkit.data as ekd
import numpy as np
import pandas as pd
from anemoi.transform.fields import new_field_from_latitudes_longitudes
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_field_with_valid_datetime
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter
from anemoi.transform.filters import filter_registry

LOG = logging.getLogger(__name__)


@filter_registry.register("griddify-from-superob", aliases=["to_gridded_superob"])
class GriddifyFromSuperOb(Filter):

    def __init__(
        self,
        grib_template: str,
        grid_template: str
    ):
        
        self.template = ekd.from_source("file", grib_template)[0]
        coords = np.load(grid_template)
        self.latitudes, self.longitudes = coords['latitude'], coords['longitude']
        self.length = len(self.latitudes)

    def forward(self, frame):

        assert "spatial_index" in frame.columns
        result = []
        value_columns = [col for col in frame.columns if col not in ("date", "latitude", "longitude", "spatial_index")]

        for col in value_columns:
            for date, group in frame.groupby("date"):
                # Only keep relevant columns for this variable and date
                df = group[["date", "latitude", "longitude", "spatial_index", col]]
                gridded_values = np.full(self.length, np.nan, dtype="float32")
                spatial_indices = df['spatial_index'].astype(int)
                gridded_values[spatial_indices] = df[col]

                result.append(
                    new_field_with_valid_datetime(
                        new_field_from_latitudes_longitudes(
                        new_field_from_numpy(gridded_values, template=self.template, valid_datetime=date, param=col),
                        latitudes=self.latitudes,
                        longitudes=self.longitudes
                        ),
                        date=date,
                    )
                )

        return new_fieldlist_from_list(result)
