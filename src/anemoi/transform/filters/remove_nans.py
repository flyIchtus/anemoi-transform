# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import tqdm

from ..fields import new_field_from_latitudes_longitudes
from ..fields import new_field_from_numpy
from ..fields import new_fieldlist_from_list
from ..filter import Filter
from . import filter_registry

LOG = logging.getLogger(__name__)


@filter_registry.register("remove_nans")
class RemoveNaNs(Filter):
    """A filter to mask out NaNs."""

    def __init__(self, *, method="mask", check=False):
        self.method = method
        self.check = check

        assert method == "mask", f"Method {method} not implemented"
        assert not check, "Check not implemented"

        self._mask = None
        self._latitudes = None
        self._longitudes = None

    def forward(self, fields):
        import numpy as np

        if self._mask is None:
            first = fields[0]
            data = first.to_numpy(flatten=True)
            self._mask = ~np.isnan(data)

            latitudes, longitudes = first.grid_points()
            self._latitudes = latitudes[self._mask]
            self._longitudes = longitudes[self._mask]

        result = []
        for field in tqdm.tqdm(fields, desc="Remove NaNs"):

            data = field.to_numpy(flatten=True)
            result.append(
                new_field_from_latitudes_longitudes(
                    new_field_from_numpy(data[self._mask], template=field),
                    latitudes=self._latitudes,
                    longitudes=self._longitudes,
                )
            )

        return new_fieldlist_from_list(result)

    def backward(self, data):
        raise NotImplementedError("IconRefinement is not reversible")
