# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import numpy.ma as ma
import pandas as pd

from anemoi.transform.filter import Filter
from anemoi.transform.filters.tabular import filter_registry


def compute_haversine_distances(trajectories: np.ndarray, reference_points: np.ndarray) -> np.ndarray:
    """Compute the distance"""

    dlat = (np.pi / 180.0) * (trajectories[:, :, 0] - reference_points[:, :, 0])
    dlon = (np.pi / 180.0) * (trajectories[:, :, 1] - reference_points[:, :, 1])
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos((np.pi / 180.0) * reference_points[:, :, 0]) * np.cos((np.pi / 180.0) * trajectories[:, :, 0]) * np.sin(dlon / 2) ** 2
    )
    #print(a)
    #print(1-a)
    distances = 2 * np.atan2(np.sqrt(a), np.sqrt(1 - a))
    #print("unique distance", np.unique(distances[0] * 6371.0, axis=0))

    distances = np.expand_dims(distances,axis=-1) * 6371.0
    #    distances = distances[:,:,np.newaxis] * 6371.0

    #print("distances", distances[0], distances.shape)
    #print("unique distance", np.unique(distances[0], axis=0))
    # distances in km with earth radius
    return distances

def filter_trajectories(trajectories: np.ndarray, reference_points: np.ndarray, minimum_distance: float) -> tuple[np.ndarray, np.ndarray]:
    distances = np.repeat(compute_haversine_distances(trajectories.data, reference_points),2,2)

    mask = ma.getmask(ma.masked_less(distances, minimum_distance))
    #print("distance mask", mask, mask.shape)
    #print(trajectories.mask)
    trajectories.mask = mask | trajectories.mask
    #print(trajectories.mask)
    return trajectories


def parse_distance(distance: str | int) -> float:
    if isinstance(distance, int):
        return float(distance)
    elif isinstance(distance, str):
        if distance.endswith('km'):
            try:
                return float(distance[:-2])
            except TypeError as err:
                raise ValueError(f"Malformed distance input {distance}") from err
        elif distance.endswith('m'):
            try:
                return float(distance[:-1]) * 1000.0
            except TypeError as err:
                raise ValueError(f"Malformed distance input {distance}") from err
        
        raise ValueError(f"Malformed distance input {distance}, must be either '25', or '25m', or '25km")


@filter_registry.register("spatial-thinning")
class SpatialThinning(Filter):
    """Observation spatial thinning filter (limiting obs error correlation).

    The configuration should be a dictionary with the following keys:

    - ``min_dist``: str, the minimum distance of two consecutive observations
    - ``starting_point``: str = first, the method to choose a trajectory start 
        in a given time window (choice between 'first', 'middle', 'last')

    Examples
    --------
    .. code-block:: yaml

      input:
        pipe:
          - source:
              ...
          - spatial-thinning:
              trajectory_column_name: aircraft
              min_dist: 25km
              starting_point: middle
    """

    def __init__(
        self,
        *,
        min_dist: int,
        thinning_id: str,
        drop_thinning_id: bool,
        starting_point: str ='first'

    ):
        self.min_dist = parse_distance(min_dist)
        print("MIN DIST", self.min_dist)
        self.thinning_id = thinning_id
        self.drop_thinning_id = drop_thinning_id
        self.starting_point = starting_point

    def forward(self, df: pd.DataFrame) -> pd.DataFrame:
      
        extract_df = df[["latitude", "longitude"] + [self.thinning_id]]
        
        pivoted_traj = extract_df.pivot(columns=[self.thinning_id], values=['latitude', 'longitude'])
        pivoted_traj.reset_index()
        print(pivoted_traj.head())
        print(len(pivoted_traj))
        print(pivoted_traj.tail())
        pivoted_traj = pivoted_traj.ffill().bfill()
        print(pivoted_traj.head())
        print(len(pivoted_traj))
        print(pivoted_traj.tail())

        coords = pivoted_traj.columns.levels[0]
        thinners = pivoted_traj.columns.levels[1]

        num_coords = len(coords)
        assert num_coords==2, f"Incorrect coordinates, found {len(num_coords)}"
        num_trajs = len(thinners)

        print("num_trajs", num_trajs)

        assert num_trajs==len(pivoted_traj.columns) // len(coords)

        # trajectories as a raw array of shape [traj_id, time, coords]
        traj_arr = pivoted_traj.to_numpy().reshape(len(pivoted_traj), len(coords), num_trajs).transpose(2,0,1)
        traj_arr = ma.array(traj_arr)
        print(traj_arr.shape, traj_arr.count())
        saved_points = []
        count = 0
        print(traj_arr)
        while (traj_arr.count())>0 and count < int(self.min_dist * 1000):
            print(traj_arr.count())
            count += 1
            print("counter", count)

            # last indices defaulting to last element
            ind_not_masked = np.ones((num_trajs,1,len(coords))) * (traj_arr.shape[1] - 1)
            if count==0:
                print("initial ind_not_masked",ind_not_masked)
            not_masked_edges = ma.notmasked_edges(traj_arr,axis=1)[0]
            
            not_masked_shape = (len(not_masked_edges[1]) // len(coords), 1, len(coords))
            values = not_masked_edges[1].reshape(not_masked_shape)
            print(values)
            indices = not_masked_edges[0].reshape(not_masked_shape)
            print(indices)

            np.put_along_axis(ind_not_masked, indices, values, axis=0) 
            #print(ind_not_masked.shape)
            print(ind_not_masked)
            saved_points.append(np.take_along_axis(traj_arr.data, ind_not_masked.astype(np.int64), axis=1))
            traj_arr = filter_trajectories(traj_arr, saved_points[-1], self.min_dist)

        print(count)
        print(len(saved_points), saved_points[0].shape)
        print(saved_points)

        saved_points = np.stack(saved_points).squeeze()
        saved_points = saved_points.transpose(0,2,1).reshape(saved_points.shape[0],num_trajs * len(coords))
        traj_ids =  extract_df[self.thinning_id].drop_duplicates()
        print(traj_ids)
        new_cols = [f'latitude_{idx}' for idx in traj_ids ] + [f'longitude_{idx}' for idx in traj_ids ]
        extracted_trajs = pd.DataFrame(saved_points, columns=new_cols)
        print(extracted_trajs.head())

        melted = extracted_trajs.melt(var_name='variable', value_name='value')
        print(melted)

        melted[['type', self.thinning_id]] = melted['variable'].str.split('_', expand=True)
        melted = melted.drop(columns=['variable'])
        print(melted)
        # Pivot to get the desired structure
        pivoted = melted.pivot_table(
                    index=melted.index,  # or use a unique identifier if you have one
                    columns=['type', 'aircraftRegistrationNumberOrOtherIdentification'],
                    values='value'
                ).reset_index(drop=True)
        # sort along dates
        print(pivoted)
        pivoted = pivoted.stack(level=0)
        print(pivoted)
        pivoted = pivoted.reset_index()
        print(pivoted)

        # Rename columns for clarity
        #pivoted.columns = [
        #    'aircraftRegistrationNumberOrOtherIdentification',
        #    'latitude',
        #    'longitude'
        #]
        #print(pivoted)

        final = pivoted.pivot(
        index=['index', 'aircraftRegistrationNumberOrOtherIdentification'],
        columns='type',
        values='value'
        ).reset_index()
        print(final)

    # Flatten column names and drop the old index if needed
        final.columns = ['index', 'aircraftRegistrationNumberOrOtherIdentification', 'latitude', 'longitude']

        exit(0)
        df = df.sort_values('dates')

        return df