# IONO_SCINT_MAP - An ionospheric scintillation map generation toolkit
# Copyright (C) 2026  André Ricardo Fazanaro Martinon, Stephan Stephany, and
# Eurico Rodrigues de Paula
#
# This program is free software; you can redistribute it and/or# modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, see <https://www.gnu.org/licenses/>.
import numpy as np
import polars as pl

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import iono_scint_map.features as features

from iono_scint_map.dataset import (ScintillationIndex, ScintillationMapDataset,
                                    PreprocessingOptions)
from iono_scint_map.interpolation import InterpolationOptions
from iono_scint_map.util import Benchmark


class DatasetProcessingStage(ABC):
    @abstractmethod
    def validate(self,
                dataset: ScintillationMapDataset) -> bool:
        pass

    @abstractmethod
    def process(self,
                dataset: ScintillationMapDataset) -> ScintillationMapDataset:
        pass

class DatasetProcessingPipeline:
    def __init__(self):
        self.stages: List[DatasetProcessingStage] = []

    def add_stage(self,
                  stage: DatasetProcessingStage):
        if stage and stage not in self.stages:
            self.stages.append(stage)
        return self

    def process(self,
                dataset: ScintillationMapDataset) -> ScintillationMapDataset:
        for stage in self.stages:
            with Benchmark(f'PIPELINE STAGE [{stage.__class__.__name__}]'):
                if stage.validate(dataset):
                    dataset = stage.process(dataset)
        return dataset


class DataCleaningAndFiltering(DatasetProcessingStage):
    def __init__(self):
        pass

    def validate(self, dataset: ScintillationMapDataset) -> bool:
        if (len(dataset.scint_data) > 0 and len(dataset.station_data) > 0 and
                isinstance(dataset.scint_index, ScintillationIndex)):
            return True
        else:
            return False

    def process(self,
                dataset: ScintillationMapDataset) -> ScintillationMapDataset:

        # Elevation cut-off
        dataset.scint_data = dataset.scint_data.filter(
            pl.col('el') >= dataset.elevation)

        # Constellations
        dataset.scint_data = dataset.scint_data.filter(
            pl.col('constellation').is_in(
                [str(constellation) for constellation in
                 dataset.constellations]))

        # Remove data from specific stations
        dataset.scint_data = dataset.scint_data.filter(
            ~pl.col('station').is_in(dataset.remove_stations))

        # Default p values for NaN
        spectral = f'p_{dataset.scint_index.signal}'
        spectral_values = dataset.scint_data[spectral].to_numpy(writable=True)
        spectral_values[np.isnan(spectral_values)] = dataset.default_p
        dataset.scint_data = dataset.scint_data.replace_column(
            dataset.scint_data.get_column_index(spectral),
            pl.Series(spectral, spectral_values, pl.Float32)
        )

        # Drop rows with nans and nulls
        dataset.scint_data = dataset.scint_data.drop_nans()
        dataset.scint_data = dataset.scint_data.drop_nulls()

        # Remove outliers
        scint_index_values = dataset.scint_data[
            dataset.scint_index.value].to_numpy(writable=True)

        low_limit = dataset.scint_index.limits['min']
        high_limit = dataset.scint_index.limits['max']

        scint_index_values[scint_index_values < low_limit] = low_limit
        scint_index_values[scint_index_values > high_limit] = high_limit

        dataset.scint_data = dataset.scint_data.replace_column(
            dataset.scint_data.get_column_index(dataset.scint_index.value),
            pl.Series(dataset.scint_index.value, scint_index_values, pl.Float32)
        )

        return dataset


class IPPProjection(DatasetProcessingStage):
    def __init__(self):
        pass

    def validate(self, dataset: ScintillationMapDataset) -> bool:
        if (len(dataset.scint_data) > 0 and len(dataset.station_data) > 0 and
                isinstance(dataset.scint_index, ScintillationIndex)):
            return True
        else:
            return False

    def process(self,
                dataset: ScintillationMapDataset) -> ScintillationMapDataset:
        dataset.scint_data = dataset.scint_data.join(
            dataset.station_data.select(
                ['name', 'lat', 'lon']), left_on='station', right_on='name')
        dataset.scint_data = dataset.scint_data.rename({'lat': 'r_lat',
                                                        'lon': 'r_lon'})
        i_lat, i_lon = features.ipp_geometric_method(
            dataset.scint_data['r_lat'].to_numpy(),
            dataset.scint_data['r_lon'].to_numpy(),
            dataset.scint_data['az'].to_numpy(),
            dataset.scint_data['el'].to_numpy()
        )
        dataset.scint_data = dataset.scint_data.with_columns(i_lat=i_lat,
                                                             i_lon=i_lon)
        return dataset

class ScintIndexProjection(DatasetProcessingStage):
    def __init__(self):
        pass

    def validate(self, dataset: ScintillationMapDataset) -> bool:
        if (len(dataset.scint_data) > 0 and len(dataset.station_data) > 0 and
                isinstance(dataset.scint_index, ScintillationIndex)):
            return True
        else:
            return False

    def process(self,
                dataset: ScintillationMapDataset) -> ScintillationMapDataset:
        if dataset.preprocessing.projection == 'vertical':
            projection_function = (f'{dataset.scint_index.index}_'
                                   f'vertical_projection')
            vertical_projection = getattr(features, projection_function)

            spectral = f'p_{dataset.scint_index.signal}'

            scint_index_vertical = vertical_projection(
                dataset.scint_data[dataset.scint_index.value].to_numpy(),
                dataset.scint_data['el'].to_numpy(),
                dataset.scint_data[spectral].to_numpy()
            )
            dataset.scint_data = dataset.scint_data.with_columns(
                s4v=scint_index_vertical)

            dataset.scint_data = dataset.scint_data.rename(
                {'s4v': f'{dataset.scint_index.value}v'})

        return dataset


class IPPGrouping(DatasetProcessingStage):
    def __init__(self):
        pass

    @staticmethod
    def to_bin(x, step):
        return np.round(x / step) * step

    def validate(self, dataset: ScintillationMapDataset) -> bool:
        if (len(dataset.scint_data) > 0 and len(dataset.station_data) > 0 and
                isinstance(dataset.scint_index, ScintillationIndex)):
            return True
        else:
            return False

    def process(self,
                dataset: ScintillationMapDataset) -> ScintillationMapDataset:
        if dataset.preprocessing.grouping == 'regular':
            round_lat = IPPGrouping.to_bin(
                dataset.scint_data['i_lat'].to_numpy(),
                step=dataset.ipp_group_resolution
            )

            round_lon = IPPGrouping.to_bin(
                dataset.scint_data['i_lon'].to_numpy(),
                step=dataset.ipp_group_resolution
            )

            dataset.scint_data = dataset.scint_data.with_columns(
                round_lat=round_lat,
                round_lon=round_lon
            )

        return dataset


class IPPAggregation(DatasetProcessingStage):
    def __init__(self):
        pass

    @staticmethod
    def sai(scint_index: str, df: pl.DataFrame):
        agg_scint_index = np.array([], dtype='float32')
        lat = np.array([], dtype='float32')
        lon = np.array([], dtype='float32')

        for name, data in df.group_by(['round_lat', 'round_lon']):
            agg_scint_index = np.append(
                agg_scint_index, data[scint_index].mean()
            )
            lat = np.append(lat, data['i_lat'].mean())
            lon = np.append(lon, data['i_lon'].mean())

        grouped = pl.DataFrame(data={
            'scint_index': agg_scint_index, 'lat': lat, 'lon': lon,
        }, schema={
            'scint_index': pl.Float32, 'lat': pl.Float32, 'lon': pl.Float32
        }).sort(by='lat', descending=True)

        return grouped

    @staticmethod
    def smi(scint_index: str, df: pl.DataFrame):
        agg_scint_index = np.array([], dtype='float32')
        lat = np.array([], dtype='float32')
        lon = np.array([], dtype='float32')

        for name, data in df.group_by(['round_lat', 'round_lon']):
            idx_max = data[scint_index].arg_max()

            agg_scint_index = np.append(
                agg_scint_index, data[scint_index][idx_max]
            )
            lat = np.append(lat, data['i_lat'][idx_max])
            lon = np.append(lon, data['i_lon'][idx_max])

        grouped = pl.DataFrame(data={
            'scint_index': agg_scint_index, 'lat': lat, 'lon': lon,
        }, schema={
            'scint_index': pl.Float32, 'lat': pl.Float32, 'lon': pl.Float32
        }).sort(by='lat', descending=True)

        return grouped

    @staticmethod
    def sqi(scint_index: str, df: pl.DataFrame):
        agg_scint_index = np.array([], dtype='float32')
        lat = np.array([], dtype='float32')
        lon = np.array([], dtype='float32')
        with Benchmark('AGG QUANTILE'):
            for name, data in df.group_by(['round_lat', 'round_lon']):
                new_data = data.filter(
                    pl.col(scint_index) >= pl.col(scint_index).quantile(
                        quantile=0.75, interpolation='linear')
                )

                agg_scint_index = np.append(
                    agg_scint_index, new_data[scint_index].mean()
                )
                lat = np.append(lat, new_data['i_lat'].mean())
                lon = np.append(lon, new_data['i_lon'].mean())

        with Benchmark('CREATE DATAFRAME'):
            grouped = pl.DataFrame(data={
                'scint_index': agg_scint_index, 'lat': lat, 'lon': lon,
            }, schema={
                'scint_index': pl.Float32, 'lat': pl.Float32, 'lon': pl.Float32
            }).sort(by='lat', descending=True)

        return grouped

    @staticmethod
    def vai(scint_index: str, df: pl.DataFrame):
        scint_index = f'{scint_index}v'
        return IPPAggregation.sai(scint_index, df)

    @staticmethod
    def vmi(scint_index: str, df: pl.DataFrame):
        scint_index = f'{scint_index}v'
        return IPPAggregation.smi(scint_index, df)

    @staticmethod
    def vqi(scint_index: str, df: pl.DataFrame):
        scint_index = f'{scint_index}v'
        return IPPAggregation.sqi(scint_index, df)

    def validate(self, dataset: ScintillationMapDataset) -> bool:
        if (len(dataset.scint_data) > 0 and len(dataset.station_data) > 0 and
                isinstance(dataset.scint_index, ScintillationIndex)):
            return True
        else:
            return False

    def process(self,
                dataset: ScintillationMapDataset) -> ScintillationMapDataset:

        aggregation_function = getattr(IPPAggregation,
                                       dataset.preprocessing.value)
        dataset.grouped = aggregation_function(dataset.scint_index.value,
                                               dataset.scint_data)
        dataset.grouped = dataset.grouped.rename(
            {'scint_index': dataset.scint_index.value})

        return dataset

class MapInterpolation(DatasetProcessingStage):
    def __init__(self):
        pass

    def validate(self, dataset: ScintillationMapDataset) -> bool:
        if isinstance(dataset.grouped, pl.DataFrame):
            return True
        else:
            return False

    def process(self,
                dataset: ScintillationMapDataset) -> ScintillationMapDataset:

        interp_obj = dataset.interpolation.interpolation_class(
            extent=dataset.map_extent,
            step=dataset.interpolation_grid_resolution)

        print(interp_obj, interp_obj.shape, dataset.grouped.shape)

        interp_obj.interpolate(
            dataset.grouped['lon'].to_numpy(),
            dataset.grouped['lat'].to_numpy(),
            dataset.grouped[dataset.scint_index.value].to_numpy()
        )

        interp_obj.interpolated_map[np.isnan(interp_obj.interpolated_map)] = 0
        dataset.interpolated_map = interp_obj.interpolated_map.copy()

        return dataset

if __name__ == '__main__':
    input_dir = '/environment/development/inpe/src/iono-scint-map/tests_data/'
    input_dir = Path(input_dir).resolve()
    scint_map_data = ScintillationMapDataset(
        ScintillationIndex.S4_1, interpolation=InterpolationOptions.GPR)
    scint_map_data.add_scintillation_data(input_dir / 'train_map_data.csv')
    scint_map_data.add_station_data(input_dir / 'inct_stations.parquet')

    print(scint_map_data.scint_data.shape)
    print(np.nansum(scint_map_data.scint_data['p_1'].to_numpy()))
    print(np.nanmin(scint_map_data.scint_data['s4_1'].to_numpy()))
    print(np.nanmax(scint_map_data.scint_data['s4_1'].to_numpy()))

    scint_map_pipeline = DatasetProcessingPipeline()
    scint_map_pipeline.add_stage(DataCleaningAndFiltering())
    scint_map_pipeline.add_stage(IPPProjection())
    scint_map_pipeline.add_stage(ScintIndexProjection())
    scint_map_pipeline.add_stage(IPPGrouping())
    scint_map_pipeline.add_stage(IPPAggregation())
    scint_map_pipeline.add_stage(MapInterpolation())

    scint_map_data = scint_map_pipeline.process(scint_map_data)

    print(scint_map_data.scint_data.shape)
    print(np.sum(scint_map_data.scint_data['p_1'].to_numpy()))
    print(np.min(scint_map_data.scint_data['s4_1'].to_numpy()))
    print(np.max(scint_map_data.scint_data['s4_1'].to_numpy()))

    print(scint_map_data.scint_data.select(['s4_1', 's4_1v', 'el', 'p_1',
                                            'round_lat', 'round_lon',
                                            'i_lat', 'i_lon']))
    print(scint_map_data.grouped)

    print(scint_map_data.interpolated_map)
    print(scint_map_data.interpolated_map.shape)

