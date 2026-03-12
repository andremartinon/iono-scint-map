# IONO-SCINT-MAP - An ionospheric scintillation map generation toolkit
# Copyright (C) 2026 National Institute for Space Research (INPE)
#
# Authors: André Ricardo Fazanaro Martinon, Stephan Stephany, and
# Eurico Rodrigues de Paula
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, see <https://www.gnu.org/licenses/>.
import enum
import h5py
import numpy as np
import polars as pl

from datetime import datetime, UTC
from functools import singledispatchmethod
from pathlib import Path
from typing import List, Optional, Tuple, Iterable

from iono_scint_map.constant import (SCINT_DATA_MUST_HAVE_COLUMNS,
                                     STATION_DATA_MUST_HAVE_COLUMNS,
                                     SCINT_DATA_DTYPES, STATION_DATA_DTYPES)
from iono_scint_map.interpolation import InterpolationOptions


class Constellation(enum.StrEnum):
    """GNSS satellites constellations.

    Used to filter scintillation data by a set of satellites constellations.
    Employed in the ScintillationMapDataset class and in command line interface.
    """
    GPS = 'GPS'
    """GPS satellites constellation - PRN: [1..32]."""

    GLONASS = 'GLONASS'
    """GLONASS satellites constellation - PRN: [1..32]."""

    GALILEO = 'GALILEO'
    """GALILEO satellites constellation - PRN: [1..32]."""

    BEIDOU = 'BEIDOU'
    """BEIDOU satellites constellation - PRN: [1..32]."""

    SBAS = 'SBAS'
    """SBAS satellites constellation - PRN: [1..32]."""

    QZSS = 'QZSS'
    """QZSS satellites constellation - PRN: [1..32]."""

    NAVIC = 'NAVIC'
    """NAVIC satellites constellation - PRN: [1..32]."""


class ScintillationType(enum.StrEnum):
    """Ionospheric scintillation types enumeration.

    Used together to ScintillationIndex enumeration to identify the
    scintillation type of each scintillation index.
    """
    AMPLITUDE = 'amplitude'
    """For ionospheric scintillation which affects the signal amplitude."""

    PHASE = 'phase'
    """For ionospheric scintillation which affects the signal phase."""

    ROTI = 'roti'
    """For ionospheric scintillation which can be measured by the ROTI (Rate of
    change of TEC index).
    """


class ScintillationIndex(enum.Enum):
    """Ionospheric scintillation index enumeration.


    """
    S4_1 = ('s4', 1, ScintillationType.AMPLITUDE, [0, 1.4])

    S4_2 = ('s4', 2, ScintillationType.AMPLITUDE, [0, 1.4])

    S4_3 = ('s4', 3, ScintillationType.AMPLITUDE, [0, 1.4])

    PHI60_1 = ('phi60', 1, ScintillationType.PHASE, [0, 1.4])

    PHI60_2 = ('phi60', 2, ScintillationType.PHASE, [0, 1.4])

    PHI60_3 = ('phi60', 3, ScintillationType.PHASE, [0, 1.4])

    ROTI = ('roti', 1, ScintillationType.ROTI, [0, 0.4])

    def __new__(cls, value, signal, scint_type, limits):
        obj = object.__new__(cls)
        obj._value_ = f'{value}' if value == 'roti' else f'{value}_{signal}'
        obj.index = value
        obj.signal = signal
        obj.type = scint_type
        obj.limits = {
            'min': limits[0],
            'max': limits[1],
        }

        return obj


class PreprocessingOptions(enum.Enum):
    SAI = ('sai', 'slant', 'mean', 'regular', True)
    SMI = ('smi', 'slant', 'max', 'regular', True)
    SQI = ('sqi', 'slant', 'quantile', 'regular', True)
    VAI = ('vai', 'vertical', 'mean', 'regular', True)
    VMI = ('vmi', 'vertical', 'max', 'regular', True)
    VQI = ('vqi', 'vertical', 'quantile', 'regular', True)

    def __new__(cls, value: str, projection: str, aggregation: str,
                grouping: str, adjust_centers: bool):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.projection = projection
        obj.aggregation = aggregation
        obj.grouping = grouping
        obj.adjust_centers = adjust_centers

        return obj


class ScintillationMapDataset:
    def __init__(self,
                 scint_index: ScintillationIndex,
                 elevation: float = 30,
                 default_p: float = 2.6,
                 constellations: Iterable[Constellation] = (
                         Constellation.GPS,
                         Constellation.GLONASS,
                         Constellation.BEIDOU,
                         Constellation.GALILEO
                 ),
                 remove_stations: Iterable[str] = (),
                 preprocessing: PreprocessingOptions = PreprocessingOptions.VQI,
                 interpolation: InterpolationOptions = InterpolationOptions.GPR,
                 ipp_group_resolution: float = 1.0,
                 interpolation_grid_resolution: float = 0.25,
                 map_extent: Tuple[float, float, float, float] =
                 (-78.0, -30.0, -39.0, 9.0),
                 start_timestamp: datetime = None,
                 end_timestamp: datetime = None):

        self.scint_index: ScintillationIndex = scint_index
        self.scint_data: pl.DataFrame = pl.DataFrame()
        self.station_data: pl.DataFrame = pl.DataFrame()
        self.grouped: pl.DataFrame = pl.DataFrame()
        self.interpolated_map: np.ndarray = np.array([])

        self.elevation: float = elevation
        self.default_p: float = default_p
        self.constellations: Iterable[Constellation] = constellations
        self.remove_stations: Iterable[str] = remove_stations
        self.preprocessing: PreprocessingOptions = preprocessing
        self.interpolation: InterpolationOptions = interpolation
        self.ipp_group_resolution: float = ipp_group_resolution
        self.interpolation_grid_resolution: float = interpolation_grid_resolution
        self.map_extent: Tuple[float, float, float, float] = map_extent
        self.start_timestamp: datetime = start_timestamp
        self.end_timestamp: datetime = end_timestamp

    @staticmethod
    def validate_dataframe(df: pl.DataFrame,
                           must_have_columns: List[Iterable[str]]
                           ) -> bool:
        for must_have_column in must_have_columns:
            if not set(df.columns).intersection(must_have_column):
                return False

        return True

    @staticmethod
    def validate_scint_index(df: pl.DataFrame,
                             scint_index: ScintillationIndex) -> bool:
        if scint_index.value not in df.columns:
             return False
        else:
            return True

    @staticmethod
    def normalize_columns_name(df: pl.DataFrame,
                               must_have_columns: list) -> pl.DataFrame:
        for must_have_column in must_have_columns:
            df = df.rename(dict.fromkeys(must_have_column, must_have_column[0]),
                           strict=False)
        return df

    @staticmethod
    def cast_dataframe(df: pl.DataFrame, dtypes: dict) -> pl.DataFrame:
        for column, dtype in dtypes.items():
            if column == 'datetime':
                if  df[column].dtype == pl.String:
                    df = df.with_columns(pl.col(column).str.to_datetime(
                        '%Y-%m-%d %H:%M:%S', time_unit='ms'))
            else:
                df = df.with_columns(pl.col(column).cast(dtype))
        return df

    @staticmethod
    def cast_scint_index(df: pl.DataFrame,
                         scint_index: ScintillationIndex) -> pl.DataFrame:
        df = df.with_columns(
            pl.col(str(scint_index.value)).cast(
                pl.Float32))
        if f'p_{scint_index.signal}' in df.columns:
            df = df.with_columns(
                pl.col(f'p_{scint_index.signal}').cast(pl.Float32))

        return df

    @singledispatchmethod
    def add_scintillation_data(self, arg):
        raise NotImplementedError("Cannot read scintillation data")

    @add_scintillation_data.register
    def _(self, df: pl.DataFrame):
        if not (
            ScintillationMapDataset.validate_dataframe(
                df, SCINT_DATA_MUST_HAVE_COLUMNS) and
            ScintillationMapDataset.validate_scint_index(
                df, self.scint_index)
        ):
            raise ValueError('Invalid scintillation data file')
        else:
            self.scint_data = ScintillationMapDataset.cast_scint_index(
                ScintillationMapDataset.cast_dataframe(
                    ScintillationMapDataset.normalize_columns_name(
                        df, SCINT_DATA_MUST_HAVE_COLUMNS),
                    SCINT_DATA_DTYPES),
                self.scint_index
            )
            if self.start_timestamp is None and self.end_timestamp is None:
                self.start_timestamp = self.scint_data['datetime'].dt.min()
                self.end_timestamp = self.scint_data['datetime'].dt.max()

    @add_scintillation_data.register
    def _(self, file_path: Path):
        if file_path.is_file() and file_path.suffix == '.csv':
            df = pl.read_csv(file_path, infer_schema=False)
        elif file_path.is_file() and file_path.suffix == '.parquet':
            df = pl.read_parquet(file_path)
        else:
            raise NotImplementedError("Cannot read scintillation data")

        if not (
            ScintillationMapDataset.validate_dataframe(
                df, SCINT_DATA_MUST_HAVE_COLUMNS) and
            ScintillationMapDataset.validate_scint_index(
                df, self.scint_index)
        ):
            raise ValueError('Invalid scintillation data file')
        else:
            self.scint_data = ScintillationMapDataset.cast_scint_index(
                ScintillationMapDataset.cast_dataframe(
                    ScintillationMapDataset.normalize_columns_name(
                        df, SCINT_DATA_MUST_HAVE_COLUMNS),
                    SCINT_DATA_DTYPES),
                self.scint_index
            )
            if self.start_timestamp is None and self.end_timestamp is None:
                self.start_timestamp = self.scint_data['datetime'].dt.min()
                self.end_timestamp = self.scint_data['datetime'].dt.max()

    @singledispatchmethod
    def add_station_data(self, arg):
        raise NotImplementedError("Cannot read station data")

    @add_station_data.register
    def _(self, df: pl.DataFrame):
        if not ScintillationMapDataset.validate_dataframe(
                df, STATION_DATA_MUST_HAVE_COLUMNS):
            raise ValueError('Invalid station data file')
        else:
            self.station_data = ScintillationMapDataset.cast_dataframe(
                ScintillationMapDataset.normalize_columns_name(
                    df, STATION_DATA_MUST_HAVE_COLUMNS),
                STATION_DATA_DTYPES)

    @add_station_data.register
    def _(self, file_path: Path):
        if file_path.is_file() and file_path.suffix == '.csv':
            df = pl.read_csv(file_path, infer_schema=False)
        elif file_path.is_file() and file_path.suffix == '.parquet':
            df = pl.read_parquet(file_path)
        else:
            raise NotImplementedError("Cannot read station data")

        if not ScintillationMapDataset.validate_dataframe(
                df, STATION_DATA_MUST_HAVE_COLUMNS):
            raise ValueError('Invalid station data file')
        else:
            self.station_data = ScintillationMapDataset.cast_dataframe(
                ScintillationMapDataset.normalize_columns_name(
                    df, STATION_DATA_MUST_HAVE_COLUMNS),
                STATION_DATA_DTYPES)

    def to_hdf5(self, file_path: Path = 'scint_map.hdf5'):
        with h5py.File(Path(file_path), 'w') as f:
            # SCINTILLATION MAP
            map = f.create_dataset(name='scint-map',
                                 shape=self.interpolated_map.shape,
                                 dtype=np.float32,
                                 compression="gzip",
                                 compression_opts=9,
                                 data=self.interpolated_map)

            map.attrs['start_timestamp'] =\
                self.start_timestamp.strftime('%Y-%m-%dT%H:%M:%S')

            map.attrs['end_timestamp'] =\
                self.end_timestamp.strftime('%Y-%m-%dT%H:%M:%S')

            map.attrs['preprocessing'] = self.preprocessing.value.upper()
            map.attrs['elevation'] = self.elevation
            map.attrs['constellations'] = [c.value for c in self.constellations]
            map.attrs['remove_stations'] = self.remove_stations
            map.attrs['scint_index'] = self.scint_index.index.upper()
            map.attrs['scint_index_signal'] = self.scint_index.signal
            map.attrs['scint_index_type'] = self.scint_index.type.upper()
            map.attrs['scint_index_limit'] = (self.scint_index.limits['min'],
                                            self.scint_index.limits['max'])
            map.attrs['extent'] = self.map_extent
            map.attrs['interpolation_grid_resolution'] =\
                self.interpolation_grid_resolution

            map.attrs['ipp_group_resolution'] = self.ipp_group_resolution
            map.attrs['default_p'] = self.default_p
            map.attrs['interpolation'] = self.interpolation.value.upper()
            map.attrs['stations'] =\
                self.scint_data['station'].unique().sort().to_list()

            for station in map.attrs['stations']:
                station_info = self.station_data.filter(
                    pl.col('name') == station)
                map.attrs[station] = [
                    station_info['lat'],
                    station_info['lon'],
                    station_info['alt']
                ]

            # IPP PROJECTION DATA
            ipps_array = self.scint_data.with_columns(
                (
                    pl.col('datetime').dt.timestamp(time_unit='ms') * 1e-3
                ).alias('timestamp')
            ).select([
                'timestamp',
                self.scint_index.value,
                'prn',
                'i_lat',
                'i_lon',
                'constellation'
            ]).to_numpy()

            constellation_index = lambda x: Constellation._member_names_.index(
                Constellation(x).name)
            ipps_array[:, 5] = np.vectorize(
                constellation_index)(ipps_array[:, 5])

            ipps = f.create_dataset(name='ipps',
                                    shape=ipps_array.shape,
                                    dtype=np.float64,
                                    compression="gzip",
                                    compression_opts=9,
                                    data=ipps_array.astype(np.float64))

            ipps.attrs['columns'] = ['datetime',
                                     self.scint_index.value,
                                     'prn',
                                     'lat',
                                     'lon',
                                     'constellation']

            # PROJECTED, GROUPED AND AGGREGATED IPPs DATA
            agg_ipps_array = self.grouped.to_numpy()
            ipps = f.create_dataset(name='agg-ipps',
                                    shape=agg_ipps_array.shape,
                                    dtype=np.float32,
                                    compression="gzip",
                                    compression_opts=9,
                                    data=agg_ipps_array)
            ipps.attrs['columns'] = [self.scint_index.value, 'lat', 'lon']
            ipps.attrs['aggregation'] = self.preprocessing.aggregation
            ipps.attrs['grouping'] = self.preprocessing.grouping
            ipps.attrs['projection'] = self.preprocessing.projection
            ipps.attrs['adjust_centers'] = self.preprocessing.adjust_centers

    @staticmethod
    def from_hdf5(file_path: Path):
        with (h5py.File(Path(file_path), 'r') as f):
            # SCINTILLATION MAP
            scint_index = f"{f['scint-map'].attrs['scint_index']}"
            if scint_index.lower() != 'roti':
                scint_index = (scint_index +
                               f"_{f['scint-map'].attrs['scint_index_signal']}")
            scint_index = ScintillationIndex(scint_index.lower())

            start = datetime.strptime(f['scint-map'].attrs['start_timestamp'],
                                      '%Y-%m-%dT%H:%M:%S')
            end = datetime.strptime(f['scint-map'].attrs['end_timestamp'],
                                    '%Y-%m-%dT%H:%M:%S')

            preprocessing = PreprocessingOptions(
                f['scint-map'].attrs['preprocessing'].lower())

            elevation = f['scint-map'].attrs['elevation']
            constellations = [Constellation(c)
                              for c in f['scint-map'].attrs['constellations']]
            remove_stations = f['scint-map'].attrs['remove_stations']
            extent = f['scint-map'].attrs['extent']
            interpolation_grid_resolution = f['scint-map'].attrs[
                'interpolation_grid_resolution']
            ipp_group_resolution = f['scint-map'].attrs['ipp_group_resolution']
            default_p = f['scint-map'].attrs['default_p']
            interpolation = InterpolationOptions(
                f['scint-map'].attrs['interpolation'].lower())

            scint_map_data = ScintillationMapDataset(
                scint_index,
                elevation=elevation,
                default_p=default_p,
                constellations=constellations,
                remove_stations=remove_stations,
                preprocessing=preprocessing,
                interpolation = interpolation,
                ipp_group_resolution=ipp_group_resolution,
                interpolation_grid_resolution=interpolation_grid_resolution,
                map_extent=extent,
                start_timestamp=start,
                end_timestamp=end)
            scint_map_data.interpolated_map = f['scint-map'][:]

            station_list = []
            for station in f['scint-map'].attrs['stations']:
                coordinates = f['scint-map'].attrs[station][:, 0]

                station_list.append({
                    'name': station,
                    'lon': coordinates[1],
                    'lat': coordinates[0],
                    'alt': coordinates[2]
                })
            scint_map_data.station_data = pl.from_dicts(station_list)

            # IPP PROJECTION DATA
            scint_map_data.scint_data = pl.from_numpy(
                f['ipps'][:], schema=list(f['ipps'].attrs['columns'])
            )

            scint_map_data.scint_data = scint_map_data.scint_data.with_columns(
                pl.col('datetime').map_elements(
                    lambda x: datetime.fromtimestamp(int(x), UTC),
                    return_dtype=pl.Datetime),
                pl.col('prn').cast(pl.UInt16),
                pl.col('constellation').map_elements(
                    lambda x: Constellation._member_names_[int(x)],
                    return_dtype=pl.String
                ),
                pl.col(str(scint_index.value)).cast(pl.Float32),
                pl.col('lat').cast(pl.Float32),
                pl.col('lon').cast(pl.Float32)
            )

            # PROJECTED, GROUPED AND AGGREGATED IPPs DATA
            scint_map_data.grouped = pl.from_numpy(
                f['agg-ipps'][:], schema=list(f['agg-ipps'].attrs['columns'])
            )

        return scint_map_data
