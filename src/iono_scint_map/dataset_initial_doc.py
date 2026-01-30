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
import enum

import numpy as np
import polars as pl

from functools import singledispatchmethod
from pathlib import Path
from typing import List, Optional, Tuple, Iterable

from iono_scint_map.constant import (SCINT_DATA_MUST_HAVE_COLUMNS,
                                     STATION_DATA_MUST_HAVE_COLUMNS,
                                     SCINT_DATA_DTYPES, STATION_DATA_DTYPES)
from iono_scint_map.interpolation import InterpolationOptions


class Constellation(enum.StrEnum):
    GPS = 'GPS'
    GLONASS = 'GLONASS'
    GALILEO = 'GALILEO'
    BEIDOU = 'BEIDOU'
    SBAS = 'SBAS'
    QZSS = 'QZSS'
    NAVIC = 'NAVIC'
    TBD = 'TBD'

class ScintillationType(enum.StrEnum):
    AMPLITUDE = 'amplitude'
    PHASE = 'phase'
    ROTI = 'roti'


class ScintillationIndex(enum.Enum):
    S4_1 = ('s4', 1, ScintillationType.AMPLITUDE, [0, 1.4])
    S4_2 = ('s4', 2, ScintillationType.AMPLITUDE, [0, 1.4])
    S4_3 = ('s4', 3, ScintillationType.AMPLITUDE, [0, 1.4])
    PHI60_1 = ('phi60', 1, ScintillationType.PHASE, [0, 1.4])
    PHI60_2 = ('phi60', 2, ScintillationType.PHASE, [0, 1.4])
    PHI60_3 = ('phi60', 3, ScintillationType.PHASE, [0, 1.4])
    ROTI = ('roti', 1, ScintillationType.ROTI, [0, 0.4])

    def __new__(cls, value, signal, scint_type, limits):
        obj = object.__new__(cls)
        obj._value_ = f'{value}_{signal}'
        obj.index = value
        obj.signal = signal
        obj.type = scint_type
        obj.limits = {
            'min': limits[0],
            'max': limits[1],
        }

        return obj


class PreprocessingOptions(enum.Enum):
    """
    Scintillation maps preprocessing options enumeration, each are:

    References
    ----------
        [1] 
    """
    SAI = ('sai', 'slant', 'mean', 'regular', True)
    SMI = ('smi', 'slant', 'max', 'regular', True)
    SQI = ('sqi', 'slant', 'quantile', 'regular', True)
    VAI = ('vai', 'vertical', 'mean', 'regular', True)
    VMI = ('vmi', 'vertical', 'max', 'regular', True)
    VQI = ('vqi', 'vertical', 'quantile', 'regular', True)

    def __new__(cls, value, projection, aggregation, grouping, adjust_centers):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.projection = projection
        obj.aggregation = aggregation
        obj.grouping = grouping
        obj.adjust_centers = adjust_centers

        return obj


class ScintillationMapDataset:
    """
    A data structure for storing scintillation and GNSS station data. It also
    stores the configuration required for generating the scintillation map and,
    finally, the resulting scintillation map matrix.


    ...

    Attributes
    ----------
    scint_index : ScintillationIndex
        The scintillation index present in the scintillation data that will
        be used to generate the scintillation map.

    elevation : float
        The satellite's elevation limit to avoid multipath effects on
        scintillation data.

    default_p : float
        A default value for the spectral slope of the phase PSD in the 0.1
        to 25 Hz range. Typically, 2.6, according to RINO, C. L. (1979) in
        "A power law phase screen model for ionospheric scintillation: 1.
        weak scatter". https://doi.org/10.1029/RS014i006p01135

    constellations : List[Constellation]

    remove_stations : List[str]

    preprocessing : PreprocessingOptions

    interpolation : InterpolationOptions

    ipp_group_resolution : float

    interpolation_grid_resolution : float

    map_extent : Tuple[float, float, float, float]

    scint_data : pl.DataFrame

    station_data : pl.DataFrame

    grouped : pl.DataFrame

    interpolated_map : pl.DataFrame

    Methods
    -------
    validate_dataframe()

    validate_scint_index()

    normalize_columns_name()

    cast_dataframe()

    cast_scint_index()

    add_scintillation_data()

    add_station_data()

    See also
    --------

    """
    def __init__(self,
                 scint_index: ScintillationIndex,
                 elevation: Optional[float] = 30,
                 default_p: Optional[float] = 2.6,
                 constellations: Optional[List[Constellation]] = (
                         Constellation.GPS,
                         Constellation.GLONASS,
                         Constellation.BEIDOU,
                         Constellation.GALILEO
                 ),
                 remove_stations: Optional[List[str]] = (),
                 preprocessing: Optional[PreprocessingOptions] = PreprocessingOptions.VQI,
                 interpolation: Optional[InterpolationOptions] = InterpolationOptions.GPR,
                 ipp_group_resolution: Optional[float] = 1.0,
                 interpolation_grid_resolution: Optional[float] = 0.25,
                 map_extent: Optional[Tuple[float, float, float, float]] = (
                         -78.0, -30.0, -39.0, 9.0)
                 ):
        self.scint_index: ScintillationIndex = scint_index
        self.scint_data: pl.DataFrame = None
        self.station_data: pl.DataFrame = None
        self.grouped: pl.DataFrame = None
        self.interpolated_map: np.ndarray = None

        self.elevation = elevation
        self.default_p = default_p
        self.constellations = constellations
        self.remove_stations = remove_stations
        self.preprocessing = preprocessing
        self.interpolation = interpolation
        self.ipp_group_resolution = ipp_group_resolution
        self.interpolation_grid_resolution = interpolation_grid_resolution
        self.map_extent = map_extent

    @staticmethod
    def validate_dataframe(df: pl.DataFrame,
                           must_have_columns: List[Iterable[str]]
                           ) -> bool:
        """
        Validate whether the dataframe contains the required columns.

        :param df: The dataframe to validate.
        :param must_have_columns: The required columns to validate.

        :return: True if the dataframe contains the required columns,
        False otherwise.
        """
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


if __name__ == '__main__':
    input_dir = '/environment/development/inpe/src/iono-scint-map/tests_data/'
    input_dir = Path(input_dir).resolve()
    scint_map_data = ScintillationMapDataset(ScintillationIndex.S4_1)
    scint_map_data.add_scintillation_data(input_dir / 'train_map_data.csv')
    scint_map_data.add_station_data(input_dir / 'inct_stations.parquet')

    print(scint_map_data)

    print(scint_map_data.scint_data.columns)
    print(scint_map_data.scint_data.dtypes)
    print(scint_map_data.scint_data)

    print(scint_map_data.station_data.columns)
    print(scint_map_data.station_data.dtypes)
    print(scint_map_data.station_data)
