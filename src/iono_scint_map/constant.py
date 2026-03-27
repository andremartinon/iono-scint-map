# This file is part of IONO-SCINT-MAP - an ionospheric scintillation map
# generation toolkit.
#
# Copyright (C) 2026 National Institute for Space Research (INPE)
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

import os
import polars as pl

from typing import List, Iterable

__author__ = ['André Ricardo Fazanaro Martinon']
__copyright__ = 'Copyright 2026, National Institute for Space Research (INPE)'
__credits__ = ['Stephan Stephany', 'Eurico Rodrigues de Paula']
__license__ = 'AGPL-3.0-or-later'
__maintainer__ = 'André Ricardo Fazanaro Martinon'
__email__ = 'andre.martinon@inpe.br'
__status__ = 'Production'


CONSOLE_WIDTH: int = 80

EARTH_RADIUS_KM: float = 6371.0087714

IONO_SCINT_MAP_LOG_LEVEL = os.getenv('IONO_SCINT_MAP_LOG_LEVEL', 'INFO')

SCINT_DATA_DTYPES: dict[str, type[pl.Datetime|pl.UInt16|pl.Float32|pl.String]] = {
    'datetime': pl.Datetime,
    'prn': pl.UInt16,
    'az': pl.Float32,
    'el': pl.Float32,
    'station': pl.String,
    'constellation': pl.String
}

SCINT_DATA_MUST_HAVE_COLUMNS: List[Iterable[str]] = [
    ('datetime', 'timestamp',),
    ('prn', 'svid',),
    ('az', 'azimuth',),
    ('el', 'elevation',),
    ('station', 'station_id', 'station_name',),
    ('constellation', 'sat_system', 'gnss_system', 'system',)
]

STATION_DATA_DTYPES: dict[str, type[pl.Float32|pl.String]] = {
    'name': pl.String,
    'lat': pl.Float32,
    'lon': pl.Float32,
    'alt': pl.Float32,
    'x': pl.Float32,
    'y': pl.Float32,
    'z': pl.Float32
}

STATION_DATA_MUST_HAVE_COLUMNS: List[Iterable[str]] = [
    ('name', 'id', 'station_name', 'station_id',),
    ('lat', 'latitude',),
    ('lon', 'longitude', 'long',),
    ('alt', 'altitude',),
    ('x',),
    ('y',),
    ('z',)
]
