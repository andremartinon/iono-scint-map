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

import numpy as np

from numba import njit
from typing import Union, Iterable, Tuple

from iono_scint_map.constant import EARTH_RADIUS

# @njit(parallel=True)
def ipp_geometric_method(receiver_lat: Union[Iterable[float], float],
                         receiver_lon:  Union[Iterable[float], float],
                         azimuth: Union[Iterable[float], float],
                         elevation: Union[Iterable[float], float],
                         ipp_height: float = 350.0) -> \
        Tuple[Union[Iterable[float], float], Union[Iterable[float], float]]:

    """Calculates the Ionospheric Pierce Point (IPP) latitude and longitude.
    Using the geometric method.

    Parameters
    ----------
    receiver_lat : array of float or float
        Receiver latitude in degrees.
    receiver_lon : array of float or float
        Receiver longitude in degrees.
    azimuth : array of float or float
        Satellite azimuth in degrees.
    elevation : array of float or float
        Satellite elevation in degrees.
    ipp_height : float
        IPP height in km.

    Returns
    -------
    tuple
        A tuple of IPP latitudes and longitudes in degrees.
    """

    # Average Earth radius in meters
    re = EARTH_RADIUS * 1000

    # IPP height in meters
    h = ipp_height * 1000

    az = np.deg2rad(azimuth)
    el = np.deg2rad(elevation)

    r_lat = np.deg2rad(receiver_lat)
    r_lon = np.deg2rad(receiver_lon)

    psi = np.arccos((re / (re + h)) * np.cos(el)) - el

    lat = r_lat + (psi * np.cos(az))
    lon = r_lon + (psi * np.sin(az) / np.cos(lat))

    return np.rad2deg(lat), np.rad2deg(lon)


def s4_vertical_projection(slant_s4: Union[Iterable[float], float],
                           elevation: Union[Iterable[float], float],
                           p: Union[Iterable[float], float] = 2.6,
                           ipp_height: float = 350) ->\
        Union[Iterable[float], float]:

    # Average Earth radius in meters
    re = EARTH_RADIUS * 1000

    # IPP height in meters
    h = ipp_height * 1000

    # Elevation in radians
    el = np.deg2rad(elevation)

    f = 1 / np.sqrt(1 - (re * np.cos(el) / (re + h)) ** 2)

    b = (p + 1) / 4

    return slant_s4 * (1 / f) ** b


def s4_slant_projection(vertical_s4: Union[Iterable[float], float],
                        elevation: Union[Iterable[float], float],
                        p: Union[Iterable[float], float] = 2.6,
                        ipp_height: float = 350) -> \
        Union[Iterable[float], float]:

    # Average Earth radius in meters
    re = EARTH_RADIUS * 1000

    # IPP height in meters
    h = ipp_height * 1000

    # Elevation in radians
    el = np.deg2rad(elevation)

    f = 1 / np.sqrt(1 - (re * np.cos(el) / (re + h)) ** 2)

    b = (p + 1) / 4

    return vertical_s4 * (f / 1) ** b


def phi60_vertical_projection(slant_phi: Union[Iterable[float], float],
                              elevation: Union[Iterable[float], float],
                              p: Union[Iterable[float], float] = 2.6,
                              ipp_height: float = 350) -> Union[Iterable[float], float]:

    # Average Earth radius in meters
    re = EARTH_RADIUS * 1000

    # IPP height in meters
    h = ipp_height * 1000

    # Elevation in radians
    el = np.deg2rad(elevation)

    f = 1 / np.sqrt(1 - (re * np.cos(el) / (re + h)) ** 2)

    b = 0.5

    return slant_phi * (1 / f) ** b


def phi60_slant_projection(vertical_phi: Union[Iterable[float], float],
                           elevation: Union[Iterable[float], float],
                           p: Union[Iterable[float], float] = 2.6,
                           ipp_height: float = 350) -> Union[Iterable[float], float]:

    # Average Earth radius in meters
    re = EARTH_RADIUS * 1000

    # IPP height in meters
    h = ipp_height * 1000

    # Elevation in radians
    el = np.deg2rad(elevation)

    f = 1 / np.sqrt(1 - (re * np.cos(el) / (re + h)) ** 2)

    b = 0.5

    return vertical_phi * (f / 1) ** b


def roti_vertical_projection(slant_roti: Union[Iterable[float], float],
                             elevation: Union[Iterable[float], float],
                             ipp_height: float = 350) -> Union[Iterable[float], float]:

    # Average Earth radius in meters
    re = EARTH_RADIUS * 1000

    # IPP height in meters
    h = ipp_height * 1000

    # Elevation in radians
    el = np.deg2rad(elevation)

    f = 1 / np.sqrt(1 - (re * np.cos(el) / (re + h)) ** 2)

    return slant_roti * (1 / f)


def roti_slant_projection(vertical_roti: Union[Iterable[float], float],
                          elevation: Union[Iterable[float], float],
                          ipp_height: float = 350) -> Union[Iterable[float], float]:

    # Average Earth radius in meters
    re = EARTH_RADIUS * 1000

    # IPP height in meters
    h = ipp_height * 1000

    # Elevation in radians
    el = np.deg2rad(elevation)

    f = 1 / np.sqrt(1 - (re * np.cos(el) / (re + h)) ** 2)

    return vertical_roti * (f / 1)