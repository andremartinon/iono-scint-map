# IONO_SCINT_MAP - An ionospheric scintillation map generation toolkit
# Copyright (C) 2026  André Ricardo Fazanaro Martinon, Stephan Stephany, and
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

from iono_scint_map.constant import EARTH_RADIUS


def calc_haversine_distance(lon_1, lat_1, lon_2, lat_2, r):
    d_lon = np.abs(lon_1 - lon_2)

    a = np.power(np.cos(lat_2) * np.sin(d_lon), 2)
    b = np.power(np.cos(lat_1) * np.sin(lat_2) -
                 np.sin(lat_1) * np.cos(lat_2) * np.cos(d_lon), 2)
    c = (np.sin(lat_1) * np.sin(lat_2) +
         np.cos(lat_1) * np.cos(lat_2) * np.cos(d_lon))

    return np.abs(r * np.arctan(np.sqrt(a + b) / c))


def haversine_distance(xa, xb, r=EARTH_RADIUS):
    lon_1 = xa[:, 0]*np.pi/180
    lat_1 = xa[:, 1]*np.pi/180
    lon_2 = xb[:, 0]*np.pi/180
    lat_2 = xb[:, 1]*np.pi/180

    return calc_haversine_distance(lon_1, lat_1, lon_2, lat_2, r)


def cdist(xa, xb):
    xb_lon, xa_lon = np.meshgrid(xb[:, 0], xa[:, 0])
    xb_lat, xa_lat = np.meshgrid(xb[:, 1], xa[:, 1])

    xa = np.c_[xa_lon.ravel(), xa_lat.ravel()]
    xb = np.c_[xb_lon.ravel(), xb_lat.ravel()]

    dists = haversine_distance(xa, xb)
    return dists.reshape(xb_lon.shape)


def pdist(x):
    def nump2(n, k):
        a = np.ones((k, n - k + 1), dtype=int)
        a[0] = np.arange(n - k + 1)
        for j in range(1, k):
            reps = (n - k + j) - a[j - 1]
            a = np.repeat(a, reps, axis=1)
            ind = np.add.accumulate(reps)
            a[j, ind[:-1]] = 1 - reps[1:]
            a[j, 0] = j
            a[j] = np.add.accumulate(a[j])
        return a

    def get_arrays(x):
        index = nump2(len(x), 2)
        a_index = index[0, :]
        b_index = index[1, :]

        return x[a_index], x[b_index]

    xa, xb = get_arrays(x)
    return haversine_distance(xa, xb)


if __name__ == '__main__':

    print(calc_haversine_distance(-46.6333*np.pi/180,
                                  -23.5505*np.pi/180,
                                  -43.1729*np.pi/180,
                                  -22.9068*np.pi/180,
                                  r=6335.439))
    print(calc_haversine_distance(-46.6333*np.pi/180,
                                  -23.5505*np.pi/180,
                                  -43.1729*np.pi/180,
                                  -22.9068*np.pi/180,
                                  r=EARTH_RADIUS))
    print(calc_haversine_distance(np.full(500000000, -46.0)*np.pi/180,
                                  np.full(500000000, -23.0)*np.pi/180,
                                  np.full(500000000, -43.2)*np.pi/180,
                                  np.full(500000000, -22.9)*np.pi/180,
                                  r=EARTH_RADIUS))

