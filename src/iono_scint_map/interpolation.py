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
import enum
import numpy as np

import iono_scint_map.haversine_metric as hm

from pprint import pprint
from scipy.interpolate import griddata, Rbf
from scipy.spatial.distance import cdist, pdist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel
from typing import Iterable

from iono_scint_map.kernels import ModifiedRationalQuadratic
from iono_scint_map.util import Benchmark


class Interpolation:

    def __init__(self, extent: Iterable[float], step: float = 1):
        self.lon_min, self.lon_max, self.lat_min, self.lat_max = extent
        self.step = step

        longitudes = np.arange(self.lon_min, self.lon_max+self.step, self.step)
        latitudes = np.arange(self.lat_min, self.lat_max+self.step, self.step)

        grid_lon, grid_lat = np.meshgrid(longitudes, latitudes)
        self._shape = grid_lon.shape

        self.X = np.c_[grid_lon.ravel(), grid_lat.ravel()]
        self.Y = None
        self.X_train = None
        self.Y_train = None

        self._interpolated_map = None

    def interpolate(self,
                    known_longitudes: Iterable[float],
                    known_latitudes: Iterable[float],
                    known_values: Iterable[float],
                    **kwargs):
        self.X_train = np.stack((known_longitudes, known_latitudes), axis=-1)
        self.Y_train = known_values

    @property
    def shape(self):
        return self._shape

    @property
    def interpolated_map(self) -> np.ndarray:
        return np.flip(self.Y.reshape(self.shape), axis=0)


class GriddataInterpolation(Interpolation):

    def interpolate(self,
                    known_longitudes: Iterable[float],
                    known_latitudes: Iterable[float],
                    known_values: Iterable[float], **kwargs):
        super().interpolate(known_longitudes, known_latitudes, known_values)
        self.Y = griddata(self.X_train, self.Y_train, self.X, method='cubic')


class InverseDistanceWeightingInterpolation(Interpolation):
    def interpolate(self,
                    known_longitudes: Iterable[float],
                    known_latitudes: Iterable[float],
                    known_values: Iterable[float],
                    r=990,
                    reg=1e-20,
                    p=2):
        super().interpolate(known_longitudes, known_latitudes, known_values)

        distances = hm.cdist(self.X_train, self.X)
        weights = (np.maximum(r - distances, 0) / (r * distances + reg)) ** p
        weights /= weights.sum(axis=0) + reg

        self.Y = np.dot(weights.T, self.Y_train)


class RadialBasisFunctionInterpolation(Interpolation):
    @staticmethod
    def haversine_distance(xa, xb, r=6335.439):
        lon_1 = xa[0]*np.pi/180
        lat_1 = xa[1]*np.pi/180
        lon_2 = xb[0]*np.pi/180
        lat_2 = xb[1]*np.pi/180

        return hm.calc_haversine_distance(lon_1, lat_1, lon_2, lat_2, r)

    def interpolate(self,
                    known_longitudes: Iterable[float],
                    known_latitudes: Iterable[float],
                    known_values: Iterable[float],
                    function='thin_plate',
                    smooth=-0.1,
                    epsilon=None):

        super().interpolate(known_longitudes, known_latitudes, known_values)

        if not epsilon:
            distances = cdist(self.X_train, self.X, metric='euclidean')
            epsilon = (int(np.mean(distances)) / 10 + 1) * 110

        f = Rbf(self.X_train[:, 0],
                self.X_train[:, 1],
                self.Y_train,
                function=function,
                epsilon=epsilon,
                smooth=-0.1,
                norm=RadialBasisFunctionInterpolation.haversine_distance)

        self.Y = f(self.X[:, 0], self.X[:, 1])


class GaussianProcessInterpolation(Interpolation):
    def interpolate(self,
                    known_longitudes: Iterable[float],
                    known_latitudes: Iterable[float],
                    known_values: Iterable[float],
                    kernel=None,
                    noise=1e-10):

        super().interpolate(known_longitudes, known_latitudes, known_values)

        if not kernel:
            kernel = (ConstantKernel(1.0) *
                      ModifiedRationalQuadratic(length_scale=50, alpha=1.5) +
                      # ModifiedRationalQuadratic(length_scale=1.0,
                      #                           alpha=1.5) +
                      # ModifiedMatern(length_scale=1.0, nu=2.5) +
                      WhiteKernel())

        gpr = GaussianProcessRegressor(kernel=kernel,
                                       alpha=noise**2,
                                       n_restarts_optimizer=10,
                                       normalize_y=False)
        with Benchmark('GPR FIT'):
            gpr.fit(self.X_train, self.Y_train)

        print('Kernel parameters:')
        pprint(gpr.kernel_.get_params(deep=True))
        print('\nGaussianProcessRegressor parameters:')
        pprint(gpr.get_params(deep=True))

        with Benchmark('GPR PREDICT'):
            self.Y, std = gpr.predict(self.X, return_std=True, return_cov=False)
            print(std)
            print(std.shape)
            print(np.min(std), np.max(std))


class InterpolationOptions(enum.Enum):
    GDA = ('gda', GriddataInterpolation)
    GPR = ('gpr', GaussianProcessInterpolation)
    IDW = ('idw', InverseDistanceWeightingInterpolation)
    RBF = ('rbf', RadialBasisFunctionInterpolation)

    def __new__(cls, value, interpolation_class):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.interpolation_class = interpolation_class

        return obj
