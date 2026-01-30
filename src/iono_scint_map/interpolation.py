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
import math
import numpy as np

from pprint import pprint
from numba import njit
from scipy.interpolate import griddata, Rbf
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.special import gamma, kv
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (ConstantKernel,
                                              WhiteKernel,
                                              StationaryKernelMixin,
                                              NormalizedKernelMixin,
                                              Kernel,
                                              Hyperparameter,
                                              _check_length_scale,
                                              _approx_fprime)
from typing import Iterable


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

    @staticmethod
    def haversine_distance(xa, xb, r=6335.439):
        lon_1 = xa[:, 0]*np.pi/180
        lat_1 = xa[:, 1]*np.pi/180
        lon_2 = xb[:, 0]*np.pi/180
        lat_2 = xb[:, 1]*np.pi/180

        return Interpolation._calc_haversine_distance(lon_1, lat_1,
                                                      lon_2, lat_2, r)

    @staticmethod
    @njit
    def _calc_haversine_distance(lon_1, lat_1, lon_2, lat_2, r):
        d_lon = np.abs(lon_1 - lon_2)

        a = np.power(np.cos(lat_2)*np.sin(d_lon), 2)
        b = np.power(np.cos(lat_1)*np.sin(lat_2) -
                     np.sin(lat_1)*np.cos(lat_2)*np.cos(d_lon), 2)
        c = (np.sin(lat_1)*np.sin(lat_2) +
             np.cos(lat_1)*np.cos(lat_2)*np.cos(d_lon))

        return np.abs(r*np.arctan(np.sqrt(a + b) / c))

    @staticmethod
    def cdist(xa, xb):
        xb_lon, xa_lon = np.meshgrid(xb[:, 0], xa[:, 0])
        xb_lat, xa_lat = np.meshgrid(xb[:, 1], xa[:, 1])

        xa = np.c_[xa_lon.ravel(), xa_lat.ravel()]
        xb = np.c_[xb_lon.ravel(), xb_lat.ravel()]

        dists = Interpolation.haversine_distance(xa, xb)
        return dists.reshape(xb_lon.shape)

    @staticmethod
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
        return Interpolation.haversine_distance(xa, xb)


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

        distances = Interpolation.cdist(self.X_train, self.X)
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

        return Interpolation._calc_haversine_distance(lon_1, lat_1,
                                                      lon_2, lat_2, r)

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
                      ModifiedRationalQuadratic(length_scale=50, alpha=1.5))
                      # ModifiedRationalQuadratic(length_scale=1.0,
                      #                           alpha=1.5) +
                      # ModifiedMatern(length_scale=1.0, nu=2.5) +
                      # WhiteKernel(noise_level=0.01))

        self.gpr = GaussianProcessRegressor(kernel=kernel,
                                            alpha=noise**2,
                                            n_restarts_optimizer=10,
                                            normalize_y=False)
        with Benchmark('GPR FIT'):
            self.gpr.fit(self.X_train, self.Y_train)

        print('Kernel parameters:')
        pprint(self.gpr.kernel_.get_params(deep=True))
        print('\nGaussianProcessRegressor parameters:')
        pprint(self.gpr.get_params(deep=True))

        with Benchmark('GPR PREDICT'):
            self.Y = self.gpr.predict(self.X, return_cov=False)


class ModifiedRBF(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """ Adapted from sklearn.gaussian_process.kernels.py (sklearn v 0.19.2)
    Radial-basis function kernel (aka squared-exponential kernel).
    The RBF kernel is a stationary kernel. It is also known as the
    "squared exponential" kernel. It is parameterized by a length-scale
    parameter length_scale>0, which can either be a scalar (isotropic variant
    of the kernel) or a vector with the same number of dimensions as the inputs
    X (anisotropic variant of the kernel). The kernel is given by:
    k(x_i, x_j) = exp(- d(x_i, x_j)^2 / (2 * length_scale^2))
    This kernel is infinitely differentiable, which implies that GPs with this
    kernel as covariance function have mean square derivatives of all orders,
    and are thus very smooth.
    .. versionadded:: 0.18
    Parameters
    -----------
    length_scale : float or array with shape (n_features,), default: 1.0
        The length scale of the kernel. If a float, an isotropic kernel is
        used. If an array, an anisotropic kernel is used where each dimension
        of l defines the length-scale of the respective feature dimension.
    length_scale_bounds : pair of floats >= 0, default: (1e-5, 1e5)
        The lower and upper bound on length_scale
    """

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic:
            return Hyperparameter("length_scale", "numeric",
                                  self.length_scale_bounds,
                                  len(self.length_scale))
        return Hyperparameter(
            "length_scale", "numeric", self.length_scale_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.
        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)
        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.
        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.
        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)
        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        X = np.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)
        if Y is None:
            dists = Interpolation.pdist(X) / np.power(length_scale, 2)
            K = np.exp(-.5 * dists)
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            dists = Interpolation.cdist(X, Y) / np.power(length_scale, 2)
            K = np.exp(-.5 * dists)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, np.empty((X.shape[0], X.shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                K_gradient = \
                    (K * squareform(dists))[:, :, np.newaxis]
                return K, K_gradient
            elif self.anisotropic:
                # We need to recompute the pairwise dimension-wise distances
                K_gradient = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 \
                             / (length_scale ** 2)
                K_gradient *= K[..., np.newaxis]
                return K, K_gradient
        else:
            return K

    def __repr__(self):
        if self.anisotropic:
            return "{0}(length_scale=[{1}])".format(
                self.__class__.__name__, ", ".join(map("{0:.3g}".format,
                                                       self.length_scale)))
        else:  # isotropic
            return "{0}(length_scale={1:.3g})".format(
                self.__class__.__name__, np.ravel(self.length_scale)[0])


class ModifiedMatern(ModifiedRBF):
    """Matern kernel.
    The class of Matern kernels is a generalization of the :class:`RBF`.
    It has an additional parameter :math:`\\nu` which controls the
    smoothness of the resulting function. The smaller :math:`\\nu`,
    the less smooth the approximated function is.
    As :math:`\\nu\\rightarrow\\infty`, the kernel becomes equivalent to
    the :class:`RBF` kernel. When :math:`\\nu = 1/2`, the Matérn kernel
    becomes identical to the absolute exponential kernel.
    Important intermediate values are
    :math:`\\nu=1.5` (once differentiable functions)
    and :math:`\\nu=2.5` (twice differentiable functions).
    The kernel is given by:
    .. math::
         k(x_i, x_j) =  \\frac{1}{\\Gamma(\\nu)2^{\\nu-1}}\\Bigg(
         \\frac{\\sqrt{2\\nu}}{l} d(x_i , x_j )
         \\Bigg)^\\nu K_\\nu\\Bigg(
         \\frac{\\sqrt{2\\nu}}{l} d(x_i , x_j )\\Bigg)
    where :math:`d(\\cdot,\\cdot)` is the Euclidean distance,
    :math:`K_{\\nu}(\\cdot)` is a modified Bessel function and
    :math:`\\Gamma(\\cdot)` is the gamma function.
    See [1]_, Chapter 4, Section 4.2, for details regarding the different
    variants of the Matern kernel.
    Read more in the :ref:`User Guide <gp_kernels>`.
    .. versionadded:: 0.18
    Parameters
    ----------
    length_scale : float or ndarray of shape (n_features,), default=1.0
        The length scale of the kernel. If a float, an isotropic kernel is
        used. If an array, an anisotropic kernel is used where each dimension
        of l defines the length-scale of the respective feature dimension.
    length_scale_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'length_scale'.
        If set to "fixed", 'length_scale' cannot be changed during
        hyperparameter tuning.
    nu : float, default=1.5
        The parameter nu controlling the smoothness of the learned function.
        The smaller nu, the less smooth the approximated function is.
        For nu=inf, the kernel becomes equivalent to the RBF kernel and for
        nu=0.5 to the absolute exponential kernel. Important intermediate
        values are nu=1.5 (once differentiable functions) and nu=2.5
        (twice differentiable functions). Note that values of nu not in
        [0.5, 1.5, 2.5, inf] incur a considerably higher computational cost
        (appr. 10 times higher) since they require to evaluate the modified
        Bessel function. Furthermore, in contrast to l, nu is kept fixed to
        its initial value and not optimized.
    References
    ----------
    .. [1] `Carl Edward Rasmussen, Christopher K. I. Williams (2006).
        "Gaussian Processes for Machine Learning". The MIT Press.
        <http://www.gaussianprocess.org/gpml/>`_
    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.gaussian_process import GaussianProcessClassifier
    >>> from sklearn.gaussian_process.kernels import Matern
    >>> X, y = load_iris(return_X_y=True)
    >>> kernel = 1.0 * Matern(length_scale=1.0, nu=1.5)
    >>> gpc = GaussianProcessClassifier(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpc.score(X, y)
    0.9866...
    >>> gpc.predict_proba(X[:2,:])
    array([[0.8513..., 0.0368..., 0.1117...],
            [0.8086..., 0.0693..., 0.1220...]])
    """

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5),
                 nu=1.5):
        super().__init__(length_scale, length_scale_bounds)
        self.nu = nu

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.
        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)
        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.
        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.
        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)
        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims), \
                optional
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when `eval_gradient`
            is True.
        """
        X = np.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)
        if Y is None:
            dists = Interpolation.pdist(X / length_scale)
            # print('call pdist', X.shape)
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            dists = Interpolation.cdist(X / length_scale, Y / length_scale)
            # print('call cdist', X.shape, Y.shape)
            # print(dists)

        if self.nu == 0.5:
            K = np.exp(-dists)
        elif self.nu == 1.5:
            K = dists * math.sqrt(3)
            K = (1.0 + K) * np.exp(-K)
        elif self.nu == 2.5:
            K = dists * math.sqrt(5)
            K = (1.0 + K + K ** 2 / 3.0) * np.exp(-K)
        elif self.nu == np.inf:
            K = np.exp(-(dists ** 2) / 2.0)
        else:  # general case; expensive to evaluate
            K = dists
            K[K == 0.0] += np.finfo(float).eps  # strict zeros result in nan
            tmp = math.sqrt(2 * self.nu) * K
            K.fill((2 ** (1.0 - self.nu)) / gamma(self.nu))
            K *= tmp ** self.nu
            K *= kv(self.nu, tmp)

        if Y is None:
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                K_gradient = np.empty((X.shape[0], X.shape[0], 0))
                return K, K_gradient

            # We need to recompute the pairwise dimension-wise distances
            if self.anisotropic:
                D = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 / (
                        length_scale ** 2
                )
            else:
                D = squareform(dists ** 2)[:, :, np.newaxis]

            if self.nu == 0.5:
                denominator = np.sqrt(D.sum(axis=2))[:, :, np.newaxis]
                K_gradient = K[..., np.newaxis] * np.divide(
                    D, denominator, where=denominator != 0
                )
            elif self.nu == 1.5:
                K_gradient = 3 * D * np.exp(-np.sqrt(3 * D.sum(-1)))[
                    ..., np.newaxis]
            elif self.nu == 2.5:
                tmp = np.sqrt(5 * D.sum(-1))[..., np.newaxis]
                K_gradient = 5.0 / 3.0 * D * (tmp + 1) * np.exp(-tmp)
            elif self.nu == np.inf:
                K_gradient = D * K[..., np.newaxis]
            else:
                # approximate gradient numerically
                def f(theta):  # helper function
                    return self.clone_with_theta(theta)(X, Y)

                return K, _approx_fprime(self.theta, f, 1e-10)

            if not self.anisotropic:
                return K, K_gradient[:, :].sum(-1)[:, :, np.newaxis]
            else:
                return K, K_gradient
        else:
            return K

    def __repr__(self):
        if self.anisotropic:
            return "{0}(length_scale=[{1}], nu={2:.3g})".format(
                self.__class__.__name__,
                ", ".join(map("{0:.3g}".format, self.length_scale)),
                self.nu,
            )
        else:
            return "{0}(length_scale={1:.3g}, nu={2:.3g})".format(
                self.__class__.__name__, np.ravel(self.length_scale)[0], self.nu
            )


class ModifiedRationalQuadratic(StationaryKernelMixin, NormalizedKernelMixin,
                                Kernel):
    """Rational Quadratic kernel.
    The RationalQuadratic kernel can be seen as a scale mixture (an infinite
    sum) of RBF kernels with different characteristic length scales. It is
    parameterized by a length scale parameter :math:`l>0` and a scale
    mixture parameter :math:`\\alpha>0`. Only the isotropic variant
    where length_scale :math:`l` is a scalar is supported at the moment.
    The kernel is given by:
    .. math::
        k(x_i, x_j) = \\left(
        1 + \\frac{d(x_i, x_j)^2 }{ 2\\alpha  l^2}\\right)^{-\\alpha}
    where :math:`\\alpha` is the scale mixture parameter, :math:`l` is
    the length scale of the kernel and :math:`d(\\cdot,\\cdot)` is the
    Euclidean distance.
    For advice on how to set the parameters, see e.g. [1]_.
    Read more in the :ref:`User Guide <gp_kernels>`.
    .. versionadded:: 0.18
    Parameters
    ----------
    length_scale : float > 0, default=1.0
        The length scale of the kernel.
    alpha : float > 0, default=1.0
        Scale mixture parameter
    length_scale_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'length_scale'.
        If set to "fixed", 'length_scale' cannot be changed during
        hyperparameter tuning.
    alpha_bounds : pair of floats >= 0 or "fixed", default=(1e-5, 1e5)
        The lower and upper bound on 'alpha'.
        If set to "fixed", 'alpha' cannot be changed during
        hyperparameter tuning.
    References
    ----------
    .. [1] `David Duvenaud (2014). "The Kernel Cookbook:
        Advice on Covariance functions".
        <https://www.cs.toronto.edu/~duvenaud/cookbook/>`_
    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.gaussian_process import GaussianProcessClassifier
    >>> from sklearn.gaussian_process.kernels import RationalQuadratic
    >>> X, y = load_iris(return_X_y=True)
    >>> kernel = RationalQuadratic(length_scale=1.0, alpha=1.5)
    >>> gpc = GaussianProcessClassifier(kernel=kernel,
    ...         random_state=0).fit(X, y)
    >>> gpc.score(X, y)
    0.9733...
    >>> gpc.predict_proba(X[:2,:])
    array([[0.8881..., 0.0566..., 0.05518...],
            [0.8678..., 0.0707... , 0.0614...]])
    """

    def __init__(
            self,
            length_scale=1.0,
            alpha=1.0,
            length_scale_bounds=(1e-5, 1e5),
            alpha_bounds=(1e-5, 1e5)
    ):
        self.length_scale = length_scale
        self.alpha = alpha
        self.length_scale_bounds = length_scale_bounds
        self.alpha_bounds = alpha_bounds

    @property
    def hyperparameter_length_scale(self):
        return Hyperparameter("length_scale", "numeric",
                              self.length_scale_bounds)

    @property
    def hyperparameter_alpha(self):
        return Hyperparameter("alpha", "numeric", self.alpha_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.
        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)
        Y : ndarray of shape (n_samples_Y, n_features), default=None
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.
        eval_gradient : bool, default=False
            Determines whether the gradient with respect to the log of
            the kernel hyperparameter is computed.
            Only supported when Y is None.
        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)
        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the log of the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        if len(np.atleast_1d(self.length_scale)) > 1:
            raise AttributeError(
                "RationalQuadratic kernel only supports isotropic version, "
                "please use a single scalar for length_scale"
            )
        X = np.atleast_2d(X)
        if Y is None:
            dists = squareform(Interpolation.pdist(X))
            tmp = dists / (2 * self.alpha * self.length_scale ** 2)
            base = 1 + tmp
            K = base ** -self.alpha
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            dists = Interpolation.cdist(X, Y)
            K = (1 + dists / (
                        2 * self.alpha * self.length_scale ** 2)) ** -self.alpha

        if eval_gradient:
            # gradient with respect to length_scale
            if not self.hyperparameter_length_scale.fixed:
                length_scale_gradient = dists * K / (
                            self.length_scale ** 2 * base)
                length_scale_gradient = length_scale_gradient[:, :, np.newaxis]
            else:  # l is kept fixed
                length_scale_gradient = np.empty((K.shape[0], K.shape[1], 0))

            # gradient with respect to alpha
            if not self.hyperparameter_alpha.fixed:
                alpha_gradient = K * (
                        -self.alpha * np.log(base)
                        + dists / (2 * self.length_scale ** 2 * base)
                )
                alpha_gradient = alpha_gradient[:, :, np.newaxis]
            else:  # alpha is kept fixed
                alpha_gradient = np.empty((K.shape[0], K.shape[1], 0))

            return K, np.dstack((alpha_gradient, length_scale_gradient))
        else:
            return K

    def __repr__(self):
        return "{0}(alpha={1:.3g}, length_scale={2:.3g})".format(
            self.__class__.__name__, self.alpha, self.length_scale
        )


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


if __name__ == '__main__':
    known_lat = [-36.0, -36.0, -36.0, -36.0]
    known_lon = [-63.0, -59.0, -58.0, -73.0]
    unknown_lat = [-24.0, -24.0, -24.0, -23.0, -22.0]
    unknown_lon = [-46.0, -44.0, -43.0, -52.0, -51.0]

    known = np.stack((known_lon, known_lat), axis=-1)
    unknown = np.stack((unknown_lon, unknown_lat), axis=-1)

    dist = cdist(known, unknown,
                 metric=RadialBasisFunctionInterpolation.haversine_distance)
    print(dist)

    dist = Interpolation.cdist(known, unknown)
    print(dist)

    dist = pdist(known,
                 metric=RadialBasisFunctionInterpolation.haversine_distance)
    print(dist)

    dist = Interpolation.pdist(known)
    print(dist)