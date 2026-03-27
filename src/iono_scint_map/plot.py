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

import cartopy.crs as ccrs
import cartopy.feature as cfeatures
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import polars as pl
import ppigrf

from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter)
from scipy.spatial import ConvexHull, Delaunay
from shapely.geometry import Polygon
from shapelysmooth import chaikin_smooth
from typing import Tuple

from iono_scint_map.dataset import ScintillationMapDataset

__author__ = ['André Ricardo Fazanaro Martinon']
__copyright__ = 'Copyright 2026, National Institute for Space Research (INPE)'
__credits__ = ['Stephan Stephany', 'Eurico Rodrigues de Paula']
__license__ = 'AGPL-3.0-or-later'
__maintainer__ = 'André Ricardo Fazanaro Martinon'
__email__ = 'andre.martinon@inpe.br'
__status__ = 'Production'


line_style = dict(
    [('solid',               (0, ())),
     ('loosely dotted',      (0, (1, 10))),
     ('dotted',              (0, (1, 5))),
     ('densely dotted',      (0, (1, 1))),

     ('loosely dashed',      (0, (5, 10))),
     ('dashed',              (0, (5, 5))),
     ('densely dashed',      (0, (5, 1))),

     ('loosely dashdotted',  (0, (3, 10, 1, 10))),
     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])


def create_world_map(ax,
                     extent: Tuple[float, float, float, float]=(-180, 180,
                                                                -90, 90),
                     color='gray',
                     show_xlabel=True,
                     show_ylabel=True,
                     fontsize=8,
                     linewidth=0.5):
    ax.set_extent(extent)

    land = cfeatures.LAND.with_scale('50m')
    ax.add_feature(land, facecolor='none', edgecolor=color, linewidth=linewidth)

    countries = cfeatures.BORDERS.with_scale('50m')
    ax.add_feature(countries, edgecolor=color, linestyle='-', linewidth=linewidth)

    states = cfeatures.STATES.with_scale('50m')
    ax.add_feature(states, edgecolor=color, linestyle='-', linewidth=linewidth)

    gl = ax.gridlines(draw_labels=True, linewidth=1, color=color,
                      linestyle=line_style['dotted'])
    gl.top_labels = False
    gl.right_labels = False

    if show_xlabel:
        gl.xformatter = LongitudeFormatter(direction_label=False)
        gl.xlocator = mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10])
        gl.xlabel_style = {'size': fontsize, 'weight': 'normal'}
    else:
        gl.bottom_labels = False

    if show_ylabel:
        gl.yformatter = LatitudeFormatter(direction_label=False)
        gl.ylocator = mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10])
        gl.ylabel_style = {'size': fontsize, 'weight': 'normal'}
    else:
        gl.left_labels = False

    gl.xlines = False
    gl.ylines = False


def plot_igrf(ax, igrf_date, extent=(-180, 180, -90, 90),
              levels=(-30, -20, -10, 0, 10, 20, 30),
              line_widths=(1, 1.5, 1, 2, 1, 1.5, 1), step=1, color='dimgray',
              fontsize=8):
    def fmt_igrf_latitude(x):
        s = f'{x:.1f}'
        if s.endswith("0"):
            s = f"{x:.0f}"
        return rf"{s} ^{{\circ}}" if plt.rcParams["text.usetex"] else \
            rf"${s}^{{\mathbf{{\circ}}}}$"

    lon_min, lon_max, lat_min, lat_max = extent

    lat = np.arange(lat_min, lat_max + step, step, dtype='float')
    lon = np.arange(lon_min, lon_max + step, step, dtype='float')

    grid_lon, grid_lat = np.meshgrid(lon, lat)

    Be, Bn, Bu = ppigrf.igrf(grid_lon,
                             grid_lat,
                             h=0,
                             date=igrf_date)
    inclination, declination = ppigrf.get_inclination_declination(Be, Bn, Bu)
    table = np.rad2deg(np.arctan(np.tan(np.deg2rad(inclination[0])) * 0.5))

    contour = ax.contour(grid_lon,
                         grid_lat,
                         table,
                         levels=levels,
                         # colors=['blue', 'blue', 'blue', 'green', 'red',
                         #         'red', 'red'],
                         colors=color,
                         alpha=1,
                         linewidths=line_widths,
                         zorder=2)
    ax.clabel(contour, inline=True, inline_spacing=4, fontsize=fontsize,
              fmt=fmt_igrf_latitude)


def plot_scintillation_map(ax,
                           scint_map_data: ScintillationMapDataset,
                           clipping: bool = False,
                           show_convex_hull: bool = False):
    lon_min, lon_max, lat_min, lat_max = scint_map_data.map_extent
    lat = np.arange(lat_min,
                    lat_max + scint_map_data.interpolation_grid_resolution,
                    scint_map_data.interpolation_grid_resolution)
    lon = np.arange(lon_min,
                    lon_max + scint_map_data.interpolation_grid_resolution,
                    scint_map_data.interpolation_grid_resolution)
    grid_lon, grid_lat = np.meshgrid(lon, lat)

    shape = scint_map_data.interpolated_map.shape
    scint_map = np.flip(scint_map_data.interpolated_map, axis=0).ravel()

    if clipping:
        points = scint_map_data.scint_data.select(['lat', 'lon']).to_numpy()
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]

        perimeter = cv2.arcLength(hull_points, True)
        epsilon = 0.01 * perimeter

        hull_points = np.squeeze(cv2.approxPolyDP(hull_points,
                                                  epsilon,
                                                  True))

        expansion_factor = 1.1
        centroid = np.mean(hull_points, axis=0)
        expanded_points = centroid + (hull_points - centroid) * expansion_factor
        hull = ConvexHull(expanded_points)

        vertices = np.column_stack((expanded_points[hull.vertices][:,1],
                                    expanded_points[hull.vertices][:,0]))
        polygon = Polygon(vertices)
        smooth_polygon = chaikin_smooth(polygon)
        x, y = smooth_polygon.exterior.xy

        smooth_points = np.column_stack((y, x))
        smooth_hull = ConvexHull(smooth_points)

        grid_points = np.vstack([grid_lat.ravel(), grid_lon.ravel()]).T
        delaunay = Delaunay(smooth_points[smooth_hull.vertices])
        is_outside = delaunay.find_simplex(grid_points) < 0
        scint_map[is_outside] = np.nan

        if show_convex_hull:
            for simplex in smooth_hull.simplices:
                ax.plot(smooth_points[simplex, 1],
                        smooth_points[simplex, 0],
                        c='red',
                        zorder=1006)

    scint_map = scint_map.reshape(shape)

    cmap = mpl.colormaps.get_cmap("jet").copy()
    cmap.set_under(alpha=0)
    cmap.set_bad(alpha=0)

    map = ax.pcolormesh(grid_lon,
                        grid_lat,
                        scint_map,
                        vmin=scint_map_data.scint_index.limits['min'],
                        vmax=scint_map_data.scint_index.limits['max'],
                        cmap=cmap,
                        alpha=1,
                        zorder=-2,
                        edgecolors='none',
                        antialiased=True,
                        shading='gouraud')
    return map


def plot_scintillation_map_axis(ax,
                                scint_map_data: ScintillationMapDataset,
                                clipping: bool = False,
                                show_convex_hull: bool = False):
    fig = ax.get_figure()

    map = plot_scintillation_map(ax, scint_map_data, clipping, show_convex_hull)

    ax.set_xticks([-80, -70, -60, -50, -40, -30], [],
                  crs=ccrs.PlateCarree())
    ax.axes.xaxis.set_ticklabels([])
    ax.set_yticks([-40, -30, -20, -10, 0, 10], [],
                  crs=ccrs.PlateCarree())
    ax.axes.yaxis.set_ticklabels([])

    ax.set_aspect(1)

    ax.text(0.5, -0.054, 'Geographic Longitude', transform=ax.transAxes,
            ha='center', va='top', fontsize=18)
    ax.text(-0.12, 0.5, 'Geographic Latitude', transform=ax.transAxes,
            rotation='vertical', va='center', fontsize=18)
    fig.text(0.97,
             0.022,
             f"[{np.nanmin(scint_map_data.interpolated_map):.2f}, "
             f"{np.nanmax(scint_map_data.interpolated_map):.2f}] "
             f"{scint_map_data.interpolation.value.upper()}"
             f"({scint_map_data.preprocessing.value.upper()})",
             ha='right',
             va='bottom',
             fontsize=18)
    fig.suptitle(f"{scint_map_data.end_timestamp} UT", fontsize=24)

    if scint_map_data.scint_index.index == 'roti':
        ticks = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        label = 'ROTI (TECU/s)'
    elif scint_map_data.scint_index.index == 'phi60':
        ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
        label = r'$\sigma_\phi$ (rad)'
    else:
        ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
        label = f'{scint_map_data.scint_index.index.upper()}'

    cb = fig.colorbar(map, location='right', ticks=ticks, shrink=1, pad=0.02)
    cb.set_label(label=label, size=18)

    map.figure.axes[0].tick_params(axis="both", labelsize=18)
    map.figure.axes[1].tick_params(axis="y", labelsize=18)

    fig.subplots_adjust(top=0.93, bottom=0.09, left=0.11, right=1,
                        hspace=0.0, wspace=0.0)


def plot_scintillation_map_no_axis(ax,
                                   scint_map_data: ScintillationMapDataset,
                                   clipping: bool = False,
                                   show_convex_hull: bool = False):
    fig = ax.get_figure()
    ax.axis('off')

    map = plot_scintillation_map(ax, scint_map_data, clipping, show_convex_hull)

    fig.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)


def plot_ipp_map_axis(ax,
                 scint_map_data: ScintillationMapDataset,
                 size: int = 64,
                 agg: bool = True):
    fig = ax.get_figure()
    scint_index = scint_map_data.scint_index.value

    if agg:
        data = scint_map_data.grouped
        method = f"{scint_map_data.preprocessing.value.upper()}"
        projection = scint_map_data.preprocessing.projection.capitalize()
    else:
        data = scint_map_data.scint_data
        method = ''
        projection = 'Slant'


    cmap = mpl.colormaps.get_cmap("jet").copy()
    cmap.set_under('w', alpha=0)
    cmap.set_bad(alpha=0)

    map = ax.scatter(data['lon'],
                     data['lat'],
                     c=data[scint_index],
                     vmin=scint_map_data.scint_index.limits['min'],
                     vmax=scint_map_data.scint_index.limits['max'],
                     marker='s',
                     s=size,
                     cmap=cmap)

    ax.set_xticks([-80, -70, -60, -50, -40, -30], [],
                  crs=ccrs.PlateCarree())
    ax.axes.xaxis.set_ticklabels([])
    ax.set_yticks([-40, -30, -20, -10, 0, 10], [],
                  crs=ccrs.PlateCarree())
    ax.axes.yaxis.set_ticklabels([])

    ax.set_aspect(1)

    ax.text(0.5, -0.054, 'Geographic Longitude', transform=ax.transAxes,
            ha='center', va='top', fontsize=18)
    ax.text(-0.12, 0.5, 'Geographic Latitude', transform=ax.transAxes,
            rotation='vertical', va='center', fontsize=18)
    fig.text(0.97,
             0.022,
             f"[{np.nanmin(data[scint_index]):.2f}, "
             f"{np.nanmax(data[scint_index]):.2f}] {method}",
             ha='right',
             va='bottom',
             fontsize=18)
    fig.suptitle(f"{scint_map_data.end_timestamp} UT", fontsize=24)

    if scint_map_data.scint_index.index == 'roti':
        ticks = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        label = 'ROTI (TECU/s)'
    elif scint_map_data.scint_index.index == 'phi60':
        ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
        label = r'$\sigma_\phi$ (rad)'
    else:
        ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
        label = f'{scint_map_data.scint_index.index.upper()}'

    cb = fig.colorbar(map, location='right', ticks=ticks, shrink=1, pad=0.02)
    cb.set_label(label=rf'{projection} {label}', size=18)

    map.figure.axes[0].tick_params(axis="both", labelsize=18)
    map.figure.axes[1].tick_params(axis="y", labelsize=18)

    fig.subplots_adjust(top=0.93, bottom=0.09, left=0.11, right=1,
                        hspace=0.0, wspace=0.0)

def plot_ipp_map_no_axis(ax,
                 scint_map_data: ScintillationMapDataset,
                 size: int = 64,
                 agg: bool = True):
    fig = ax.get_figure()
    ax.axis('off')

    scint_index = scint_map_data.scint_index.value

    if agg:
        data = scint_map_data.grouped
    else:
        data = scint_map_data.scint_data

    cmap = mpl.colormaps.get_cmap("jet").copy()
    cmap.set_under('w', alpha=0)
    cmap.set_bad(alpha=0)
    ax.set_aspect(1)
    map = ax.scatter(data['lon'],
                     data['lat'],
                     c=data[scint_index],
                     vmin=scint_map_data.scint_index.limits['min'],
                     vmax=scint_map_data.scint_index.limits['max'],
                     marker='s',
                     s=size,
                     cmap=cmap)


    fig.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)


def plot_gnss_stations(ax,
                       scint_map_data: ScintillationMapDataset):
    for name in sorted(scint_map_data.station_data['name'].unique().to_list()):
        station = scint_map_data.station_data.filter(pl.col('name') == name)
        ax.scatter(station['lon'],
                   station['lat'],
                   s=121,
                   c='lightgray',
                   edgecolor='black',
                   marker='v',
                   zorder=1006)