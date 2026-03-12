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
import cartopy.crs as ccrs
import click
import datetime

from gettext import gettext as _
from matplotlib.figure import Figure
from pathlib import Path

from rich.console import Console
from rich.padding import Padding
from rich.table import Table
from rich.text import Text
from typing import Tuple, Iterable, List, Optional

from iono_scint_map import __version__
from iono_scint_map.constant import CONSOLE_WIDTH
from iono_scint_map.dataset import (Constellation, PreprocessingOptions,
                                    ScintillationIndex, ScintillationMapDataset)
from iono_scint_map.interpolation import InterpolationOptions
from iono_scint_map.plot import (create_world_map, plot_igrf,
                                 plot_scintillation_map_axis,
                                 plot_scintillation_map_no_axis,
                                 plot_gnss_stations,
                                 plot_ipp_map)
from iono_scint_map.pipeline import (DatasetProcessingPipeline,
                                     DataCleaningAndFiltering, IPPProjection,
                                     ScintIndexProjection, IPPGrouping,
                                     IPPAggregation, MapInterpolation)


def print_create_map_config(scint_dataset: ScintillationMapDataset,
                            scint_data_file: Path,
                            station_data_file: Path,
                            output_file: Path):

    console = Console(width=CONSOLE_WIDTH)

    table = Table(title=_('Scintillation Map Generation Parameters'),
                  title_style='bold italic green', row_styles=['dim', ''])

    table.add_column(_('Parameter'), justify="left", style="bright_green")

    table.add_column(_("Value"), justify="left", style="cyan", overflow='fold',
                     highlight=True)
    table.add_row(*(
        _('Scintillation index'),
        scint_dataset.scint_index.index.upper()
    ))

    table.add_row(*(
        _('Scintillation index type'),
        scint_dataset.scint_index.type.upper()
    ))

    msg = _('min: %(min_value).2f, max: %(max_value).2f')
    table.add_row(*(
        _('Scintillation index limits'),
        msg % {'min_value': scint_dataset.scint_index.limits["min"],
               'max_value': scint_dataset.scint_index.limits["max"]}
    ))

    table.add_row(*(
        _('Interpolated map extents'),
        f'[{"°, ".join(map(str, scint_dataset.map_extent))}°]'
    ))

    table.add_row(*(
        _('Elevation limit'),
        f'{scint_dataset.elevation}°'
    ))

    table.add_row(*(
        _('Constellations'),
        f"[{', '.join([c.value for c in scint_dataset.constellations])}]"
    ))

    table.add_row(*(
        _('Preprocessing options'),
        scint_dataset.preprocessing.value.upper()
    ))

    table.add_row(*(
        _('Interpolation method'),
        scint_dataset.interpolation.value.upper()
    ))

    table.add_row(*(
        _('Interpolation grid resolution'),
        f'{scint_dataset.interpolation_grid_resolution}° x '
        f'{scint_dataset.interpolation_grid_resolution}°'
    ))

    table.add_row(*(
        _('IPP group resolution'),
        f'{scint_dataset.ipp_group_resolution}° x '
        f'{scint_dataset.ipp_group_resolution}°'
    ))

    table.add_row(*(
        _('Remove data from stations'),
        f"[{', '.join([s for s in scint_dataset.remove_stations])}]"
    ))

    table.add_row(*(
        _('Default spectral p value'),
        str(scint_dataset.default_p)
    ))

    table.add_row(*(
        _('Start time'),
        scint_dataset.start_timestamp.strftime('%Y-%m-%d %H:%M:%S')
    ))

    table.add_row(*(
        _('End time'),
        scint_dataset.end_timestamp.strftime('%Y-%m-%d %H:%M:%S')
    ))

    table.add_row(*(
        _('Scintillation data file'),
        str(scint_data_file)
    ))

    table.add_row(*(
        _('GNSS station data file'),
        str(station_data_file)
    ))

    table.add_row(*(
        _('HDF5 output map file'),
        str(output_file)
    ))

    console.print(table)


@click.version_option()
@click.group()
def cli():
    msg_version = _('IONO-SCINT-MAP - Ionospheric Scintillation Map Generation '
                    'Tool, version %(version)s')

    msg_copyright = ('Copyright (C) 2026 National Institute for Space Research (INPE)')

    msg_authors = _('Authors: André R. F. Martinon, Stephan Stephany, and Eurico R. de Paula.')

    msg = _("This is free software; see the source code for copying conditions."
            " There is ABSOLUTELY NO WARRANTY; not even for MERCHANTABILITY or "
            "FITNESS FOR A PARTICULAR PURPOSE. For details, type "
            "'iono-scint-map show'.\n\nPlease reference the paper 'A new "
            "approach for the generation of real-time GNSS low-latitude "
            "ionospheric scintillation maps' when using the software for "
            "academic work (publications, thesis etc). "
            "<https://doi.org/10.1051/swsc/2023015>")

    console = Console(width=CONSOLE_WIDTH, highlight=False)

    console.rule()
    console.print(msg_version % {'version': __version__},
                  justify='center', style='bold')
    console.print()
    console.print(msg_authors,
                  justify='center', style='green bold')
    console.print(msg_copyright,
                  justify='center', style='cyan')
    console.rule()
    console.print(Padding(Text(msg, justify='full')))
    console.rule()


@cli.command('show')
def show_warranty_information() -> None:
    """Show software licensing information.
    """

    msg_warranty_1 = _('IONO-SCINT-MAP is free software: you can redistribute '
                       'it and/or modify it under the terms of the GNU General '
                       'Public License as published by the Free Software '
                       'Foundation, either version 3 of the License, or (at '
                       'your option) any later version.')
    msg_warranty_2 = _('IONO-SCINT-MAP is distributed in the hope that it will '
                       'be useful, but WITHOUT ANY WARRANTY; without even the '
                       'implied warranty of MERCHANTABILITY or FITNESS FOR A '
                       'PARTICULAR PURPOSE. See the GNU General Public License '
                       'for more details.')
    msg_warranty_3 = _('You should have received a copy of the GNU General '
                       'Public License along with IONO-SCINT-MAP; see the file '
                       'COPYING. If not, see <https://www.gnu.org/licenses/>.')

    console = Console(width=CONSOLE_WIDTH)
    console.print(Padding(
        Text(msg_warranty_1 + '\n\n' +
             msg_warranty_2 + '\n\n' +
             msg_warranty_3, justify='full'), pad=(1, 4)), style='bold white')


# COMMAND: Create Scintillation Map
@cli.command('create')
# Scintillation index
@click.option('-s', '--scint-index',
              type=click.Choice(ScintillationIndex, case_sensitive=False),
              default=ScintillationIndex.S4_1, show_default=True,
              help=_('Select the scintillation index to generate the '
                     'scintillation map. The suffixes represent the signal used'
                     ' to measure the indices. For example, use S4_1 for S4 '
                     'measured in the L1CA band, or equivalent.'))
# Map extent
@click.option('-x', '--extent', nargs=4, type=click.Tuple(
    [click.FloatRange(-180.0, 180.0),
     click.FloatRange(-180.0, 180.0),
     click.FloatRange(-90.0, 90.0),
     click.FloatRange(-90.0, 90.0)]),
              default=[-78.0, -30.0, -39.0, 9.0], show_default=True,
              help=_('Set the boundary box of the interpolation grid.'))
# Elevation cut-off
@click.option('-e', '--elevation',
              type=click.FloatRange(0.0, 90.0), default=30.0,
              show_default=True,
              help=_('Set the minimum elevation angle (degrees) between the '
                     'receiver and the satellite for the scintillation index '
                     'to be considered valid.'))
# Constellations
@click.option('-c', '--constellation',
              type=click.Choice(Constellation, case_sensitive=False),
              default=[Constellation.GPS, Constellation.GLONASS,
                       Constellation.GALILEO, Constellation.BEIDOU],
              show_default=True, multiple=True,
              help=_('Choose the GNSS constellations to be considered when '
                     'filtering scintillation data set.'))
# Preprocessing options
@click.option('-p', '--preprocessing',
              type=click.Choice(PreprocessingOptions, case_sensitive=False),
              default=PreprocessingOptions.VQI, show_default=True,
              help=_('Choose a preprocessing option for grouping and '
                     'aggregating IPPs: '
                     'SAI (Slant S4 values reduced by the average value at the '
                     'centroid); SMI (Slant S4 values reduced by the maximum '
                     'value at the centroid); SQI (Slant S4 values reduced by '
                     'the average value above the 3rd quartile at the centroid)'
                     '; VAI (Vertical S4 values reduced by the average value '
                     'at the centroid); VMI (Vertical S4 values reduced by the '
                     'maximum value at the centroid); VQI (Vertical S4 values '
                     'reduced by the average value above the 3rd quartile at '
                     'the centroid)'))
# Interpolation options
@click.option('-i', '--interpolation',
              type=click.Choice(InterpolationOptions, case_sensitive=False),
              default=InterpolationOptions.GPR, show_default=True,
              help=_('Choose an interpolation algorithm: '
                     'GDA (Scipy GridData); '
                     'GPR (Gaussian Process Regression); '
                     'IDW (Inverse Distance Weighting); '
                     'RBF (Radial Basis Functions)'))
# Interpolation options
@click.option('--interpolation-grid-resolution',
              type=click.FLOAT,
              default=0.25, show_default=True,
              help=_('Set the resolution (lat x lon, degrees) of the '
                     'interpolation grid. '))
# Interpolation options
@click.option('--ipp-group-resolution',
              type=click.FLOAT,
              default=1.0, show_default=True,
              help=_('Set the resolution (lat x lon, degrees) of the IPPs '
                     'grouping and aggregation cell.'))
# Remove stations
@click.option('-r', '--remove-station',
              type=click.STRING,
              default=None, show_default=True, multiple=True,
              help=_('Remove all the GNSS station data from the scintillation '
                     'data set. Inform each ones you want to remove.'))
# Default p
@click.option('--default-p',
              type=click.FLOAT,
              default=2.6, show_default=True,
              help=_('Spectral slope of detrended phase in the 0.1 to 25Hz '
                     'range'))
@click.option('--start',
              type=click.DateTime(formats=['%Y-%m-%dT%H:%M:%S']), default=None,
              help=_('Specify the start time to filter the scintillation data '
                     'set.'))
@click.option('--end',
              type=click.DateTime(['%Y-%m-%dT%H:%M:%S']), default=None,
              help=_('Specify the end time to filter the scintillation data '
                     'set.'))
# Scintillation data file
@click.argument('scint_data_file',
                type=click.Path(exists=True, path_type=Path, resolve_path=True))
# GNSS station information file
@click.argument('gnss_stations_file',
                type=click.Path(exists=True, path_type=Path, resolve_path=True))
# HDF5 output file name
@click.argument('output_file',
                type=click.Path(dir_okay=True,
                                path_type=Path,
                                resolve_path=True), default='scint_map.h5')
def create_scint_map(scint_index: ScintillationIndex,
                     extent: Tuple[float, float, float, float],
                     elevation: float,
                     constellation: Iterable[Constellation],
                     preprocessing: PreprocessingOptions,
                     interpolation: InterpolationOptions,
                     interpolation_grid_resolution: float,
                     ipp_group_resolution: float,
                     remove_station: List[str],
                     default_p: float,
                     start: Optional[datetime.datetime],
                     end: Optional[datetime.datetime],
                     scint_data_file: Path,
                     gnss_stations_file: Path,
                     output_file: Path):
    """Create an ionospheric scintillation map.

    Arguments:

    SCINT_DATA_FILE: file path

        The scintillation data file path. Only CSV and PARQUET formats are
        accepted.

    GNSS_STATIONS_FILE: file path

        The GNSS stations information data file path. Only CSV and PARQUET
        formats are accepted.

    OUTPUT_FILE: file path (Default: ./scint_map.h5)

        The file path for the output interpolated map data in HDF5 format.
    """

    scint_map_data = ScintillationMapDataset(
        scint_index=scint_index,
        elevation=elevation,
        default_p=default_p,
        constellations=constellation,
        remove_stations=remove_station,
        preprocessing=preprocessing,
        interpolation=interpolation,
        ipp_group_resolution=ipp_group_resolution,
        interpolation_grid_resolution=interpolation_grid_resolution,
        map_extent=extent,
        start_timestamp=start,
        end_timestamp=end
    )

    scint_map_data.add_scintillation_data(scint_data_file)
    scint_map_data.add_station_data(gnss_stations_file)

    print_create_map_config(scint_map_data, scint_data_file,
                            gnss_stations_file, output_file)

    scint_map_pipeline = DatasetProcessingPipeline()
    scint_map_pipeline.add_stage(DataCleaningAndFiltering())
    scint_map_pipeline.add_stage(IPPProjection())
    scint_map_pipeline.add_stage(ScintIndexProjection())
    scint_map_pipeline.add_stage(IPPGrouping())
    scint_map_pipeline.add_stage(IPPAggregation())
    scint_map_pipeline.add_stage(MapInterpolation())

    scint_map_data = scint_map_pipeline.process(scint_map_data)

    scint_map_data.to_hdf5(output_file)


# COMMAND: Plot Scintillation Map
@cli.command('plot')
# Scintillation map HDF5 file
@click.argument('scint_map_file',
                type=click.Path(exists=True, path_type=Path, resolve_path=True))
@click.option('--dip/--no-dip', is_flag=True, default=True,
              show_default=True,
              help=_('Choose to plot, or not, the magnetic latitudes. The '
                     'magnetic inclination is calculated using the IGRF model '
                     'valid for the date range of the scintillation data set.'))
@click.option('--png/--no-png', is_flag=True, default=True,
              show_default=True,
              help=_('Choose to save, or not, the resulting plot in the PNG '
                     'format.'))
@click.option('--pdf/--no-pdf', is_flag=True, default=True,
              show_default=True,
              help=_('Choose to save, or not, the resulting plot in the PDF '
                     'format.'))
@click.option('--dpi', type=click.IntRange(72, 1200),
              default=300, show_default=True,
              help=_('Set the DPI (Dots Per Inch) resolution of the resulting '
                     'PNG image file.'))
@click.option('--stations/--no-stations', is_flag=True,
              default=True, show_default=True,
              help=_('Choose to plot, or not, a marker for each GNSS station '
                     'location'))
@click.option('--axis/--no-axis', is_flag=True,
              default=True, show_default=True,
              help=_('Choose to plot, or not, the Longitude and Latitude '
                     'axes.'))
@click.option('--clipping/--no-clipping', is_flag=True,
              default=False, show_default=True,
              help=_('Choose to clipping, or not, the map to the area '
                     'delimited by the convex hull of all IPPs.'))
@click.option('--convex-hull/--no-convex-hull', is_flag=True,
              default=False, show_default=True,
              help=_('Choose to plot, or not, the convex hull polygon.'))
@click.option('--map-extent', nargs=4, type=click.Tuple(
    [click.FloatRange(-180.0, 180.0),
     click.FloatRange(-180.0, 180.0),
     click.FloatRange(-90.0, 90.0),
     click.FloatRange(-90.0, 90.0)]),
              default=[-81.0, -27.0, -39.0, 11.0], show_default=True,
              help=_('Set the boundary box used to plot the map.'))
@click.option('--dip-extent', nargs=4, type=click.Tuple(
    [click.FloatRange(-180.0, 180.0),
     click.FloatRange(-180.0, 180.0),
     click.FloatRange(-90.0, 90.0),
     click.FloatRange(-90.0, 90.0)]),
              default=[-81.0, -27.0, -45.0, 15.0], show_default=True,
              help=_('Set the boundary box used to plot the magnetic '
                     'latitudes.'))
@click.option('--output-dir', type=click.Path(
    exists=True, path_type=Path, resolve_path=True),
              help=_('Set the output directory path. Default: ./'))
def plot_scint_map(scint_map_file: Path,
                   dip: bool,
                   png: bool,
                   pdf: bool,
                   dpi: int,
                   stations: bool,
                   axis: bool,
                   clipping: bool,
                   convex_hull: bool,
                   map_extent: Tuple[float, float, float, float],
                   dip_extent: Tuple[float, float, float, float],
                   output_dir: str):
    """Plot an ionospheric scintillation map.

    Arguments:

    SCINT_MAP_FILE: file path

        The interpolated scintillation map data file path. Only HDF5 files
        generated by the iono-scint-map create command is accepted.
    """

    scint_map_data = ScintillationMapDataset.from_hdf5(scint_map_file)

    if axis:
        fig = Figure(figsize=(12.3, 10.8))
        transparent = False
    else:
        fig = Figure(figsize=(10.8, 10.8))
        igrf_extent = scint_map_data.map_extent
        map_extent = scint_map_data.map_extent
        transparent = True


    ax = fig.subplots(1, subplot_kw=dict(projection=ccrs.PlateCarree()))

    if axis:
        create_world_map(ax, map_extent, color='black', fontsize=18,
                         linewidth=1)

    if dip:
        plot_igrf(ax,
                  scint_map_data.start_timestamp,
                  extent=dip_extent,
                  color='black',
                  fontsize=18)
    if axis:
        plot_scintillation_map_axis(ax, scint_map_data, clipping, convex_hull)
    else:
        plot_scintillation_map_no_axis(ax, scint_map_data, clipping, convex_hull)

    if stations:
        plot_gnss_stations(ax, scint_map_data)

    if not output_dir:
        output_dir = Path('.').resolve()

    file_name = scint_map_file.with_suffix('').name
    if png:
        fig.savefig((output_dir / file_name).with_suffix('.png'),
                    format='png',
                    dpi=dpi,
                    transparent=transparent)
    if pdf:
        fig.savefig((output_dir / file_name).with_suffix('.pdf'),
                    format='pdf')


# COMMAND: Plot IPPs Map
@cli.command('plot-ipp')
# Scintillation map HDF5 file
@click.argument('scint_map_file',
                type=click.Path(exists=True, path_type=Path, resolve_path=True))
@click.option('--dip/--no-dip', is_flag=True, default=True,
              show_default=True,
              help=_('Choose to plot, or not, the magnetic latitudes. The '
                     'magnetic inclination is calculated using the IGRF model '
                     'valid for the date range of the scintillation data set.'))
@click.option('--png/--no-png', is_flag=True, default=True,
              show_default=True,
              help=_('Choose to save, or not, the resulting plot in the PNG '
                     'format.'))
@click.option('--pdf/--no-pdf', is_flag=True, default=True,
              show_default=True,
              help=_('Choose to save, or not, the resulting plot in the PDF '
                     'format.'))
@click.option('--dpi', type=click.IntRange(72, 1200),
              default=300, show_default=True,
              help=_('Set the DPI (Dots Per Inch) resolution of the resulting '
                     'PNG image file.'))
@click.option('--stations/--no-stations', is_flag=True,
              default=True, show_default=True,
              help=_('Choose to plot, or not, a marker for each GNSS station '
                     'location'))
@click.option('--agg/--no-agg', is_flag=True, default=False,
              show_default=True,
              help=_('Choose between show the grouped and aggregated IPPs or '
                     'all the original IPPs before grouping and aggregation.'))
@click.option('--map-extent', nargs=4, type=click.Tuple(
    [click.FloatRange(-180.0, 180.0),
     click.FloatRange(-180.0, 180.0),
     click.FloatRange(-90.0, 90.0),
     click.FloatRange(-90.0, 90.0)]),
              default=[-81.0, -27.0, -39.0, 11.0], show_default=True,
              help=_('Set the boundary box used to plot the map.'))
@click.option('--dip-extent', nargs=4, type=click.Tuple(
    [click.FloatRange(-180.0, 180.0),
     click.FloatRange(-180.0, 180.0),
     click.FloatRange(-90.0, 90.0),
     click.FloatRange(-90.0, 90.0)]),
              default=[-81.0, -27.0, -45.0, 15.0], show_default=True,
              help=_('Set the boundary box used to plot the magnetic '
                     'latitudes.'))
@click.option('--size',
              type=click.Choice([c**2 for c in range(1, 15)]),
              default=64, show_default=True,
              help=_('Set the marker size of each plotted IPP.'))
@click.option('--output-dir', type=click.Path(
    exists=True, path_type=Path, resolve_path=True),
              help=_('Set the output directory path. Default: ./'))
def plot_scint_map(scint_map_file: Path,
                   dip: bool,
                   png: bool,
                   pdf: bool,
                   dpi: int,
                   stations: bool,
                   agg: bool,
                   map_extent: Tuple[float, float, float, float],
                   dip_extent: Tuple[float, float, float, float],
                   size: int,
                   output_dir: str):
    """Plot an IPP (Ionospheric Pierce Points) samples map.

    Arguments:

    SCINT_MAP_FILE: file path

        The interpolated scintillation map data file path. Only HDF5 files
        generated by the iono-scint-map create command is accepted.
    """

    scint_map_data = ScintillationMapDataset.from_hdf5(scint_map_file)

    fig = Figure(figsize=(12.3, 10.8))
    ax = fig.subplots(1, subplot_kw=dict(projection=ccrs.PlateCarree()))
    create_world_map(ax, map_extent, color='black', fontsize=18, linewidth=1)

    if dip:
        plot_igrf(ax,
                  scint_map_data.start_timestamp,
                  extent=dip_extent,
                  color='black',
                  fontsize=18)

    plot_ipp_map(ax, scint_map_data, size, agg)

    if stations:
        plot_gnss_stations(ax, scint_map_data)

    if not output_dir:
        output_dir = Path('.').resolve()

    file_name = scint_map_file.with_suffix('').name + '_ipp'
    if png:
        fig.savefig((output_dir / file_name).with_suffix('.png'),
                    format='png',
                    dpi=dpi)
    if pdf:
        fig.savefig((output_dir / file_name).with_suffix('.pdf'),
                    format='pdf')


if __name__ == '__main__':
    cli()
