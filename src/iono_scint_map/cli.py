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
import datetime

import click

from pathlib import Path

from pygments.lexer import default
from rich.console import Console
from rich.table import Table
# from rich.text import Text
from typing import Tuple, Iterable, List, Optional

from iono_scint_map import __version__
from iono_scint_map.dataset import (Constellation, PreprocessingOptions,
                                    ScintillationIndex, ScintillationMapDataset)
from iono_scint_map.interpolation import InterpolationOptions
from iono_scint_map.pipeline import (DatasetProcessingPipeline,
                                     DataCleaningAndFiltering, IPPProjection,
                                     ScintIndexProjection, IPPGrouping,
                                     IPPAggregation, MapInterpolation)


@click.version_option()
@click.group()
def cli():
    click.secho(message='-------------------------------------------------'
                        '------------------',
                bold=False,
                fg='white')
    click.secho(message=f'Ionospheric Scintillation Map Generation Tool - '
                        f'version {__version__}',
                bold=True,
                fg='green')
    click.secho(message='Copyright (\u00A9) 2026 - National Institute for '
                        'Spatial Research (INPE)',
                bold=False,
                fg='cyan')
    click.secho(message='-------------------------------------------------'
                        '------------------',
                bold=False,
                fg='white')
    click.echo()


# COMMAND: Create Scintillation Map
@cli.command('create')
# Scintillation index
@click.option('-s', '--scint-index',
              type=click.Choice(ScintillationIndex, case_sensitive=False),
              default=ScintillationIndex.S4_1, show_default=True,
              help='Select the scintillation index to generate the '
                   'scintillation map. The suffixes represent the signal used '
                   'to measure the indices. For example, use S4_1 for S4 '
                   'measured in the L1CA band, or equivalent.')
# Map extent
@click.option('-x', '--extent', nargs=4, type=click.Tuple(
    [click.FloatRange(-180.0, 180.0),
     click.FloatRange(-180.0, 180.0),
     click.FloatRange(-90.0, 90.0),
     click.FloatRange(-90.0, 90.0)]),
              default=[-78.0, -30.0, -39.0, 9.0], show_default=True)
# Elevation cut-off
@click.option('-e', '--elevation', type=click.FloatRange(
    0.0, 90.0), default=30.0, show_default=True)
# Constellations
@click.option('-c', '--constellation',
              type=click.Choice(Constellation, case_sensitive=False),
              default=[Constellation.GPS, Constellation.GLONASS,
                       Constellation.GALILEO, Constellation.BEIDOU],
              show_default=True, multiple=True)
# Preprocessing options
@click.option('-p', '--preprocessing',
              type=click.Choice(PreprocessingOptions, case_sensitive=False),
              default=PreprocessingOptions.VQI, show_default=True)
# Interpolation options
@click.option('-i', '--interpolation',
              type=click.Choice(InterpolationOptions, case_sensitive=False),
              default=InterpolationOptions.GPR, show_default=True)
# Interpolation options
@click.option('--interpolation-grid-resolution',
              type=click.FLOAT,
              default=0.25, show_default=True)
# Interpolation options
@click.option('--ipp-group-resolution',
              type=click.FLOAT,
              default=1.0, show_default=True)
# Remove stations
@click.option('-r', '--remove-station',
              type=click.STRING,
              default=None, show_default=True, multiple=True)
# Default p
@click.option('--default-p',
              type=click.FLOAT,
              default=2.6, show_default=True)
@click.option('--start',
              type=click.DateTime(formats=['%Y-%m-%dT%H:%M:%S']), default=None)
@click.option('--end',
              type=click.DateTime(['%Y-%m-%dT%H:%M:%S']), default=None)
# Scintillation data file
@click.argument('scint_data_file',
                type=click.Path(exists=True, path_type=Path, resolve_path=True))
# GNSS station information file
@click.argument('gnss_stations_file',
                type=click.Path(exists=True, path_type=Path, resolve_path=True))
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
                     gnss_stations_file: Path):
    """Create an ionospheric scintillation map

    """
    console = Console()
    table = Table(title="Scintillation Map Generation Parameters")
    table.add_column("Parameter", justify="left", style="cyan")
    table.add_column("Value", justify="left", style="white")
    table.add_row('Scintillation index', scint_index.index.upper())
    table.add_row('Scintillation index type',
                  scint_index.type.upper())
    table.add_row('Scintillation index limits',
                  f'min: {scint_index.limits["min"]}, '
                  f'max: {scint_index.limits["max"]}')
    table.add_row('Interpolated map extents',
                  f'[{"°, ".join(map(str, extent))}]')
    table.add_row('Elevation limit', f'{elevation}°')
    # table.add_row('Constellations', constellation)
    table.add_row('Preprocessing options',
                  preprocessing.value.upper())
    table.add_row('Interpolation method',
                  interpolation.value.upper())
    table.add_row('Interpolation grid resolution',
                  f'{interpolation_grid_resolution}° x '
                  f'{interpolation_grid_resolution}°')
    table.add_row('IPP group resolution',
                  f'{ipp_group_resolution}° x '
                  f'{ipp_group_resolution}°')
    # table.add_row('Remove data from stations', remove_station)
    table.add_row('Default spectral p value', str(default_p))
    if start is not None:
        table.add_row('Start time',
                      start.strftime('%Y-%m-%dT%H:%M:%S'))
    if end is not None:
        table.add_row('End time',
                      end.strftime('%Y-%m-%dT%H:%M:%S'))

    table.add_row('Scintillation data file', str(scint_data_file))
    table.add_row('GNSS station data file', str(gnss_stations_file))
    console.print(table)

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

    scint_map_pipeline = DatasetProcessingPipeline()
    scint_map_pipeline.add_stage(DataCleaningAndFiltering())
    scint_map_pipeline.add_stage(IPPProjection())
    scint_map_pipeline.add_stage(ScintIndexProjection())
    scint_map_pipeline.add_stage(IPPGrouping())
    scint_map_pipeline.add_stage(IPPAggregation())
    scint_map_pipeline.add_stage(MapInterpolation())

    scint_map_data = scint_map_pipeline.process(scint_map_data)

    #scint_map_data.to_hdf5()
    print(scint_map_data.grouped)
    print(scint_map_data.interpolated_map)


if __name__ == '__main__':
    cli()
