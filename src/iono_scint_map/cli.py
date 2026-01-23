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
from typing import Literal

import click
import enum

from iono_scint_map import __version__


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

class Constellation(enum.StrEnum):
    GPS = enum.auto()
    GLONASS = enum.auto()
    GALILEO = enum.auto()
    BEIDOU = enum.auto()
    SBAS = enum.auto()
    QZSS = enum.auto()
    NAVIC = enum.auto()
    TBD = enum.auto()

class Preprocessing(enum.Enum):
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

class InterpolationOptions(enum.Enum):
    GDA = enum.auto()
    GPR = enum.auto()
    IDW = enum.auto()
    RBF = enum.auto()

@click.version_option()
@click.group()
def cli():
    click.secho(message=f'Ionospheric Scintillation Map Tool - v. {__version__}',
                bold=True, blink=True, fg='green')

# COMMAND: Create Scintillation Map
@cli.command('create')

# Scintillation index
@click.option('-i', '--scint-index',
              type=click.Choice(ScintillationIndex, case_sensitive=False),
              default=ScintillationIndex.S4_1, show_default=True,
              help='Select the scintillation index to generate the scintillation'
                   ' map. The suffixes represent the signal used to measure the '
                   'indices. For example, use S4_1 for S4 measured in the L1CA '
                   'band, or equivalent.')

# Map extent
@click.option('-x', '--extent', nargs=4, type=click.Tuple(
              [float, float, float, float]), default=[-78.0, -30.0, -39.0, 9.0],
              show_default=True)

# Elevation cut-off
@click.option('-e', '--elevation', type=click.FLOAT,
              default=30.0, show_default=True)

# Constellations
@click.option('-c', '--constellation',
              type=click.Choice(Constellation, case_sensitive=False),
              default=[Constellation.GPS, Constellation.GLONASS,
                       Constellation.GALILEO, Constellation.BEIDOU],
              show_default=True, multiple=True)

# Preprocessing options
@click.option('-pp', '--preprocessing',
              type=click.Choice(Preprocessing, case_sensitive=False),
              default=Preprocessing.VQI, show_default=True)

# Interpolation options
@click.option('-ip', '--interpolation',
              type=click.Choice(InterpolationOptions, case_sensitive=False),
              default=InterpolationOptions.GPR, show_default=True)


def create_scint_map(scint_index, extent, min_elevation, constellation,
                     preprocessing, interpolation):
    click.echo(scint_index.index)
    click.echo(scint_index.type)
    click.echo(scint_index.limits)
    click.echo(extent)
    click.echo(min_elevation)
    click.echo(constellation)
    click.echo(preprocessing)
    click.echo(interpolation)

if __name__ == '__main__':
    cli()
