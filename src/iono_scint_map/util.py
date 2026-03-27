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

import logging
import time

__author__ = ['André Ricardo Fazanaro Martinon']
__copyright__ = 'Copyright 2026, National Institute for Space Research (INPE)'
__credits__ = ['Stephan Stephany', 'Eurico Rodrigues de Paula']
__license__ = 'AGPL-3.0-or-later'
__maintainer__ = 'André Ricardo Fazanaro Martinon'
__email__ = 'andre.martinon@inpe.br'
__status__ = 'Production'


# define the benchmark context manager
class Benchmark(object):
    # constructor
    def __init__(self, name):
        # store the name of this benchmark
        self.name = name

    # enter the context manager
    def __enter__(self):
        # record the start time
        self.time_start = time.perf_counter_ns()
        # return this object
        return self

    # exit the context manager
    def __exit__(self, exc_type, exc_value, traceback):
        # record the end time
        logger = logging.getLogger('benchmark')
        self.time_end = time.perf_counter_ns()
        # calculate the duration
        self.duration = self.time_end - self.time_start
        # report the duration
        logger.debug(f'{self.name} took {self.duration*1e-9:.9f} seconds')
        # do not suppress any exception
        return False