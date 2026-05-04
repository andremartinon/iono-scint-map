<h1 style="text-align: center">
  📦 IONO-SCINT-MAP
</h1>

<p style="text-align: center; font-weight: bold; font-size: large">
Ionospheric Scintillation Map Generation Tool
</p>

<p style="text-align: center">
   <a href="https://www.gnu.org/licenses/agpl-3.0">
      <img src="https://img.shields.io/badge/License-GNU_AGPLv3-purple.svg?style=flat-square&logo=gnu"
       alt="License: AGPL v3" />
   </a>
   <a href="https://www.python.org/">
      <img src="https://img.shields.io/badge/Python-3.12+-red.svg?style=flat-square&logo=python&logoColor=white"
       alt="Python 3.12 or later">
   </a>
   <a href="https://www.linuxfoundation.org/">
      <img src="https://img.shields.io/badge/Linux-kernel_6.x-cyan.svg?style=flat-square&logo=linux&logoColor=white"
       alt="OS linux-generic">
   </a>
   <a href="https://doi.org/10.1051/swsc/2023015">
      <img src="https://img.shields.io/badge/DOI-10.1051%2Fswsc%2F2023015-blue?style=flat-square"
   alt="DOI to cite IONO-SCINT-MAP">
   </a>
</p>

## 🌟 Highlights

---

Here are the main takeaways of this tool:

- It can be used as a Python package or as a command-line interface tool;
- Gaussian Process Regression interpolation method;
- Preprocessing options and the interpolation method allow for the generation of
  more accurate scintillation maps;
- Low computational cost compatible with real-time demands;
- Interpolated matrix stored in HDF5 format;
- Many options for plotting scintillation maps.

## ℹ️ Overview

---

The IONO-SCINT-MAP is a tool which allow to interpolate scintillation data, i.e.
ionospheric scintillation index measurements, from a network of distributed GNSS
monitoring stations to generate a 2D scintillation map.


The IONO-SCINT-MAP library provides tools for generating three types of 
ionospheric scintillation maps, which are related to the S4, $\sigma_\phi$, and
ROTI scintillation indices. These maps can be used to monitor ionospheric 
scintillation in real time, in studies with historical data, and also to train
machine learning models that predict ionospheric scintillation.

The making of these scintillation maps requires the interpolation of IPP 
samples, given by the ionospheric scintillation index values for each IPP of 
each satellite-station link considering the set of GNSS stations of the given 
area and time interval. Interpolation is performed with the aim of filling in
IPP data gaps, resulting in a smoother map for the considered grid in longitude
and latitude, and for the integration interval.

These maps were implemented using a new approach proposed by MARTINON et al. 
(2023). This new approach consists of a set of preprocessing options and an 
interpolation method. It allows to generate more accurate scintillation maps 
with a low computational cost that is compatible with real-time demands. Former 
approaches to investigate ionospheric scintillation over the Brazilian territory 
include the generation of regional S4 index maps by REZENDE et al. (2007) and 
VANI (2018) which employed different preprocessing options and interpolation
methods.

A more detailed discussion about the ionospheric scintillation maps can be found
in MARTINON (2024).

### ✍️ Authors

---

This software is copyrighted to the National Institute for Space Research 
(INPE/Brazil), and developed by the following researchers:

* André Ricardo Fazanaro Martinon 
([andre.martinon@inpe.br](mailto:andre.martinon@inpe.br))

* Stephan Stephany 
([stephan.stephany@inpe.br](mailto:stephan.stephany@inpe.br))

* Eurico Rodrigues de Paula 
([eurico.paula@inpe.br](mailto:eurico.paula@inpe.br))

## ⬇️ Installation

---

To install the iono-scint-map tool, you must use a Python package manager and
install the package from this GitHub repository, as shown below:

```bash
$ pip install git+ssh://git@github.com/andremartinon/iono-scint-map.git 
```

## 🚀 Usage

---

To use iono-scint-map as a command-line tool (CLI), after installation you can
run iono-scint-map command from a terminal, as shown below:

```bash
$ iono-scint-map --help
```
```
Usage: iono-scint-map [OPTIONS] COMMAND [ARGS]...

  IONO-SCINT-MAP - Ionospheric Scintillation Map Generation Tool

  Copyright (C) 2026 National Institute for Space Research (INPE)

  Authors: André R. F. Martinon, Stephan Stephany, and Eurico R. de Paula.

  This is free software; see the source code for copying conditions. There is
  ABSOLUTELY NO WARRANTY; not even for MERCHANTABILITY or FITNESS FOR A
  PARTICULAR PURPOSE. For details, type 'iono-scint-map show'.

  Please reference the paper 'A new approach for the generation of real-time
  GNSS low-latitude ionospheric scintillation maps' when using the software
  for academic work (publications, thesis etc). Please check:
  <https://doi.org/10.1051/swsc/2023015>

Options:
  --version  Show the version and exit.
  --help     Show the help message and exit.

Commands:
  create    Create an ionospheric scintillation map datafile.
  plot      Plot an ionospheric scintillation map from the datafile.
  plot-ipp  Plot a map of IPP (Ionospheric Pierce Points) samples.
  show      Show the software licensing information.
```

There is a help content for each iono-scint-map command, that can be seen using:

```bash
$ iono-scint-map [create|plot|plot-ipp|show] --help
```

For example, to interpolate a scintillation map using the datasets available in
this repository (folder tests_data), you could use the following command:

```bash
$ iono-scint-map create tests_data/train_map_data.csv tests_data/inct_stations.parquet
```
A HDF5 file will be created in the current folder, using the default file name
`scint_map.h5`. Now to plot the interpolated scintillation map to PNG and PDF
files format you could use the command as seen below:

```bash
$ iono-scint-map plot scint_map.h5
```

The files `scint_map.png` and `scint_map.pdf` will be created in the current
folder.

### Minimal Python example

```python
import cartopy.crs as ccrs

from iono_scint_map.dataset import ScintillationIndex, ScintillationMapDataset
from iono_scint_map.pipeline import (DatasetProcessingPipeline,
                                     DataCleaningAndFiltering, IPPProjection,
                                     ScintIndexProjection, IPPGrouping,
                                     IPPAggregation, MapInterpolation)
from iono_scint_map.plot import (create_world_map, plot_igrf,
                                 plot_scintillation_map_axis,
                                 plot_gnss_stations)
from matplotlib.figure import Figure
from pathlib import Path

input_dir = Path('.').resolve() / 'tests_data'

scint_map_data = ScintillationMapDataset(ScintillationIndex.S4_1)

scint_map_data.add_scintillation_data(input_dir / 'train_map_data.csv')
scint_map_data.add_station_data(input_dir / 'inct_stations.parquet')

scint_map_pipeline = DatasetProcessingPipeline()
scint_map_pipeline.add_stage(DataCleaningAndFiltering())
scint_map_pipeline.add_stage(IPPProjection())
scint_map_pipeline.add_stage(ScintIndexProjection())
scint_map_pipeline.add_stage(IPPGrouping())
scint_map_pipeline.add_stage(IPPAggregation())
scint_map_pipeline.add_stage(MapInterpolation())

scint_map_data = scint_map_pipeline.process(scint_map_data)

map_extent = [-81.0, -27.0, -39.0, 11.0]
dip_extent = map_extent.copy()
dip_extent[2] -= 6
dip_extent[3] += 4

fig = Figure(figsize=(12.3, 10.8))
ax = fig.subplots(1, subplot_kw=dict(projection=ccrs.PlateCarree()))

create_world_map(ax, map_extent, color='black', fontsize=18, linewidth=1)
plot_igrf(ax, scint_map_data.start_timestamp, extent=dip_extent, color='black', fontsize=18)
plot_scintillation_map_axis(ax, scint_map_data)
plot_gnss_stations(ax, scint_map_data)

output_dir = Path('.').resolve()
file_name = 'scint_map'
scint_map_data.to_hdf5((output_dir / file_name).with_suffix('.h5'))
fig.savefig((output_dir / file_name).with_suffix('.png'), format='png', dpi=100)
fig.savefig((output_dir / file_name).with_suffix('.pdf'), format='pdf')
```

## 💭 Feedback and Contributing

---

Use the
["Issues"](https://github.com/andremartinon/iono-scint-map/issues) section of the 
repository to report bugs or request new features. The 
["Pull requests"](https://github.com/andremartinon/iono-scint-map/pulls) section
is used to contribute to the development of iono-scint-map.

## ❗ Citation

Please reference the paper **'A new approach for the generation of real-time 
GNSS low-latitude ionospheric scintillation maps'** when using the software for 
academic work (publications, thesis etc). The paper is available from:
<https://doi.org/10.1051/swsc/2023015>.

## 📃 References

MARTINON, A. R. F.; STEPHANY, S.; PAULA, E. R. de. A new approach for
the generation of real-time GNSS low-latitude ionospheric scintillation maps.
**Journal of Space Weather and Space Climate**, v. 13, p. 18, 2023. Available
from: <<https://doi.org/10.1051/swsc/2023015>>.

MARTINON, A. R. F. **Computational statistics and machine learning approaches
for monitoring and predicting ionospheric scintillation**. 142 p. Thesis
(Doutorado em Computação Aplicada) — Instituto Nacional de Pesquisas Espaciais
(INPE), São José dos Campos, 2024. Available from: 
<<http://urlib.net/ibi/8JMKD3MGP3W34T/4B9G5HL>>.

REZENDE, L. F. C.; PAULA, E. R. de; STEPHANY, S.; KANTOR, I. J.;
MUELLA, M. T. A. H.; SIQUEIRA, P. M. de; CORREA, K. S. Survey and
prediction of the ionospheric scintillation using data mining techniques. 
**Space Weather**, v. 8, n. 6, 2010. Available from:
<<https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2009SW000532>>.

VANI, B. C. **Investigações sobre modelagem, mitigação e predição de
cintilação ionosférica na região brasileira**. Thesis (Doutorado em Ciências
Cartográficas) — Universidade Estadual Paulista (Unesp), Faculdade de Ciências e
Tecnologia, Presidente Prudente, 2018. Available from:
<<http://hdl.handle.net/11449/153701>>.