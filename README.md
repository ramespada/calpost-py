# `calpost`

> A Python library or procesing CALPUFF (version > 7.2.1) output files

This library is capable of:
- Read CALPUFF output files such as `CONC.DAT`
- Read `CALMET.DAT` meteorological files
- Quick inspection of metadata,
- Simple matplotlib animations for concentration and wind fields.


## Dependencies

External Python dependencies:
- `numpy`
- `matplotlib`

## Package Layout

- [calpost/calpuff_reader.py](/home/ram/github/ramespada/calpuff-post/calpost/calpuff_reader.py)
  CALPUFF output reader and concentration animation
- [calpost/calmet_reader.py](/home/ram/github/ramespada/calpuff-post/calpost/calmet_reader.py)
  CALMET reader and wind animation
- [calpost/utils.py](/home/ram/github/ramespada/calpuff-post/calpost/utils.py)
  Shared binary/parsing helpers

## Installation

This repository is not packaged on PyPI. Use it directly from the repo root or add the repo parent directory to `PYTHONPATH`.

Example:

```bash
pip install numpy matplotlib
```

Then run Python from the directory that contains the `calpost` package:

```bash
cd /path/to/calpuff-post
python3
```

## Main API

```python
from calpost import CalpuffOutput, CalmetDataset
```

### CALPUFF reader

```python
from calpost import CalpuffOutput

puff = CalpuffOutput.read("CONC.DAT")
print(puff.info())
```

Available methods and properties:
- `CalpuffOutput.read(path)`
- `puff.info()`
- `puff.get_coordinates()`
- `puff.get_data(specie)`
- `puff.get_2d_field(specie)`
- `puff.get_gridded_data(specie)`
- `puff.get_discrete_data(specie)`
- `puff.get_time_bounds()`
- `puff.get_time_avg_max(specie, interval, rank=1)`
- `puff.plot_concentration_animation(...)`
- `puff.nt`

### CALMET reader

```python
from calpost import CalmetDataset

met = CalmetDataset.read("calmet.dat")
print(met.info())
```

Available methods and properties:
- `CalmetDataset.read(path)`
- `met.info()`
- `met.get_time_bounds()`
- `met.get_static_field(label)`
- `met.get_2d_field(label)`
- `met.get_3d_field(label, level=None)`
- `met.plot_wind_animation(...)`
- `met.nx`, `met.ny`, `met.nz`, `met.nt`


## Examples

### Read a CALPUFF concentration file

```python
from calpost import CalpuffOutput

puff = CalpuffOutput.read("CONC.DAT")
print(puff.info())

specie = puff.species[0]
conc = puff.get_data(specie)

print(conc.shape)   # [time, y, x] for gridded receptors
```

### Get receptor coordinates

```python
X, Y = puff.get_coordinates()
```

For gridded receptors, `X` and `Y` are meshgrids. For discrete receptors, they are the receptor coordinates.

### Compute an n-th maximum of time-averaged CALPUFF values

```python
specie = puff.species[0]
c_24hr_2nd_max = puff.get_time_avg_max(specie, interval=24, rank=2)
```

### Read CALMET data

```python
from calpost import CalmetDataset

met = CalmetDataset.read("calmet.dat")
print(met.info())

u = met.get_3d_field("U", level=1)      # [time, y, x]
v = met.get_3d_field("V", level=1)
z0 = met.get_static_field("Z0")         # [y, x]
```

### Animate CALPUFF concentrations

```python
import matplotlib.pyplot as plt
from calpost import CalpuffOutput

puff = CalpuffOutput.read("CONC.DAT")
fig, anim = puff.plot_concentration_animation("SO2")
plt.show()
```

Discrete receptor files are also supported:
- if the receptors form a complete regular grid, interpolation uses bilinear interpolation;
- otherwise it falls back to inverse-distance weighting.

Optional interpolation controls:

```python
fig, anim = puff.plot_concentration_animation(
    "SO2",
    grid_shape=(120, 80),
    cell_size=(250.0, 250.0),
)
```

### Animate CALMET winds

```python
import matplotlib.pyplot as plt
from calpost import CalmetDataset

met = CalmetDataset.read("calmet.dat")
fig, anim = met.plot_wind_animation(level=1, stride=4)
plt.show()
```


--- 
## Command-line usage

### CALPUFF CLI

Read metadata only:

```bash
python3 -m calpost.calpuff_reader CONC.DAT
```

Animate a species:

```bash
python3 -m calpost.calpuff_reader CONC.DAT SO2 --animate
```

Animation options:

```bash
python3 -m calpost.calpuff_reader CONC.DAT SO2 --animate --interval 150 --cmap plasma
python3 -m calpost.calpuff_reader CONC.DAT SO2 --animate --grid-shape 120 80
python3 -m calpost.calpuff_reader CONC.DAT SO2 --animate --cell-size 500
python3 -m calpost.calpuff_reader CONC.DAT SO2 --animate --grid-shape 120 80 --cell-size 250 250
```

Notes:
- `--grid-shape NX NY` is optional and mostly useful for discrete-receptor interpolation.
- `--cell-size DX` uses the same spacing in both directions.
- `--cell-size DX DY` uses different `dx`, `dy`.

### CALMET CLI

Read metadata only:

```bash
python3 -m calpost.calmet_reader calmet.dat
```

Animate winds:

```bash
python3 -m calpost.calmet_reader calmet.dat --animate
python3 -m calpost.calmet_reader calmet.dat --animate --level 2
```

## Notes and limitations

- CALPUFF plotting depends on `matplotlib`; parsing and extraction do not.
- CALMET parsing targets the `CALMET.DAT` layout extracted from the `calmet.for` writer path currently in this repo.
- The CALPUFF reader is aimed at post-processing and inspection, not full model validation.
- Discrete-receptor animation is meant for visualization; interpolated concentration surfaces should not be treated as authoritative without checking the receptor layout and interpolation assumptions.


## To-do

- Add real tests with sample CONC.DAT and CALMET.DAT files covering header parsing, field extraction, and animations.
- Validate CALPUFF support on more output types beyond concentration-oriented use cases, especially DFLX, WFLX, VIS, and related variants.
- Add save-to-file support for animations (gif/mp4) instead of only interactive display.
- Add optional static plotting helpers for common quicklooks, not only animations.
- Improve interpolation controls for discrete CALPUFF receptors, including clearer handling of extrapolation and missing areas.
- Add support for complex-terrain receptor extraction and plotting in the CALPUFF reader.
- Add example notebooks or scripts for typical workflows: extraction, maxima, concentration animation, wind animation.

