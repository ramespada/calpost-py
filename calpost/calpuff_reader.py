#!/bin/python3

from __future__ import annotations

import argparse
import struct
import warnings
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime, timedelta

import numpy as np

from .utils import (
    FortranSequentialReader,
    decode_string,
    read_string_array,
    reshape_fortran_grid,
    _decompress,
)


class CalpuffOutput:
    """Reader for CALPUFF post-processing output files.

    Instances are usually created with :meth:`read`, which parses the file
    header and exposes metadata, receptor coordinates, and per-species data
    extraction helpers.
    """

    def __init__(self):

        # File metadata
        self.filepath: str = ""
        self.dataset_name: str = ""
        self.dataset_version: str = ""
        self.data_model: str = ""
        self.comments: List[str] = []
        self.file_type: str = ""                    # 'CONC', 'DFLX', 'WFLX', 'VISB'. 'RHO', 'T2D', 'FOG'
        self.model_version: str = ""                # > 7.2.1
        self.title: str = ""

        # Internal vars: (used to handle data)
        self.NCOM: int = 0                          # Number of commented lines at header section.
        self.is_compressed: bool = 0                # Is main data compressed?

        # Date-time specifications
        self.start_date: datetime = None            # first reported date-time "YYYY-MM-DD HH:MM:SS"
        self.end_date: datetime = None              # last reported date-time  "YYYY-MM-DD HH:MM:SS"
        self.time_step: timedelta = None            # time step in seconds.
        self.run_length: int = 0                    # number of time steps on run.
        self.timezone: timedelta = timedelta(0)     # timezone

        # Species information
        self.nsp: int = 0                           # Number of species reported
        self.species: List[str] = []                # List of species with their properties

        # Proj specifications (in the future it should be just a projstring)
        self.proj: Dict = {
            'crs': 'UTM',                           # Coordinate system
            'zone': 0,                              # UTM zone
            'hemis': '',                            # UTM zone
            'datum': 'WGS84'                        # Datum
        }

        # Grid specifications ("sample" grid)
        self.nx = 0
        self.ny = 0
        self.dx = 0.0
        self.dy = 0.0
        self.x0 = 0.0
        self.y0 = 0.0

        # Receptors information
        self.receptor_type: int = 0                 # 0: gridded, 1: discrete, 2:complex-terrain
        self.gridded_receptors: bool = 0            # Is there a grid of receptors?

        self.ngrec: int = 0                         # Number of gridded receptors
        self.ndrec: int = 0                         # Number of discrete receptors
        self.nctrec: int = 0                        # Number of complex-terrain receptors

        self.rgroups: List[int]
        self.igrp: Optional[np.ndarray]

        self.x: Optional[np.ndarray]
        self.y: Optional[np.ndarray]
        self.z: Optional[np.ndarray]                # ground elevation
        self.h: Optional[np.ndarray]                # above ground elevation
        self.ihill: Optional[np.ndarray]            # hill id group (for complex-terrain receptors)

        # Source information
        self.nsrctype: int = 0                      # Number of source types: POINT, LINE, ...
        self.nsrcbytype: List[int]
        self.src_names: List[str]

    @property
    def path(self) -> Path:
        """Return the file path as a :class:`pathlib.Path`."""
        return Path(self.filepath)

    @property
    def nt(self) -> int:
        """Return the number of available timesteps."""
        return self.run_length

    @property
    def metadata(self) -> Dict[str, object]:
        """Return a compact metadata dictionary for quick inspection."""
        return {
            "path": self.path,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "data_model": self.data_model,
            "model_version": self.model_version,
            "title": self.title,
            "comments": list(self.comments),
            "compressed": self.is_compressed,
        }

    @classmethod
    def read(cls, filepath: str) -> "CalpuffOutput":
        """Read a CALPUFF output file and return a populated dataset."""
        out = cls()
        out.filepath = filepath

        with FortranSequentialReader(filepath) as reader:
            record = reader.read_record()
            out.dataset_name = decode_string(record[0:16])
            out.dataset_version = decode_string(record[16:32])
            out.data_model = decode_string(record[32:96])

            ncom = struct.unpack(f"{reader.endian}i", reader.read_record())[0]
            out.comments = [decode_string(reader.read_record()) for _ in range(ncom)]

            record = reader.read_record()
            offset = 0
            model, version, level = struct.unpack_from(f"{reader.endian}12s12s12s", record, offset)
            offset += 36
            ibyr, ibjul, ibhr, ibsec, abtz = struct.unpack_from(f"{reader.endian}4i8s", record, offset)
            offset += 24
            irlg, iavg, nsecdt = struct.unpack_from(f"{reader.endian}3i", record, offset)
            offset += 12
            nx, ny, dxkm, dykm, ione, xorigkm, yorigkm = struct.unpack_from(
                f"{reader.endian}2i2f i 2f", record, offset
            )
            offset += 28

            nssta = struct.unpack_from(f"{reader.endian}i", record, offset)
            offset += 4
            ibcomp, iecomp, jbcomp, jecomp = struct.unpack_from(f"{reader.endian}4i", record, offset)
            offset += 16
            ibsamp, jbsamp, iesamp, jesamp = struct.unpack_from(f"{reader.endian}4i", record, offset)
            offset += 16
            meshdn = struct.unpack_from(f"{reader.endian}i", record, offset)[0]
            offset += 4

            nsrctype, msource, ndrec, nrgrp, nctrec = struct.unpack_from(
                f"{reader.endian}5i", record, offset
            )
            offset += 20
            lsamp, nspout, lcomprs, i2dmet = struct.unpack_from(f"{reader.endian}4i", record, offset)
            offset += 16
            iutmzn, feast, fnorth, rnlat0, relon0, xlat1, xlat2 = struct.unpack_from(
                f"{reader.endian}i6f", record, offset
            )
            offset += 28
            pmap, utmhem, datum, daten, clat0, clon0, clat1, clat2 = struct.unpack_from(
                f"{reader.endian}8s4s8s12s16s16s16s16s", record, offset
            )

            out.nx = (iesamp - ibsamp) * meshdn + 1
            out.ny = (jesamp - jbsamp) * meshdn + 1
            out.dx = (dxkm / meshdn) * 1000.0
            out.dy = (dykm / meshdn) * 1000.0
            out.x0 = (xorigkm + ibsamp * dxkm) * 1000
            out.y0 = (yorigkm + jbsamp * dykm) * 1000

            record = reader.read_record()
            nsrcbytype = struct.unpack(f"{reader.endian}{nsrctype}i", record)

            title = decode_string(reader.read_record())

            csout = [value[0:11].strip() for value in read_string_array(reader.read_record(), 15, nspout)]

            acunit = read_string_array(reader.read_record(), 16, nspout)

            if ndrec > 0:
                record = reader.read_record()
                offset = 0
                out.x = np.frombuffer(record, dtype=np.dtype(f"{reader.endian}f4"), count=ndrec, offset=offset).copy() * 1e3
                offset += 4 * ndrec
                out.y = np.frombuffer(record, dtype=np.dtype(f"{reader.endian}f4"), count=ndrec, offset=offset).copy() * 1e3
                offset += 4 * ndrec
                out.z = np.frombuffer(record, dtype=np.dtype(f"{reader.endian}f4"), count=ndrec, offset=offset).copy()
                offset += 4 * ndrec
                out.h = np.frombuffer(record, dtype=np.dtype(f"{reader.endian}f4"), count=ndrec, offset=offset).copy()
                offset += 4 * ndrec
                out.igrp = np.frombuffer(record, dtype=np.dtype(f"{reader.endian}i4"), count=ndrec, offset=offset).copy()

                rgrpnam = [""] * nrgrp
                record = reader.read_record()
                for i in range(nrgrp):
                    rgrpnam[i] = decode_string(record[i * 80 : (i + 1) * 80])
                out.rgroups = rgrpnam

            if nctrec > 0:
                record = reader.read_record()
                offset = 0
                out.x = np.frombuffer(record, dtype=np.dtype(f"{reader.endian}f4"), count=nctrec, offset=offset).copy() * 1e3
                offset += 4 * nctrec
                out.y = np.frombuffer(record, dtype=np.dtype(f"{reader.endian}f4"), count=nctrec, offset=offset).copy() * 1e3
                offset += 4 * nctrec
                out.z = np.frombuffer(record, dtype=np.dtype(f"{reader.endian}f4"), count=nctrec, offset=offset).copy()
                offset += 4 * nctrec
                out.ihill = np.frombuffer(record, dtype=np.dtype(f"{reader.endian}i4"), count=nctrec, offset=offset).copy()

            cnamsrc = [""] * nsrctype
            for itype in range(nsrctype):
                n = nsrcbytype[itype]
                if n > 0:
                    record = reader.read_record()
                    cnamsrc = read_string_array(record[4:], 16, n)

            out.model_version = version.decode("utf-8").strip()
            out.start_date = datetime.strptime(f"{ibyr} {ibjul} {ibhr} {ibsec}", "%Y %j %H %S")
            out.run_length = irlg
            out.ave_time = timedelta(hours=iavg)
            out.time_step = timedelta(seconds=nsecdt)
            out.end_date = out.start_date + out.time_step * out.run_length

            out.proj['crs'] = pmap.decode("utf-8").strip()
            out.proj['datum'] = datum.decode("utf-8").strip()
            out.proj['zone'] = iutmzn
            out.proj['hemis'] = utmhem.decode("utf-8").strip()

            out.nsp = nspout
            out.species = csout
            out.units = acunit
            out.title = title

            out.gridded_receptors = lsamp
            out.ngrec = out.nx * out.ny
            out.ndrec = ndrec
            out.nctrec = nctrec

            out.nrgrp = nrgrp

            out.nsrctype = nsrctype
            out.nsrcbytype = nsrcbytype
            out.src_names = cnamsrc

            out.is_compressed = lcomprs
            out.NCOM = ncom
            out._endian = reader.endian

        return out

    def info(self) -> str:
        """Return a human-readable metadata summary."""
        idate = self.start_date.strftime("%Y %j %H:%M:%S")
        edate = self.end_date.strftime("%Y %j %H:%M:%S")

        lines = [
            f"File: {self.path}",
                f"Model version: {self.model_version}",
                f"Dataset: {self.dataset_name} v{self.dataset_version}",
                f"Title: {self.title}",
        ]
        if self.is_compressed:
            lines.append("Compressed: True")
        lines.extend(
            [
                f"Run start: {idate}",
                f"Run end: {edate}",
                f"Time step: {self.time_step}",
                f"Timesteps loaded: {self.run_length}",
                f"Grid: {self.nx} x {self.ny}",
                f"Cell size: {self.dx} x {self.dy} m",
                f"Origin: ({self.x0}, {self.y0})",
                (
                    "Projection: "
                    f"{self.proj['datum']}, {self.proj['crs']}, "
                    f"{self.proj['zone']} {self.proj['hemis']}"
                ),
                f"Species ({self.nsp}): {self.species}",
                f"Units: {self.units}",
                f"Receptors: gridded={self.ngrec}, discrete={self.ndrec}, ct={self.nctrec}",
                f"Source types: {self.nsrctype}",
                f"Sources by type: {self.nsrcbytype}",
                f"Source names: {self.src_names}",
            ]
        )
        return "\n".join(lines)

    def get_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Return receptor coordinates.

        For gridded receptors, returns `meshgrid` arrays `(X, Y)`.
        For discrete or complex-terrain receptors, returns the 1D coordinate
        arrays `(x, y)`.
        """

        if self.gridded_receptors:
            x = self.x0 + np.arange(self.nx) * self.dx
            y = self.y0 + np.arange(self.ny) * self.dy
            X, Y = np.meshgrid(x, y)
        elif self.ndrec > 0:
            X = self.x
            Y = self.y
        elif self.nctrec > 0:
            X = self.x
            Y = self.y

        return X, Y

    def get_time_bounds(self) -> list[tuple[datetime, datetime]]:
        """Return begin/end datetimes for each timestep."""
        return [
            (self.start_date + frame * self.time_step, self.start_date + (frame + 1) * self.time_step)
            for frame in range(self.nt)
        ]

    def get_gridded_data(self, pollut: str) -> np.ndarray:
        """Return gridded receptor data for one species.

        Returns an array with shape `[time, y, x]`.
        """
        if not self.gridded_receptors:
            raise ValueError(f"No gridded receptors found in file {self.filepath}")
        if pollut not in self.species:
            raise ValueError(f"No pollutant {pollut} found in file {self.filepath}")

        return self._extract_species_data(pollut, receptor_kind="gridded")

    def get_discrete_data(self, pollut: str) -> np.ndarray:
        """Return discrete receptor data for one species.

        Returns an array with shape `[time, receptor]`.
        """
        if self.ndrec < 1:
            raise ValueError(f"No discrete receptors found in file {self.filepath}")
        if pollut not in self.species:
            raise ValueError(f"No pollutant {pollut} found in file {self.filepath}")

        return self._extract_species_data(pollut, receptor_kind="discrete")

    def get_data(self, pollut: str) -> np.ndarray:
        """Return the main data array for one species.

        This dispatches to gridded or discrete receptor extraction depending
        on the contents of the file.
        """

        if self.gridded_receptors:
            data = self.get_gridded_data(pollut)
        elif self.ndrec > self.ngrec:
            data = self.get_discrete_data(pollut)
        elif self.nctrec > self.ngrec:
            raise ValueError(f"get_data not implemented yet for Complex terrain receptors.")
        else:
            raise ValueError(f"Not sufficient receptors found to make data extraction")

        return data

    def get_2d_field(self, pollut: str) -> np.ndarray:
        """Alias for :meth:`get_data` for API consistency with CALMET."""
        return self.get_data(pollut)

    def get_field(self, pollut: str) -> np.ndarray:
        """Alias for :meth:`get_data`."""
        return self.get_data(pollut)

    def get_time_avg_max(self, pollut: str, interval: int, rank: int = 1) -> np.ndarray:
        """Return the `rank`-th maximum time-averaged field for a species.

        Parameters
        ----------
        pollut:
            Species name.
        interval:
            Number of timesteps included in each average.
        rank:
            Rank of the maximum field to return, where `1` means the highest.
        """
        data = self.get_data(pollut)

        nt = data.shape[0]
        if nt % interval != 0:
            trimmed = nt - (nt % interval)
            data = data[:trimmed]
            print(f"Warning: data trimmed to {trimmed} time steps to match interval")

        n_chunks = data.shape[0] // interval
        new_shape = (n_chunks, interval) + data.shape[1:]
        data_reshaped = data.reshape(new_shape)
        averaged = data_reshaped.mean(axis=1)
        if rank > averaged.shape[0]:
            raise ValueError(
                f"Requested nth max ({rank}) is larger than the number of averaged time blocks ({averaged.shape[0]})"
            )

        sorted_vals = np.sort(averaged, axis=0)[::-1]
        nth_max = sorted_vals[rank - 1]
        return nth_max

    def plot_concentration_animation(
        self,
        pollut: str,
        interval: int = 200,
        cmap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        grid_shape: Optional[tuple[int, int]] = None,
        cell_size: Optional[tuple[float, float]] = None,
    ) -> tuple[object, object]:
        """Build a matplotlib animation of a concentration field.

        Gridded receptors are plotted directly. Discrete receptors are
        interpolated to the CALPUFF output grid using bilinear interpolation
        when the receptor layout is a complete regular grid, otherwise
        inverse-distance weighting.
        """
        if self.gridded_receptors:
            data = self.get_gridded_data(pollut)
            X, Y = self.get_coordinates()
        elif self.ndrec > 0:
            discrete = self.get_discrete_data(pollut)
            X, Y, data = self._interpolate_discrete_series(
                discrete,
                grid_shape=grid_shape,
                cell_size=cell_size,
            )
        else:
            raise ValueError(
                f"Concentration animation requires gridded or discrete receptors in file {self.filepath}"
            )

        if vmin is None:
            vmin = float(np.nanmin(data))
        if vmax is None:
            vmax = float(np.nanmax(data))

        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        fig, ax = plt.subplots()
        mesh = ax.pcolormesh(
            X,
            Y,
            data[0],
            shading="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        title = ax.set_title(self._animation_title(0, pollut))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        colorbar = fig.colorbar(mesh, ax=ax)
        if hasattr(self, "units") and pollut in self.species:
            colorbar.set_label(self.units[self.species.index(pollut)])
        else:
            colorbar.set_label(pollut)

        def update(frame: int):
            mesh.set_array(data[frame].ravel())
            title.set_text(self._animation_title(frame, pollut))
            return mesh, title

        anim = FuncAnimation(fig, update, frames=self.run_length, interval=interval, blit=False)
        return fig, anim

    def _animation_title(self, frame: int, pollut: str) -> str:
        begin = self.start_date + frame * self.time_step
        end = begin + self.time_step
        return f"{pollut} | {begin.isoformat(sep=' ')} -> {end.isoformat(sep=' ')}"

    def _extract_species_data(self, pollut: str, receptor_kind: str) -> np.ndarray:
        if pollut not in self.species:
            raise ValueError(f"No pollutant {pollut} found in file {self.filepath}")

        if receptor_kind == "gridded":
            shape = (self.nt, self.ny, self.nx)
        elif receptor_kind == "discrete":
            shape = (self.nt, self.ndrec)
        else:
            raise ValueError(f"Unsupported receptor kind: {receptor_kind}")

        out = np.zeros(shape, dtype=np.float32)

        with FortranSequentialReader(self.filepath) as reader:
            self._skip_header_records(reader)
            for t in range(self.nt):
                reader.read_record()
                reader.read_record()
                for sp in range(self.nsp):
                    species_name = self.species[sp]
                    gridded_data = None
                    discrete_data = None

                    if self.gridded_receptors:
                        gridded_data = self._read_species_record(reader, self.nx * self.ny, kind="grid")
                    if self.ndrec > 0:
                        discrete_data = self._read_species_record(reader, self.ndrec, kind="discrete")
                    if self.nctrec > 0:
                        self._read_species_record(reader, self.nctrec, kind="discrete")

                    if species_name != pollut:
                        continue

                    if receptor_kind == "gridded":
                        if gridded_data is None:
                            raise ValueError(f"No gridded receptors found in file {self.filepath}")
                        out[t] = gridded_data
                    elif receptor_kind == "discrete":
                        if discrete_data is None:
                            raise ValueError(f"No discrete receptors found in file {self.filepath}")
                        out[t] = discrete_data

        return out

    def _skip_header_records(self, reader: FortranSequentialReader) -> None:
        reader.read_record()
        ncom = struct.unpack(f"{reader.endian}i", reader.read_record())[0]
        for _ in range(ncom):
            reader.read_record()
        reader.read_record()
        reader.read_record()
        reader.read_record()
        reader.read_record()
        reader.read_record()
        if self.ndrec > 0:
            reader.read_record()
            reader.read_record()
        if self.nctrec > 0:
            reader.read_record()
        for i in range(self.nsrctype):
            if self.nsrcbytype[i] > 0:
                reader.read_record()

    def _read_species_record(
        self,
        reader: FortranSequentialReader,
        nvals: int,
        kind: str,
    ) -> np.ndarray:
        if self.is_compressed:
            count_record = reader.read_record()
            ii = struct.unpack(f"{reader.endian}i", count_record)[0]
            payload = reader.read_record()
            raw = np.frombuffer(payload[15:], dtype=np.dtype(f"{reader.endian}f4"), count=ii)
            values = np.asarray(_decompress(raw.tolist()), dtype=np.float32)
        else:
            payload = reader.read_record()
            values = np.frombuffer(payload[15:], dtype=np.dtype(f"{reader.endian}f4"), count=nvals).copy()

        if kind == "grid":
            return reshape_fortran_grid(values, self.nx, self.ny).astype(np.float32, copy=False)
        return values.astype(np.float32, copy=False)

    def _interpolate_discrete_series(
        self,
        data: np.ndarray,
        grid_shape: Optional[tuple[int, int]] = None,
        cell_size: Optional[tuple[float, float]] = None,
    ):
        x_src = np.asarray(self.x, dtype=float).reshape(-1)
        y_src = np.asarray(self.y, dtype=float).reshape(-1)
        values = np.asarray(data, dtype=float)

        if len(x_src) != values.shape[1] or len(y_src) != values.shape[1]:
            raise ValueError("Discrete receptor coordinates do not match the data shape")

        nx, ny = self._resolve_interpolation_grid_shape(grid_shape)
        dx, dy = self._resolve_interpolation_cell_size(cell_size)
        x_target = self.x0 + np.arange(nx) * dx
        y_target = self.y0 + np.arange(ny) * dy
        X, Y = np.meshgrid(x_target, y_target)

        regular = self._as_regular_receptor_grid(x_src, y_src, values)
        if regular is not None:
            x_lines, y_lines, source_grid = regular
            interpolated = self._bilinear_interpolate_series(
                source_grid,
                x_lines,
                y_lines,
                x_target,
                y_target,
            )
            return X, Y, interpolated

        target_points = np.column_stack((X.ravel(), Y.ravel()))
        source_points = np.column_stack((x_src, y_src))

        dx = target_points[:, None, 0] - source_points[None, :, 0]
        dy = target_points[:, None, 1] - source_points[None, :, 1]
        distances = np.hypot(dx, dy)

        with np.errstate(divide="ignore"):
            weights = 1.0 / np.maximum(distances, 1.0e-12) ** 2.0

        exact = distances.min(axis=1) < 1.0e-12
        if np.any(exact):
            weights[exact, :] = 0.0
            nearest = np.argmin(distances[exact, :], axis=1)
            weights[np.where(exact)[0], nearest] = 1.0

        weights /= weights.sum(axis=1, keepdims=True)

        interpolated = np.sum(values[:, None, :] * weights[None, :, :], axis=2)
        return X, Y, interpolated.reshape(values.shape[0], ny, nx)

    def _resolve_interpolation_grid_shape(
        self,
        grid_shape: Optional[tuple[int, int]],
    ) -> tuple[int, int]:
        if grid_shape is None:
            return self.nx, self.ny
        return int(grid_shape[0]), int(grid_shape[1])

    def _resolve_interpolation_cell_size(
        self,
        cell_size: Optional[tuple[float, float]],
    ) -> tuple[float, float]:
        if cell_size is None:
            return self.dx, self.dy
        return float(cell_size[0]), float(cell_size[1])

    def _as_regular_receptor_grid(
        self,
        x_src: np.ndarray,
        y_src: np.ndarray,
        values: np.ndarray,
    ):
        x_lines = np.unique(x_src)
        y_lines = np.unique(y_src)
        if len(x_lines) * len(y_lines) != len(x_src):
            return None

        source_grid = np.full((values.shape[0], len(y_lines), len(x_lines)), np.nan, dtype=float)
        x_index = {value: idx for idx, value in enumerate(x_lines)}
        y_index = {value: idx for idx, value in enumerate(y_lines)}

        for src_idx, (x_val, y_val) in enumerate(zip(x_src, y_src)):
            iy = y_index.get(y_val)
            ix = x_index.get(x_val)
            if iy is None or ix is None or not np.isnan(source_grid[0, iy, ix]):
                return None
            source_grid[:, iy, ix] = values[:, src_idx]

        if np.isnan(source_grid).any():
            return None

        return x_lines, y_lines, source_grid

    def _bilinear_interpolate_series(
        self,
        source_grid: np.ndarray,
        x_lines: np.ndarray,
        y_lines: np.ndarray,
        x_target: np.ndarray,
        y_target: np.ndarray,
    ) -> np.ndarray:
        x_low, x_high, wx = self._bilinear_axis_indices(x_target, x_lines)
        y_low, y_high, wy = self._bilinear_axis_indices(y_target, y_lines)

        q11 = source_grid[:, y_low[:, None], x_low[None, :]]
        q21 = source_grid[:, y_low[:, None], x_high[None, :]]
        q12 = source_grid[:, y_high[:, None], x_low[None, :]]
        q22 = source_grid[:, y_high[:, None], x_high[None, :]]

        wx2d = wx[None, None, :]
        wy2d = wy[None, :, None]

        return (
            q11 * (1.0 - wx2d) * (1.0 - wy2d)
            + q21 * wx2d * (1.0 - wy2d)
            + q12 * (1.0 - wx2d) * wy2d
            + q22 * wx2d * wy2d
        )

    def _bilinear_axis_indices(self, target: np.ndarray, source: np.ndarray):
        source = np.asarray(source, dtype=float)
        target = np.asarray(target, dtype=float)

        upper = np.searchsorted(source, target, side="right")
        upper = np.clip(upper, 1, len(source) - 1)
        lower = upper - 1

        s0 = source[lower]
        s1 = source[upper]
        with np.errstate(divide="ignore", invalid="ignore"):
            weight = (target - s0) / (s1 - s0)
        weight = np.where(s1 == s0, 0.0, weight)
        weight = np.clip(weight, 0.0, 1.0)
        return lower, upper, weight


def read_file(filepath: str) -> "CalpuffOutput":
    warnings.warn(
        "read_file() is deprecated; use CalpuffOutput.read() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return CalpuffOutput.read(filepath)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read CALPUFF output files")
    parser.add_argument("path", help="Path to CALPUFF output file, e.g. CONC.DAT")
    parser.add_argument(
        "pollutant",
        nargs="?",
        help="Species name used by --animate",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Display a matplotlib animation for a gridded concentration field",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=200,
        help="Animation frame interval in milliseconds",
    )
    parser.add_argument(
        "--cmap",
        default="viridis",
        help="Matplotlib colormap for the concentration field",
    )
    parser.add_argument(
        "--grid-shape",
        nargs=2,
        type=int,
        metavar=("NX", "NY"),
        help="Optional interpolation grid shape for discrete receptors",
    )
    parser.add_argument(
        "--cell-size",
        nargs="+",
        type=float,
        metavar=("DX", "DY"),
        help="Optional interpolation cell size in map units; pass DX or DX DY",
    )
    args = parser.parse_args()

    dataset = CalpuffOutput.read(args.path)
    print(dataset.info())

    if args.animate:
        if not args.pollutant:
            parser.error("the pollutant argument is required with --animate")

        import matplotlib.pyplot as plt

        _, anim = dataset.plot_concentration_animation(
            args.pollutant,
            interval=args.interval,
            cmap=args.cmap,
            grid_shape=tuple(args.grid_shape) if args.grid_shape else None,
            cell_size=_parse_cell_size_arg(args.cell_size),
        )
        _ = anim
        plt.show()


def _parse_cell_size_arg(cell_size_arg):
    if not cell_size_arg:
        return None
    if len(cell_size_arg) == 1:
        return float(cell_size_arg[0]), float(cell_size_arg[0])
    if len(cell_size_arg) == 2:
        return float(cell_size_arg[0]), float(cell_size_arg[1])
    raise ValueError("--cell-size accepts either one value (DX) or two values (DX DY)")


if __name__ == "__main__":
    main()
