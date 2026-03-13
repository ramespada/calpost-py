#!/bin/python3

from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .utils import (
    FortranRecordError,
    FortranSequentialReader,
    decode_string,
    parse_payload,
    split_labeled_record,
)


@dataclass
class RunControl:
    """Parsed CALMET run-control record."""

    start_time: datetime
    end_time: datetime
    timezone: str
    irlg: int
    irtype: int
    nx: int
    ny: int
    nz: int
    dgrid: float
    xorigr: float
    yorigr: float
    iwfcod: int
    nssta: int
    nusta: int
    npsta: int
    nowsta: int
    nlu: int
    iwat1: int
    iwat2: int
    lcalgrd: bool
    pmap: str
    datum: str
    daten: str
    feast: float
    fnorth: float
    utmhem: str
    iutmzn: int
    rnlat0: float
    relon0: float
    xlat1: float
    xlat2: float


@dataclass
class CalmetDataset:
    """Reader for `CALMET.DAT` files."""

    path: Path
    dataset_name: str = ""
    dataset_version: str = ""
    data_model: str = ""
    comments: List[str] = field(default_factory=list)
    run_control: Optional[RunControl] = None
    static_fields: Dict[str, np.ndarray] = field(default_factory=dict)
    fields_2d: Dict[str, np.ndarray] = field(default_factory=dict)
    fields_3d: Dict[str, np.ndarray] = field(default_factory=dict)
    time_bounds: List[Tuple[datetime, datetime]] = field(default_factory=list)

    @property
    def nx(self) -> int:
        """Return the grid size in the x direction."""
        return 0 if self.run_control is None else self.run_control.nx

    @property
    def ny(self) -> int:
        """Return the grid size in the y direction."""
        return 0 if self.run_control is None else self.run_control.ny

    @property
    def nz(self) -> int:
        """Return the number of vertical levels."""
        return 0 if self.run_control is None else self.run_control.nz

    @property
    def nt(self) -> int:
        """Return the number of loaded timesteps."""
        return len(self.time_bounds)

    @property
    def dx(self) -> float:
        """Return the x cell size in map units."""
        return 0.0 if self.run_control is None else self.run_control.dgrid

    @property
    def dy(self) -> float:
        """Return the y cell size in map units."""
        return 0.0 if self.run_control is None else self.run_control.dgrid

    @property
    def x(self) -> np.ndarray:
        """Return x coordinates for cell centers."""
        if self.run_control is None:
            raise ValueError("Dataset is not initialized")
        return self.run_control.xorigr + np.arange(self.nx, dtype=np.float32) * self.run_control.dgrid

    @property
    def y(self) -> np.ndarray:
        """Return y coordinates for cell centers."""
        if self.run_control is None:
            raise ValueError("Dataset is not initialized")
        return self.run_control.yorigr + np.arange(self.ny, dtype=np.float32) * self.run_control.dgrid

    @classmethod
    def read(cls, path: str | Path) -> "CalmetDataset":
        """Read a `CALMET.DAT` file and return a populated dataset."""
        dataset = cls(path=Path(path))

        with FortranSequentialReader(dataset.path) as reader:
            dataset._read_header(reader)
            dataset._read_static_fields(reader)
            dataset._read_time_steps(reader)

        return dataset

    def _read_header(self, reader: FortranSequentialReader) -> None:
        record = reader.read_record()
        self.dataset_name = decode_string(record[0:16])
        self.dataset_version = decode_string(record[16:32])
        self.data_model = decode_string(record[32:96])

        ncom_record = reader.read_record()
        ncom = struct.unpack(f"{reader.endian}i", ncom_record)[0]

        self.comments = [decode_string(reader.read_record()) for _ in range(ncom)]

        run_record = reader.read_record()
        self.run_control = _parse_run_control(run_record, reader.endian)

    def _read_static_fields(self, reader: FortranSequentialReader) -> None:
        if self.run_control is None:
            raise ValueError("Run control must be loaded before static fields")

        specs: List[Tuple[str, str, Tuple[int, ...]]] = [
            ("ZFACE", "real", (self.nz + 1,)),
        ]

        if self.run_control.nssta >= 1:
            specs.extend(
                [
                    ("XSSTA", "real", (self.run_control.nssta,)),
                    ("YSSTA", "real", (self.run_control.nssta,)),
                ]
            )
        if self.run_control.nusta >= 1:
            specs.extend(
                [
                    ("XUSTA", "real", (self.run_control.nusta,)),
                    ("YUSTA", "real", (self.run_control.nusta,)),
                ]
            )
        if self.run_control.npsta >= 1:
            specs.extend(
                [
                    ("XPSTA", "real", (self.run_control.npsta,)),
                    ("YPSTA", "real", (self.run_control.npsta,)),
                ]
            )

        specs.extend(
            [
                ("Z0", "real", (self.ny, self.nx)),
                ("ILANDU", "int", (self.ny, self.nx)),
                ("ELEV", "real", (self.ny, self.nx)),
                ("XLAI", "real", (self.ny, self.nx)),
            ]
        )

        if self.run_control.nssta >= 1:
            specs.append(("NEARS", "int", (self.ny, self.nx)))

        for expected_label, dtype_name, shape in specs:
            label, _, payload = split_labeled_record(reader.read_record(), reader.endian)
            if label != expected_label:
                raise FortranRecordError(
                    f"Expected static record {expected_label}, found {label}"
                )
            self.static_fields[label] = parse_payload(payload, dtype_name, shape, reader.endian)

    def _read_time_steps(self, reader: FortranSequentialReader) -> None:
        if self.run_control is None:
            raise ValueError("Run control must be loaded before time steps")

        step_specs = self._time_step_specs()
        layered_targets = {"U", "V", "W", "T"}
        field2d_store: Dict[str, List[np.ndarray]] = {}
        field3d_store: Dict[str, List[np.ndarray]] = {}

        while True:
            try:
                first_record = reader.read_record()
            except EOFError:
                break

            step_arrays_3d: Dict[str, List[np.ndarray]] = {}
            step_arrays_2d: Dict[str, np.ndarray] = {}
            step_start: Optional[datetime] = None
            step_end: Optional[datetime] = None

            all_records = [first_record]
            all_records.extend(reader.read_record() for _ in range(len(step_specs) - 1))

            for raw_record, (expected_label, dtype_name, shape, target_name) in zip(
                all_records, step_specs
            ):
                label, time_info, payload = split_labeled_record(raw_record, reader.endian)
                if label != expected_label:
                    raise FortranRecordError(
                        f"Expected time-step record {expected_label}, found {label}"
                    )

                begin_time = _parse_yyyyjjjhh(time_info[0], time_info[1])
                end_time = _parse_yyyyjjjhh(time_info[2], time_info[3])
                if step_start is None:
                    step_start = begin_time
                    step_end = end_time

                array = parse_payload(payload, dtype_name, shape, reader.endian)
                if target_name in layered_targets:
                    step_arrays_3d.setdefault(target_name, []).append(array)
                else:
                    step_arrays_2d[target_name] = array

            assert step_start is not None and step_end is not None
            self.time_bounds.append((step_start, step_end))

            for name, layers in step_arrays_3d.items():
                field3d_store.setdefault(name, []).append(np.stack(layers, axis=0))
            for name, array in step_arrays_2d.items():
                field2d_store.setdefault(name, []).append(array)

        self.fields_3d = {
            name: np.stack(step_values, axis=0).astype(np.float32, copy=False)
            for name, step_values in field3d_store.items()
        }
        self.fields_2d = {
            name: np.stack(step_values, axis=0) for name, step_values in field2d_store.items()
        }

    def _time_step_specs(self) -> List[Tuple[str, str, Tuple[int, ...], str]]:
        if self.run_control is None:
            raise ValueError("Run control must be loaded before reading time steps")

        specs: List[Tuple[str, str, Tuple[int, ...], str]] = []

        for level in range(1, self.nz + 1):
            level_tag = f"{level:3d}"
            specs.append((f"U-LEV{level_tag}", "real", (self.ny, self.nx), "U"))
            specs.append((f"V-LEV{level_tag}", "real", (self.ny, self.nx), "V"))
            if self.run_control.lcalgrd:
                specs.append((f"WFACE{level_tag}", "real", (self.ny, self.nx), "W"))

        if self.run_control.irtype != 0 and self.run_control.lcalgrd:
            for level in range(1, self.nz + 1):
                specs.append((f"T-LEV{level:3d}", "real", (self.ny, self.nx), "T"))

        if self.run_control.irtype != 0:
            specs.extend(
                [
                    ("IPGT", "int", (self.ny, self.nx), "IPGT"),
                    ("USTAR", "real", (self.ny, self.nx), "USTAR"),
                    ("ZI", "real", (self.ny, self.nx), "ZI"),
                    ("EL", "real", (self.ny, self.nx), "EL"),
                    ("WSTAR", "real", (self.ny, self.nx), "WSTAR"),
                ]
            )
            if self.run_control.npsta != 0:
                specs.append(("RMM", "real", (self.ny, self.nx), "RMM"))
            specs.extend(
                [
                    ("TEMPK", "real", (self.ny, self.nx), "TEMPK"),
                    ("RHO", "real", (self.ny, self.nx), "RHO"),
                    ("QSW", "real", (self.ny, self.nx), "QSW"),
                    ("IRH", "int", (self.ny, self.nx), "IRH"),
                ]
            )
            if self.run_control.npsta != 0:
                specs.append(("IPCODE", "int", (self.ny, self.nx), "IPCODE"))

        return specs

    def info(self) -> str:
        """Return a human-readable metadata summary."""
        lines = [
            f"File: {self.path}",
            f"Dataset: {self.dataset_name} v{self.dataset_version}",
            f"Model: {self.data_model}",
        ]

        if self.run_control is not None:
            rc = self.run_control
            lines.extend(
                [
                    f"Grid: {rc.nx} x {rc.ny} x {rc.nz}",
                    f"Spacing: {rc.dgrid} m",
                    f"Origin: ({rc.xorigr}, {rc.yorigr})",
                    f"Run type: {rc.irtype}",
                    f"CALGRID fields: {rc.lcalgrd}",
                    f"Timezone: {rc.timezone}",
                    f"Run start: {rc.start_time.isoformat(sep=' ')}",
                    f"Run end: {rc.end_time.isoformat(sep=' ')}",
                    f"Timesteps loaded: {self.nt}",
                    f"2D fields: {sorted(self.fields_2d)}",
                    f"3D fields: {sorted(self.fields_3d)}",
                ]
            )

        return "\n".join(lines)

    def get_time_bounds(self) -> List[Tuple[datetime, datetime]]:
        """Return begin/end datetimes for each timestep."""
        return list(self.time_bounds)

    def get_static_field(self, label: str) -> np.ndarray:
        """Return one static field by label, for example ``Z0`` or ``ELEV``."""
        return self.static_fields[label]

    def get_2d_field(self, label: str) -> np.ndarray:
        """Return one time-varying 2D field by label."""
        return self.fields_2d[label]

    def get_3d_field(self, label: str, level: Optional[int] = None) -> np.ndarray:
        """Return one time-varying 3D field by label.

        If `level` is provided, returns the single level with shape
        `[time, y, x]`. Otherwise returns the full field with shape
        `[time, level, y, x]`.
        """
        data = self.fields_3d[label]
        if level is None:
            return data
        if level < 1 or level > data.shape[1]:
            raise IndexError(f"Level must be in 1..{data.shape[1]}")
        return data[:, level - 1, :, :]

    def plot_wind_animation(
        self,
        level: int = 1,
        stride: int = 3,
        interval: int = 200,
        scale: Optional[float] = None,
        cmap: str = "viridis",
    ) -> tuple[object, object]:
        """Build a matplotlib animation of wind speed and vectors."""
        if "U" not in self.fields_3d or "V" not in self.fields_3d:
            raise ValueError("This file does not contain U/V wind fields")

        if level < 1 or level > self.nz:
            raise IndexError(f"Level must be in 1..{self.nz}")

        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        u = self.get_3d_field("U", level=level)
        v = self.get_3d_field("V", level=level)
        speed = np.hypot(u, v)

        x = self.x
        y = self.y
        x2d, y2d = np.meshgrid(x, y)
        qx = x2d[::stride, ::stride]
        qy = y2d[::stride, ::stride]

        fig, ax = plt.subplots()
        mesh = ax.pcolormesh(x2d, y2d, speed[0], shading="auto", cmap=cmap)
        quiver = ax.quiver(
            qx,
            qy,
            u[0, ::stride, ::stride],
            v[0, ::stride, ::stride],
            scale=scale,
        )
        title = ax.set_title(self._time_title(0, level))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        fig.colorbar(mesh, ax=ax, label="Wind speed")

        def update(frame: int):
            mesh.set_array(speed[frame].ravel())
            quiver.set_UVC(
                u[frame, ::stride, ::stride],
                v[frame, ::stride, ::stride],
            )
            title.set_text(self._time_title(frame, level))
            return mesh, quiver, title

        anim = FuncAnimation(fig, update, frames=self.nt, interval=interval, blit=False)
        return fig, anim

    def _time_title(self, frame: int, level: int) -> str:
        begin_time, end_time = self.time_bounds[frame]
        return (
            f"CALMET winds level {level} | "
            f"{begin_time.isoformat(sep=' ')} -> {end_time.isoformat(sep=' ')}"
        )

def _parse_run_control(payload: bytes, endian: str) -> RunControl:
    offset = 0

    ints_10 = struct.unpack_from(f"{endian}10i", payload, offset)
    offset += 40
    axtz = decode_string(payload[offset : offset + 8])
    offset += 8
    irlg, irtype = struct.unpack_from(f"{endian}2i", payload, offset)
    offset += 8
    nx, ny, nz = struct.unpack_from(f"{endian}3i", payload, offset)
    offset += 12
    dgrid, xorigr, yorigr = struct.unpack_from(f"{endian}3f", payload, offset)
    offset += 12
    iwfcod, nssta, nusta, npsta, nowsta, nlu, iwat1, iwat2 = struct.unpack_from(
        f"{endian}8i", payload, offset
    )
    offset += 32
    lcalgrd_raw = struct.unpack_from(f"{endian}i", payload, offset)[0]
    offset += 4
    pmap = decode_string(payload[offset : offset + 8])
    offset += 8
    datum = decode_string(payload[offset : offset + 8])
    offset += 8
    daten = decode_string(payload[offset : offset + 12])
    offset += 12
    feast, fnorth = struct.unpack_from(f"{endian}2f", payload, offset)
    offset += 8
    utmhem = decode_string(payload[offset : offset + 4])
    offset += 4
    iutmzn = struct.unpack_from(f"{endian}i", payload, offset)[0]
    offset += 4
    rnlat0, relon0, xlat1, xlat2 = struct.unpack_from(f"{endian}4f", payload, offset)

    start_time = datetime(
        ints_10[0], ints_10[1], ints_10[2], ints_10[3]
    ) + timedelta(seconds=ints_10[4])
    end_time = datetime(ints_10[5], ints_10[6], ints_10[7], ints_10[8]) + timedelta(
        seconds=ints_10[9]
    )

    return RunControl(
        start_time=start_time,
        end_time=end_time,
        timezone=axtz,
        irlg=irlg,
        irtype=irtype,
        nx=nx,
        ny=ny,
        nz=nz,
        dgrid=dgrid,
        xorigr=xorigr,
        yorigr=yorigr,
        iwfcod=iwfcod,
        nssta=nssta,
        nusta=nusta,
        npsta=npsta,
        nowsta=nowsta,
        nlu=nlu,
        iwat1=iwat1,
        iwat2=iwat2,
        lcalgrd=bool(lcalgrd_raw),
        pmap=pmap,
        datum=datum,
        daten=daten,
        feast=feast,
        fnorth=fnorth,
        utmhem=utmhem,
        iutmzn=iutmzn,
        rnlat0=rnlat0,
        relon0=relon0,
        xlat1=xlat1,
        xlat2=xlat2,
    )

def _parse_yyyyjjjhh(date_code: int, seconds: int) -> datetime:
    year = date_code // 1_000_000
    jday_hour = date_code % 1_000_000
    jday = jday_hour // 100
    hour = jday_hour % 100
    base = datetime(year, 1, 1) + timedelta(days=jday - 1, hours=hour)
    return base + timedelta(seconds=seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read CALMET.DAT files")
    parser.add_argument("path", help="Path to CALMET.DAT")
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Display a matplotlib animation for the wind fields",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=1,
        help="Vertical level used by --animate-winds (1-based)",
    )
    args = parser.parse_args()

    dataset = CalmetDataset.read(args.path)
    print(dataset.info())

    if args.animate:
        import matplotlib.pyplot as plt

        _, anim = dataset.plot_wind_animation(level=args.level)
        # Keep a live reference so matplotlib does not garbage-collect the animation.
        _ = anim
        plt.show()


if __name__ == "__main__":
    main()
