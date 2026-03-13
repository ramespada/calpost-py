#!/bin/python3

from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import BinaryIO, Optional, Sequence, Tuple

import numpy as np


class FortranRecordError(RuntimeError):
    """Raised when a Fortran sequential record is malformed."""

    pass


class FortranSequentialReader:
    """Reader for unformatted Fortran sequential files with 4-byte markers."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.handle: Optional[BinaryIO] = None
        self.endian = "<"

    def __enter__(self) -> "FortranSequentialReader":
        self.handle = self.path.open("rb")
        self.endian = self._detect_endian()
        self.handle.seek(0)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def _detect_endian(self) -> str:
        assert self.handle is not None
        start = self.handle.tell()
        probe = self.handle.read(4)
        if len(probe) != 4:
            raise EOFError("Empty file")

        for endian in ("<", ">"):
            self.handle.seek(start)
            marker = struct.unpack(f"{endian}i", self.handle.read(4))[0]
            if marker <= 0 or marker > 1_000_000:
                continue
            payload = self.handle.read(marker)
            tail = self.handle.read(4)
            if len(payload) != marker or len(tail) != 4:
                continue
            if struct.unpack(f"{endian}i", tail)[0] == marker:
                self.handle.seek(start)
                return endian

        raise FortranRecordError("Could not determine Fortran record endianness")

    def read_record(self) -> bytes:
        """Read and return one Fortran sequential record payload."""
        assert self.handle is not None
        head = self.handle.read(4)
        if not head:
            raise EOFError
        if len(head) != 4:
            raise EOFError("Truncated Fortran record marker")

        nbytes = struct.unpack(f"{self.endian}i", head)[0]
        if nbytes < 0:
            raise FortranRecordError(f"Invalid record length: {nbytes}")

        payload = self.handle.read(nbytes)
        tail = self.handle.read(4)
        if len(payload) != nbytes or len(tail) != 4:
            raise EOFError("Truncated Fortran record payload")

        end_nbytes = struct.unpack(f"{self.endian}i", tail)[0]
        if end_nbytes != nbytes:
            raise FortranRecordError(
                f"Mismatched Fortran record markers: {nbytes} != {end_nbytes}"
            )

        return payload


def decode_string(data: bytes) -> str:
    """Decode an ASCII byte string and strip trailing spaces."""
    return data.decode("ascii", errors="ignore").rstrip()


def split_labeled_record(
    payload: bytes, endian: str
) -> Tuple[str, Tuple[int, int, int, int], bytes]:
    """Split a CALMET labeled record into label, time tuple, and payload."""
    label = decode_string(payload[:8])
    ndathrb, ibsec, ndathre, iesec = struct.unpack_from(f"{endian}4i", payload, 8)
    return label, (ndathrb, ibsec, ndathre, iesec), payload[24:]


def reshape_fortran_grid(values: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """Reshape 1D Fortran-ordered grid values into a `[y, x]` array."""
    return values.reshape((nx, ny), order="F").T


def parse_payload(
    payload: bytes, dtype_name: str, shape: Sequence[int], endian: str
) -> np.ndarray:
    """Parse a binary payload into a NumPy array with the requested shape."""
    dtype = np.dtype(f"{endian}f4" if dtype_name == "real" else f"{endian}i4")
    count = int(np.prod(shape))
    values = np.frombuffer(payload, dtype=dtype, count=count)

    if len(shape) == 1:
        return values.copy()
    if len(shape) != 2:
        raise ValueError(f"Unsupported shape: {shape}")

    ny, nx = shape
    return reshape_fortran_grid(values, nx, ny).copy()


def _decompress(xwork):
    xdat = []
    for value in xwork:
        if value > 0.0:
            xdat.append(value)
        elif math.isnan(value):
            xdat.append(-999.9)
        else:
            xdat.extend([0.0] * int(-value))
    return xdat


def _skip_n_lines(f, n):
    for _ in range(n):
        f.seek(struct.unpack("i", f.read(4))[0] + 4, 1)


def read_string_array(payload: bytes, item_size: int, count: int) -> list[str]:
    """Parse a fixed-width string array from a binary record payload."""
    return [
        decode_string(payload[i * item_size : (i + 1) * item_size])
        for i in range(count)
    ]
