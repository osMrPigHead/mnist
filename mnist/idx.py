import gzip
import io
import struct
from typing import TYPE_CHECKING
from functools import reduce

if TYPE_CHECKING:
    from _typeshed import SupportsRead, SupportsWrite

import numpy as np

__all__ = [
    "load", "load_from", "loads",
    "dump", "dump_to", "dumps"
]

MAGIC_NUMBER_TYPE = {
    b"\x08": (np.uint8, "B", 1),
    b"\x09": (np.int8, "b", 1),
    b"\x0B": (np.int16, "h", 2),
    b"\x0C": (np.int32, "i", 4),
    b"\x0D": (np.float32, "f", 4),
    b"\x0E": (np.float64, "d", 8)
}
TYPE_MAGIC_NUMBER = {
    np.uint8: (b"\x08", "B", 1),
    np.int8: (b"\x09", "b", 1),
    np.int16: (b"\x0B", "h", 2),
    np.int32: (b"\x0C", "i", 4),
    np.float32: (b"\x0D", "f", 4),
    np.float64: (b"\x0E", "d", 8)
}


def load(fp: "SupportsRead[bytes]") -> np.ndarray:
    assert fp.read(2) == b"\0\0", "The first 2 bytes of an IDX file must be 0"
    idx_type = MAGIC_NUMBER_TYPE[fp.read(1)]
    shape = tuple(
        int.from_bytes(fp.read(4))
        for _ in range(int.from_bytes(fp.read(1)))
    )
    size = reduce(lambda x, y: x * y, shape)
    return np.array(struct.unpack(
        f">{size}{idx_type[1]}",
        fp.read(idx_type[2] * size)
    ), dtype=idx_type[0]).reshape(shape)


def dump(array: np.ndarray, fp: "SupportsWrite[bytes]") -> None:
    fp.write(b"\0\0")
    idx_type = TYPE_MAGIC_NUMBER[array.dtype.type]
    fp.write(idx_type[0])
    fp.write(array.ndim.to_bytes())
    fp.write(b"".join(size.to_bytes(4) for size in array.shape))
    fp.write(struct.pack(f">{array.size}{idx_type[1]}", *array.flat))


def load_from(filename: str, gziped: bool = False) -> np.ndarray:
    opener = gzip.open if gziped else open
    with opener(filename, "rb") as f:
        return load(f)


def loads(values: bytes) -> np.ndarray:
    return load(io.BytesIO(values))


def dump_to(array: np.ndarray, filename: str, gziped: bool = False) -> None:
    opener = gzip.open if gziped else open
    with opener(filename, "wb") as f:
        return dump(array, f)


def dumps(array: np.ndarray) -> bytes:
    dump(array, buf := io.BytesIO())
    return buf.getvalue()
