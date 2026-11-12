"""Extract selected ASL Citizen ZIP members using HTTP range requests.

Research-use only. Downloaded clips stay outside Git and are not organizer data.
"""

import argparse
import io
import struct
import urllib.request
import zipfile
import zlib
from pathlib import Path

URL = "https://download.microsoft.com/download/b/8/8/b88c0bae-e6c1-43e1-8726-98cf5af36ca4/ASL_Citizen.zip"


def fetch_range(start, end):
    request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    if len(data) != end - start + 1:
        raise RuntimeError(f"Range response {len(data)} bytes, expected {end - start + 1}")
    return data


class RangeArchive(io.RawIOBase):
    def __init__(self, size):
        self.size = size
        self.position = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        if self.position >= self.size or size == 0:
            return b""
        end = self.size - 1 if size < 0 else min(self.size - 1, self.position + size - 1)
        data = fetch_range(self.position, end)
        self.position = end + 1
        return data


def archive_index(archive_size, tail_size=16 * 1024 * 1024):
    reader = zipfile.ZipFile(RangeArchive(archive_size))
    reader.concat = 0
    return reader, 0


def extract(archive_size, name, output):
    reader, tail_start = archive_index(archive_size)
    try:
        record = reader.getinfo(name)
    except KeyError as exc:
        raise FileNotFoundError(name) from exc
    local_offset = tail_start + record.header_offset
    header = fetch_range(local_offset, local_offset + 29)
    if header[:4] != b"PK\x03\x04":
        raise RuntimeError("Invalid local file header")
    fields = struct.unpack("<4s5H3L2H", header)
    start = local_offset + 30 + fields[-2] + fields[-1]
    compressed = fetch_range(start, start + record.compress_size - 1)
    raw = zlib.decompress(compressed, -15) if record.compress_type == 8 else compressed
    if len(raw) != record.file_size:
        raise RuntimeError("Extracted size mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(raw)
    return {"name": name, "bytes": len(raw)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("output", type=Path)
    parser.add_argument("--archive-size", type=int, default=45924134223)
    args = parser.parse_args()
    print(extract(args.archive_size, args.name, args.output))


if __name__ == "__main__":
    main()
