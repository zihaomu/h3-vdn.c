#!/usr/bin/env python3
"""Convert a released OpenVDN prompt archive to portable safetensors.

The .pt file is a ZIP-based torch.save archive with a BF16 [800, 5120]
embedding and I64 [800] tags. This converter never unpickles the input.
"""

import argparse
import json
import pathlib
import struct
import zipfile


ROWS = 800
WIDTH = 5120
EMBED_BYTES = ROWS * WIDTH * 2
TAG_BYTES = ROWS * 8


def member(names: list[str], suffix: str) -> str:
    matches = [name for name in names if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"archive must contain exactly one *{suffix}")
    return matches[0]


def convert(source: pathlib.Path, destination: pathlib.Path) -> None:
    with zipfile.ZipFile(source) as archive:
        names = archive.namelist()
        pickle_name = member(names, "/data.pkl")
        embedding_name = member(names, "/data/0")
        tags_name = member(names, "/data/1")
        byteorder_name = member(names, "/byteorder")
        metadata = archive.read(pickle_name)
        if archive.read(byteorder_name).strip() != b"little":
            raise ValueError("only little-endian prompt archives are supported")
        for marker in (b"prompt_embeds", b"token_tags", b"BFloat16Storage",
                       b"LongStorage"):
            if marker not in metadata:
                raise ValueError(f"archive metadata is missing {marker!r}")
        if archive.getinfo(embedding_name).file_size != EMBED_BYTES:
            raise ValueError(f"prompt storage must be {EMBED_BYTES} bytes")
        if archive.getinfo(tags_name).file_size != TAG_BYTES:
            raise ValueError(f"tag storage must be {TAG_BYTES} bytes")
        embedding = archive.read(embedding_name)
        tags = archive.read(tags_name)

    header = {
        "__metadata__": {
            "source": "OpenVDN released prompt .pt",
            "converter": "h3-vdn.c/scripts/convert_vdn_prompt.py",
        },
        "prompt_embeds": {
            "dtype": "BF16", "shape": [ROWS, WIDTH],
            "data_offsets": [0, EMBED_BYTES],
        },
        "token_tags": {
            "dtype": "I64", "shape": [ROWS],
            "data_offsets": [EMBED_BYTES, EMBED_BYTES + TAG_BYTES],
        },
    }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as output:
        output.write(struct.pack("<Q", len(encoded)))
        output.write(encoded)
        output.write(embedding)
        output.write(tags)
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=pathlib.Path)
    parser.add_argument("destination", type=pathlib.Path)
    args = parser.parse_args()
    convert(args.source, args.destination)
    print(args.destination)


if __name__ == "__main__":
    main()
