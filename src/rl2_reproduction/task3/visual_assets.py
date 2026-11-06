"""Read existing Doom WAD texture resources without changing source assets."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import struct

import numpy as np


def sha256(data):
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class Lump:
    name: str
    data: bytes
    index: int
    offset: int

    @property
    def sha256(self):
        return sha256(self.data)


def read_wad(path):
    raw = Path(path).read_bytes()
    magic, count, directory = struct.unpack_from("<4sii", raw)
    if magic not in (b"IWAD", b"PWAD") or count < 0 or directory + count * 16 > len(raw):
        raise ValueError("Invalid WAD header/directory")
    result = []
    for index in range(count):
        offset, size, name = struct.unpack_from("<ii8s", raw, directory + index * 16)
        if offset < 0 or size < 0 or offset + size > len(raw):
            raise ValueError("WAD lump outside file")
        result.append(Lump(name.rstrip(b"\0").decode("ascii"), raw[offset:offset+size], index, offset))
    return result


class WadAssets:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.wad_sha256 = sha256(self.path.read_bytes())
        self.lumps = read_wad(self.path)
        self.by_name = {item.name: item for item in self.lumps}
        palette = self.by_name["PLAYPAL"].data
        self.palette = np.frombuffer(palette[:768], dtype=np.uint8).reshape(256, 3)
        pnames = self.by_name["PNAMES"].data
        count, = struct.unpack_from("<i", pnames)
        self.patch_names = [pnames[4+i*8:12+i*8].rstrip(b"\0").decode("ascii") for i in range(count)]
        self.textures = {}
        for lump_name in ("TEXTURE1", "TEXTURE2"):
            if lump_name not in self.by_name:
                continue
            raw = self.by_name[lump_name].data
            count, = struct.unpack_from("<i", raw)
            for index in range(count):
                offset, = struct.unpack_from("<i", raw, 4 + index * 4)
                name, masked, width, height, column_dir, patch_count = struct.unpack_from("<8sihhih", raw, offset)
                name = name.rstrip(b"\0").decode("ascii")
                patches = []
                for j in range(patch_count):
                    x, y, patch_id, stepdir, colormap = struct.unpack_from("<hhhhh", raw, offset+22+j*10)
                    patches.append(dict(x=x, y=y, name=self.patch_names[patch_id], patch_id=patch_id))
                self.textures[name] = dict(width=width, height=height, patches=patches, definition_lump=lump_name,
                                           definition_bytes=raw[offset:offset+22+patch_count*10])
        self.flat_names = []
        in_flats = False
        for item in self.lumps:
            if item.name == "F_START":
                in_flats = True
            elif item.name == "F_END":
                in_flats = False
            elif in_flats and len(item.data) == 4096:
                self.flat_names.append(item.name)
        self.texture_names = sorted(self.textures)

    def decode_patch(self, name):
        raw = self.by_name[name].data
        width, height, left, top = struct.unpack_from("<hhhh", raw)
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid patch dimensions: {name}")
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        for x in range(width):
            offset, = struct.unpack_from("<I", raw, 8+4*x)
            previous = -1
            while raw[offset] != 255:
                row, count = raw[offset], raw[offset+1]
                if row <= previous:  # Tall-patch relative topdelta extension.
                    row += previous
                previous = row
                pixels = np.frombuffer(raw[offset+3:offset+3+count], dtype=np.uint8)
                valid = min(count, height-row)
                if valid > 0:
                    rgba[row:row+valid, x, :3] = self.palette[pixels[:valid]]
                    rgba[row:row+valid, x, 3] = 255
                offset += count+4
        return rgba

    def decode_texture(self, name):
        texture = self.textures[name]
        output = np.zeros((texture["height"], texture["width"], 4), dtype=np.uint8)
        for item in texture["patches"]:
            patch = self.decode_patch(item["name"])
            x0, y0 = max(0, item["x"]), max(0, item["y"])
            x1, y1 = min(texture["width"], item["x"]+patch.shape[1]), min(texture["height"], item["y"]+patch.shape[0])
            if x1 <= x0 or y1 <= y0:
                continue
            cut = patch[y0-item["y"]:y1-item["y"], x0-item["x"]:x1-item["x"]]
            target = output[y0:y1, x0:x1]
            mask = cut[:,:,3] > 0
            target[mask] = cut[mask]
        return output

    def decode_flat(self, name):
        if name not in self.flat_names:
            raise KeyError(f"Not an existing 64x64 flat: {name}")
        return self.palette[np.frombuffer(self.by_name[name].data, dtype=np.uint8).reshape(64,64)]

    def resource_record(self, name, kind):
        common = dict(name=name, kind=kind, source_file=str(self.path), source_sha256=self.wad_sha256,
                      palette_lump="PLAYPAL", palette_index=0, palette_sha256=self.by_name["PLAYPAL"].sha256)
        if kind == "flat":
            if name not in self.flat_names:
                raise KeyError(name)
            return dict(**common, lump=name, lump_sha256=self.by_name[name].sha256, width=64, height=64)
        if kind != "texture":
            raise ValueError("kind must be flat or texture")
        definition = self.textures[name]
        return dict(**common, width=definition["width"], height=definition["height"],
                    definition_lump=definition["definition_lump"],
                    definition_lump_sha256=self.by_name[definition["definition_lump"]].sha256,
                    definition_record_sha256=sha256(definition["definition_bytes"]),
                    pnames_sha256=self.by_name["PNAMES"].sha256,
                    patches=[dict(**item, lump_sha256=self.by_name[item["name"]].sha256) for item in definition["patches"]])
