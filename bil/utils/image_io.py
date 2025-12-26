from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass
class SimpleImage:
    width: int
    height: int
    pixels: List[int]  # flat list [r, g, b, ...]

    def get_rgb(self, x: int, y: int) -> Tuple[int, int, int]:
        idx = (y * self.width + x) * 3
        return tuple(self.pixels[idx : idx + 3])  # type: ignore[return-value]


def simple_image_from_cv2_frame(frame) -> SimpleImage:
    height, width = frame.shape[:2]
    rgb = frame[:, :, ::-1]
    pixels = rgb.reshape(-1).tolist()
    return SimpleImage(width=width, height=height, pixels=pixels)


def pillow_available() -> bool:
    try:
        import PIL  # type: ignore
        import PIL.Image  # noqa: F401
        return True
    except Exception:
        return False


def _write_png_manual(img: SimpleImage, path: Path) -> None:
    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        return (
            struct.pack("!I", len(data))
            + chunk_type
            + data
            + struct.pack("!I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack("!IIBBBBB", img.width, img.height, 8, 2, 0, 0, 0)
    rows = []
    for y in range(img.height):
        row = bytes([0])  # filter type 0
        start = y * img.width * 3
        end = start + img.width * 3
        row += bytes(img.pixels[start:end])
        rows.append(row)
    idat = zlib.compress(b"".join(rows))
    png_bytes = PNG_SIGNATURE + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png_bytes)


def _read_png_manual(path: Path) -> SimpleImage:
    data = path.read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("Unsupported PNG")
    pos = len(PNG_SIGNATURE)
    width = height = None
    idat_data = b""
    while pos < len(data):
        length = struct.unpack("!I", data[pos : pos + 4])[0]
        pos += 4
        chunk_type = data[pos : pos + 4]
        pos += 4
        chunk_data = data[pos : pos + length]
        pos += length
        pos += 4  # skip CRC
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _, _, _ = struct.unpack("!IIBBBBB", chunk_data)
            if bit_depth != 8 or color_type != 2:
                raise ValueError("Only 8-bit RGB PNGs are supported")
        elif chunk_type == b"IDAT":
            idat_data += chunk_data
        elif chunk_type == b"IEND":
            break
    if width is None or height is None:
        raise ValueError("Invalid PNG file")
    raw = zlib.decompress(idat_data)
    pixels: List[int] = []
    stride = width * 3
    expected = (stride + 1) * height
    if len(raw) != expected:
        raise ValueError("Unexpected PNG data length")
    offset = 0
    for _ in range(height):
        filter_type = raw[offset]
        offset += 1
        if filter_type != 0:
            raise ValueError("Only filter type 0 is supported")
        row = raw[offset : offset + stride]
        pixels.extend(row)
        offset += stride
    return SimpleImage(width=width, height=height, pixels=pixels)


def write_image(path: Path, img: SimpleImage) -> None:
    if pillow_available():
        from PIL import Image  # type: ignore

        im = Image.frombytes("RGB", (img.width, img.height), bytes(img.pixels))
        path.parent.mkdir(parents=True, exist_ok=True)
        im.save(path, format="PNG")
        return
    _write_png_manual(img, path)


def read_image(path: Path) -> SimpleImage:
    if pillow_available():
        from PIL import Image  # type: ignore

        with Image.open(path) as im:
            rgb = im.convert("RGB")
            pixels = list(rgb.tobytes())
            width, height = rgb.size
        return SimpleImage(width=width, height=height, pixels=pixels)
    if path.suffix.lower() not in {".png"}:
        raise ImportError("Pillow required for non-PNG images. Install: pip install pillow")
    return _read_png_manual(path)


def grayscale(img: SimpleImage) -> List[float]:
    gray: List[float] = []
    for i in range(0, len(img.pixels), 3):
        r, g, b = img.pixels[i : i + 3]
        gray.append((0.299 * r + 0.587 * g + 0.114 * b))
    return gray


def mean_abs_diff(gray_a: Sequence[float], gray_b: Sequence[float]) -> float:
    total = 0.0
    count = min(len(gray_a), len(gray_b))
    for i in range(count):
        total += abs(gray_a[i] - gray_b[i])
    return total / count if count else 0.0


def average_std(imgs: Sequence[SimpleImage]) -> float:
    if not imgs:
        return 0.0
    totals = [0.0, 0.0, 0.0]
    counts = len(imgs) * imgs[0].width * imgs[0].height
    for img in imgs:
        for i in range(0, len(img.pixels), 3):
            totals[0] += img.pixels[i]
            totals[1] += img.pixels[i + 1]
            totals[2] += img.pixels[i + 2]
    means = [t / counts for t in totals]
    variance = [0.0, 0.0, 0.0]
    for img in imgs:
        for i in range(0, len(img.pixels), 3):
            variance[0] += (img.pixels[i] - means[0]) ** 2
            variance[1] += (img.pixels[i + 1] - means[1]) ** 2
            variance[2] += (img.pixels[i + 2] - means[2]) ** 2
    std = sum((v / counts) ** 0.5 for v in variance) / 3
    return float(std)


def crop(img: SimpleImage, x: int, y: int, w: int, h: int) -> SimpleImage:
    x = max(0, x)
    y = max(0, y)
    w = max(1, min(w, img.width - x))
    h = max(1, min(h, img.height - y))
    pixels: List[int] = []
    for row in range(y, y + h):
        start = (row * img.width + x) * 3
        end = start + w * 3
        pixels.extend(img.pixels[start:end])
    return SimpleImage(width=w, height=h, pixels=pixels)


def resize_nearest(img: SimpleImage, target_w: int, target_h: int) -> SimpleImage:
    pixels: List[int] = []
    for j in range(target_h):
        src_y = int(j * img.height / target_h)
        for i in range(target_w):
            src_x = int(i * img.width / target_w)
            r, g, b = img.get_rgb(src_x, src_y)
            pixels.extend([r, g, b])
    return SimpleImage(width=target_w, height=target_h, pixels=pixels)


def generate_synthetic_frames(
    out_dir: Path, count: int = 30, size: Tuple[int, int] = (320, 240)
) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for i in range(count):
        pixels: List[int] = []
        for y in range(size[1]):
            for x in range(size[0]):
                r = (x + i * 5) % 256
                g = (y + i * 3) % 256
                b = (50 + i * 2) % 256
                pixels.extend([r, g, b])
        img = SimpleImage(width=size[0], height=size[1], pixels=pixels)
        path = out_dir / f"frame_{i:04d}.png"
        write_image(path, img)
        paths.append(path)
    return paths
