from __future__ import annotations

from pathlib import Path

from PIL import Image

from uir.core.io.png_stack import list_png_paths

RESAMPLE = {
    "nearest": Image.NEAREST,
    "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC,
    "lanczos": Image.LANCZOS,
}


def upsample_png_stack(src_dir: Path, dst_dir: Path, factor: int, resample: str = "bilinear") -> int:
    if factor < 1:
        raise ValueError("factor must be >= 1")
    if resample not in RESAMPLE:
        raise ValueError(f"resample must be one of {sorted(RESAMPLE)}")

    pngs = list_png_paths(Path(src_dir))
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    rs = RESAMPLE[resample]

    out_idx = 0
    for p in pngs:
        with Image.open(p) as im:
            gray = im.convert("L")
            big = gray.resize((gray.width * factor, gray.height * factor), rs)
        for _ in range(factor):
            big.save(dst_dir / f"s_{out_idx:06d}.png")
            out_idx += 1
    return out_idx
