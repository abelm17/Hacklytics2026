import os
from pathlib import Path
from PIL import Image, ExifTags
import numpy as np
from tqdm import tqdm
from config import MAX_WIDTH
import rawpy

SUPPORTED_JPEG = {".jpg", ".jpeg", ".JPG", ".JPEG"}
SUPPORTED_RAW  = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".orf", ".rw2",
                  ".raf", ".pef", ".srw", ".CR2", ".CR3", ".NEF", ".ARW",
                  ".DNG", ".ORF", ".RW2", ".RAF", ".PEF", ".SRW"}
SUPPORTED = SUPPORTED_JPEG | SUPPORTED_RAW

def load_images(folder: str) -> list[dict]:
    """Return list of dicts with path, array, metadata."""
    paths = [p for p in Path(folder).rglob("*") if p.suffix in SUPPORTED]
    records = []
    for path in tqdm(paths, desc="Loading images"):
        try:
            if path.suffix.lower() in {s.lower() for s in SUPPORTED_RAW}:
                img = _load_raw(path)   # new function
            else:
                img = Image.open(path).convert("RGB")
                timestamp = _get_timestamp(img, path)
                img = _resize(img)
                arr = np.array(img)
                records.append({
                    "path": str(path),
                    "filename": path.name,
                    "array": arr,
                    "timestamp": timestamp,
                    "width": arr.shape[1],
                    "height": arr.shape[0],
                })
        except Exception as e:
            print(f"[WARN] Skipping {path}: {e}")
    print(f"Loaded {len(records)} images from {folder}")
    return records

def _load_raw(path: Path) -> Image.Image:
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,      # use the camera's white balance
            half_size=True,          # 2x faster, still plenty of resolution
            no_auto_bright=False,
            output_bps=8,            # 8-bit output so it matches JPEG pipeline
        )
    return Image.fromarray(rgb)

def _resize(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w > MAX_WIDTH:
        ratio = MAX_WIDTH / w
        img = img.resize((MAX_WIDTH, int(h * ratio)), Image.LANCZOS)
    return img

def _get_timestamp(img: Image.Image, path: Path) -> float:
    try:
        exif = img._getexif()
        if exif:
            for tag, val in exif.items():
                if ExifTags.TAGS.get(tag) == "DateTimeOriginal":
                    from datetime import datetime
                    dt = datetime.strptime(val, "%Y:%m:%d %H:%M:%S")
                    return dt.timestamp()
    except Exception:
        pass
    return path.stat().st_mtime
