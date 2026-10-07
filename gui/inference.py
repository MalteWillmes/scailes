"""Run the sc[ai]les wild/farmed classifier (ONNX) on scale images.

Preprocessing mirrors the paper: fit the image within 720x480 px, then resize to
384x512, scale to 0-1 and normalize with the ImageNet statistics (the notebook's
``minimal_transform_resnext``). Class order from training: Wild = 0, Farmed = 1.
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
PRE_RESIZE = (720, 480)  # (width, height) bounding box from the README
INPUT_SIZE = (512, 384)  # (width, height) -> tensor of 384 x 512
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def list_images(folder: Path) -> list[Path]:
    """All images in the folder and its subfolders."""
    return sorted(
        p
        for p in Path(folder).rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def load_model(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _to_rgb8(img: Image.Image) -> Image.Image:
    """RGB, 8 bit. 16-bit scans are stretched from their full range."""
    if img.mode.startswith("I;16") or img.mode in ("I", "F"):
        arr = np.asarray(img, dtype=np.float32)
        top = max(float(arr.max()), 1.0)
        img = Image.fromarray((arr / top * 255).astype(np.uint8))
    return img.convert("RGB")


def load_fitted(path: Path) -> Image.Image:
    """RGB image shrunk (never enlarged) to fit within 720x480, aspect kept."""
    with Image.open(path) as opened:
        img = _to_rgb8(opened)
    img.thumbnail(PRE_RESIZE, Image.Resampling.BILINEAR)
    return img


def preprocess(path: Path) -> np.ndarray:
    """Image file -> float32 array of shape (3, 384, 512)."""
    img = load_fitted(path).resize(INPUT_SIZE, Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return ((arr - MEAN) / STD).transpose(2, 0, 1)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Per-class probability. The model was trained with a sigmoid focal loss on
    one-hot targets, so each output is an independent score: P(wild) and P(farmed)
    do not have to add up to 1."""
    return 1.0 / (1.0 + np.exp(-logits))


def classify(
    session: ort.InferenceSession,
    paths: list[Path],
    root: Path,
    batch_size: int = 8,
    progress=None,
) -> pd.DataFrame:
    """One row per image: folder (relative to root), file, p_wild, p_farmed, and
    error for unreadable files."""
    rows: list[dict] = []

    def row(p: Path, p_wild: float, p_farmed: float, error: str) -> dict:
        rel = p.parent.relative_to(root).as_posix()
        return {
            "folder": "" if rel == "." else rel,
            "file": p.name,
            "p_wild": p_wild,
            "p_farmed": p_farmed,
            "error": error,
        }

    for start in range(0, len(paths), batch_size):
        chunk = paths[start : start + batch_size]
        arrays, ok = [], []
        for p in chunk:
            try:
                arrays.append(preprocess(p))
                ok.append(p)
            except Exception as exc:  # unreadable or corrupt image
                rows.append(row(p, np.nan, np.nan, str(exc)))
        if arrays:
            logits = session.run(None, {"input": np.stack(arrays)})[0]
            for p, (p_wild, p_farmed) in zip(ok, sigmoid(logits)):
                rows.append(row(p, float(p_wild), float(p_farmed), ""))
        if progress:
            progress(min(start + batch_size, len(paths)) / len(paths))
    df = pd.DataFrame(
        rows, columns=["folder", "file", "p_wild", "p_farmed", "error"]
    )
    return df.sort_values(["folder", "file"]).reset_index(drop=True)


def label(p_farmed: pd.Series, threshold: float) -> pd.Series:
    out = pd.Series(
        np.where(p_farmed >= threshold, "Farmed", "Wild"), index=p_farmed.index
    )
    return out.where(p_farmed.notna(), "Error")
