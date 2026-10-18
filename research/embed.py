"""DINOv2 embeddings of the play crops. Runs in the detector environment, which has PyTorch.

    ~/Library/Caches/clash-helper/katacr/.venv/bin/python research/embed.py

Writes captures/crops/<id>.emb.npy, one 384 vector per crop, shape (N, K, 384).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path.home() / "Library/Caches/clash-helper/katacr"
CROPS = Path(__file__).resolve().parents[1] / "captures/crops"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def model():
    m = torch.hub.load(str(ROOT / "dinov2"), "dinov2_vits14", source="local", pretrained=False)
    m.load_state_dict(torch.load(ROOT / "weights/dinov2_vits14_pretrain.pth", map_location="cpu"))
    return m.eval().to("mps")


def embed(m, crops: np.ndarray) -> np.ndarray:
    """BGR uint8 crops (M, 224, 224, 3) to unit length embeddings (M, 384)."""
    x = (crops[..., ::-1].astype(np.float32) / 255 - MEAN) / STD
    x = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2))).to("mps")
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 32):
            out.append(torch.nn.functional.normalize(m(x[i : i + 32]), dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 384), np.float32)


def main() -> None:
    m = model()
    for path in sorted(CROPS.glob("*.npz")):
        target = path.with_suffix(".emb.npy")
        if target.exists() and target.stat().st_mtime > path.stat().st_mtime and not sys.argv[1:]:
            continue
        crops = np.load(path)["crops"]
        n, k = crops.shape[:2]
        np.save(target, embed(m, crops.reshape(n * k, 224, 224, 3)).reshape(n, k, -1))
        print(f"{path.stem}: {n} plays embedded", flush=True)


if __name__ == "__main__":
    main()
