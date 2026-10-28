"""Dataset integrity, report provenance, and local artifact helpers."""

import hashlib
import importlib.metadata
import io
import json
import platform
import urllib.request
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def archive(root: Path) -> zipfile.ZipFile:
    spec = json.loads((root / "data/source.json").read_text())
    target = root / "data/raw/source.zip"
    if not target.exists():
        with urllib.request.urlopen(spec["url"], timeout=60) as response:
            content = response.read(20_000_001)
        if len(content) > 20_000_000:
            raise ValueError("Dataset archive exceeds 20 MB limit")
        verify(content, spec["sha256"])
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(content)
        temporary.replace(target)
    content = target.read_bytes()
    verify(content, spec["sha256"])
    # Read individual members; never extract arbitrary paths from a ZIP.
    return zipfile.ZipFile(io.BytesIO(content))


def verify(content: bytes, expected: str) -> None:
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError("Dataset SHA-256 mismatch; remove the cache and verify the source")


def regression(y, prediction):
    return {
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
        "r2": float(r2_score(y, prediction)),
    }


def save_report(root: Path, result: dict) -> dict:
    packages = ["numpy", "pandas", "scikit-learn", "matplotlib", "joblib", "torch", "fastapi"]
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    spec = root / "data/source.json"
    result["provenance"] = {
        "seed": 42,
        "python": platform.python_version(),
        "platform": platform.system(),
        "packages": versions,
        "source": json.loads(spec.read_text()) if spec.exists() else "sklearn.datasets.load_digits",
    }
    (root / "reports").mkdir(exist_ok=True)
    (root / "artifacts").mkdir(exist_ok=True)
    (root / "reports/metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


def figure(root: Path, name: str):
    (root / "reports").mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(root / "reports" / name, dpi=150, bbox_inches="tight")
    plt.close()


def parity(root, y, prediction):
    plt.figure(figsize=(6.5, 4.5))
    plt.scatter(y, prediction, s=18, alpha=0.55, color="#2563eb")
    limits = [min(y), max(y)]
    plt.plot(limits, limits, "--", color="#64748b")
    plt.xlabel("Observed target")
    plt.ylabel("Predicted target")
    plt.title("Untouched test set: observed vs predicted")
    figure(root, "evaluation.png")
