"""Bundled model files used by the default model-backed descriptors."""

from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parent

DPA4_MODEL = MODEL_DIR / "DPA4-Air-OMat24-v20260704.pt"
DPA4C_MODEL = MODEL_DIR / "DPA4C-Air-OMat24-v20260819.pt"
NEP_MODEL = MODEL_DIR / "nep89_20250409.txt"

__all__ = ["DPA4_MODEL", "DPA4C_MODEL", "MODEL_DIR", "NEP_MODEL"]
