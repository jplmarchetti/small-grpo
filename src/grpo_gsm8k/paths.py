"""Fixed locations inside the repository."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
MANIFEST_DIR = REPO_ROOT / "manifests"
# Large files (Hub downloads, caches) live here; on the laptop it is a symlink to ~/datasets.
DATA_DIR = REPO_ROOT / "data"
HF_CACHE_DIR = DATA_DIR / "hf_cache"

BASE_CONFIG_PATH = CONFIG_DIR / "base.yaml"
SPLITS_PATH = MANIFEST_DIR / "splits.json"
REVISIONS_PATH = MANIFEST_DIR / "revisions.json"
