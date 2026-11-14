"""Guard for the final evaluation: the frozen files must be exactly as tagged.

The GPU PC's copy of the repository has no `.git` folder, so the tags cannot be checked
there directly. Instead, manifests/frozen.json records the sha256 of every frozen file
as it is at its tag (`grader-v1`, `config-v1`), and a laptop test checks those hashes
against the tagged commits. The final evaluation refuses to run if any file differs.
"""

import hashlib
import json
from pathlib import Path

from grpo_gsm8k.paths import MANIFEST_DIR, REPO_ROOT

FROZEN_PATH = MANIFEST_DIR / "frozen.json"
FROZEN_FILES = {
    "grader-v1": ("src/grpo_gsm8k/graders.py",),
    "config-v1": ("configs/base.yaml", "manifests/preflight.json", "manifests/splits.json", "manifests/revisions.json"),
}


def sha256_of_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def load_frozen_hashes(path: Path = FROZEN_PATH) -> dict[str, dict[str, str]]:
    return json.loads(path.read_text())


def changed_frozen_files(root: Path = REPO_ROOT, path: Path = FROZEN_PATH) -> list[str]:
    """Frozen files whose current content differs from the tagged version ("tag: path")."""
    changed = []
    for tag, files in load_frozen_hashes(path).items():
        for relative_path, expected in files.items():
            file = root / relative_path
            if not file.exists() or sha256_of_bytes(file.read_bytes()) != expected:
                changed.append(f"{tag}: {relative_path}")
    return changed


def require_frozen_files(root: Path = REPO_ROOT) -> None:
    changed = changed_frozen_files(root)
    if changed:
        raise SystemExit(f"refusing to evaluate on test: frozen files changed since their tags: {changed}")
