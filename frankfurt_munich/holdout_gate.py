"""Check that the exact finalist registration is committed before using hold-out data."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REGISTRY = HERE / "modeling" / "finalists.json"


def fingerprint_files(paths: dict[str, Path]) -> dict[str, str]:
    fingerprints = {}
    for name, path in sorted(paths.items()):
        with path.open("rb") as file:
            fingerprints[name] = hashlib.file_digest(file, "sha256").hexdigest()
    return fingerprints


def source_fingerprints(project: Path) -> dict[str, str]:
    modeling = project / "modeling"
    files = [*modeling.glob("*.py"), *project.glob("*.py"), *modeling.glob("requirements*.txt")]
    return fingerprint_files({f"source/{path.relative_to(project)}": path for path in files})


def committed_registration(path: Path = REGISTRY) -> dict:
    repo = HERE.parent
    relative = path.resolve().relative_to(repo)
    committed = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=repo,
                               capture_output=True, check=False)
    if committed.returncode or committed.stdout != path.read_bytes():
        raise PermissionError("commit the exact finalist registration before opening the hold-out set")
    registration = json.loads(committed.stdout)
    finalists = registration.get("finalists") if isinstance(registration, dict) else None
    if not isinstance(finalists, list) or not finalists:
        raise PermissionError("the committed registration must list the finalists")
    development = registration.get("development_fingerprints")
    if not isinstance(development, dict):
        raise PermissionError("the committed registration must freeze development fingerprints")
    sources = {name: digest for name, digest in development.items() if name.startswith("source/")}
    if sources != source_fingerprints(HERE):
        raise PermissionError("modeling source changed after registration")
    return registration
