"""Check that the exact finalist registration is committed before using hold-out data."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REGISTRY = HERE / "modeling" / "finalists.json"


def committed_registration(path: Path = REGISTRY) -> dict:
    repo = HERE.parent
    relative = path.resolve().relative_to(repo)
    committed = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=repo,
                               capture_output=True, check=False)
    if committed.returncode or committed.stdout != path.read_bytes():
        raise PermissionError("commit the exact finalist registration before opening the hold-out set")
    return json.loads(committed.stdout)
