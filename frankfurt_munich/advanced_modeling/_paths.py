"""Import path for the advanced experiments.

The experiments reuse the shared modelling code in ../modeling and the dataset
code in .. (frankfurt_munich). Importing this module puts both on sys.path,
after this folder, so that the scripts here run from any working directory.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for path in (HERE.parent, HERE.parent / "modeling", HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
