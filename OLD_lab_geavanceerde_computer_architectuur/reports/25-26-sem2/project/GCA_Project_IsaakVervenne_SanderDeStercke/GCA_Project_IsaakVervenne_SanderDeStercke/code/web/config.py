import os
from pathlib import Path

# Absolute path to the project code directory (parent of web/).
# Override with CFD_CODE_DIR environment variable if your layout differs.
# Default assumes web/ is a direct subdirectory of code/ (i.e. code/web/).
_default = Path(__file__).resolve().parent / "../"
PROJECT_CODE_DIR = Path(os.environ.get("CFD_CODE_DIR", str(_default))).resolve()

PYTHON_DIR   = PROJECT_CODE_DIR / "python"
BINARY       = PROJECT_CODE_DIR / "fpv_cfd_stl"
MAKEFILE_DIR = PROJECT_CODE_DIR

UPLOAD_DIR   = Path(__file__).parent / "uploads"
JOBS_DIR     = Path(__file__).parent / "jobs"

UPLOAD_DIR.mkdir(exist_ok=True)
JOBS_DIR.mkdir(exist_ok=True)
