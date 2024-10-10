import os
from pathlib import Path

HOME_DIR = Path(__file__).parent.parent
DATA_DIR = Path("/pscratch/sd/i/imendoza/data")

os.environ["JAX_ENABLE_X64"] = "True"  # change only in a very controlled setting
