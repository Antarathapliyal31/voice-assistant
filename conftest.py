"""Make `from server.vad import ...` work when pytest is invoked from the project root."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
