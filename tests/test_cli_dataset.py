import subprocess
import sys
from pathlib import Path

def test_dataset_cli_help():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "dataset" / "01_get_items.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Wikidata" in result.stdout or "wikidata" in result.stdout
