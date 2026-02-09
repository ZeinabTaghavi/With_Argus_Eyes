import subprocess
import sys
from pathlib import Path

def test_training_cli_help():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "training" / "11_embedding_rank.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "retriever" in result.stdout
