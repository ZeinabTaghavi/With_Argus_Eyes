import subprocess
import sys
from pathlib import Path


def test_argus_text_risk_demo_script_help():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "notebooks" / "argus_text_risk_demo.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "ARGUS raw-text NER and risk-scoring demo" in result.stdout
    assert "--hf-cache-dir" in result.stdout
    assert "--cuda-visible-devices" in result.stdout
