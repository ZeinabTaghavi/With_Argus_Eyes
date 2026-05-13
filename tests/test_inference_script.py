import subprocess
import sys
from pathlib import Path
import importlib.util


def test_argus_text_risk_demo_script_help():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "notebooks" / "argus_text_risk_demo.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "ARGUS raw-text NER and RPS-scoring demo" in result.stdout
    assert "--hf-cache-dir" in result.stdout
    assert "--cuda-visible-devices" in result.stdout
    assert "--rps-threshold" in result.stdout


def test_compact_rps_results_matches_notebook_json_shape():
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "notebooks" / "argus_text_risk_demo.py"
    spec = importlib.util.spec_from_file_location("argus_text_risk_demo", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    rows = [
        {"entity": "Zurich", "rps_score": 0.2, "entity_type": "LOC"},
        {"entity": "ETH Zurich", "rps_score": 0.8, "entity_type": "ORG"},
    ]

    assert module.compact_rps_results(rows) == [
        {"entity": "ETH Zurich", "rps_score": 0.8},
        {"entity": "Zurich", "rps_score": 0.2},
    ]
