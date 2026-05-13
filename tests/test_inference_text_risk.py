from pathlib import Path

import joblib
import numpy as np
import pytest

from with_argus_eyes.inference import (
    ArgusTextConfig,
    analyze_text,
    available_retrievers,
    extract_entities,
    resolve_model_artifact,
)
from with_argus_eyes.inference.text_risk import _install_legacy_utils_aliases


class DummyModel:
    def predict(self, x):
        return np.array([0.2, 0.7], dtype=float)[: x.shape[0]]


class DummyRetriever:
    def encode_texts(self, texts, batch_size=32, max_length=256):
        return np.ones((len(texts), 3), dtype=np.float32)


def test_argus_text_config_defaults():
    config = ArgusTextConfig()

    assert config.retriever == "contriever"
    assert config.language == "en"
    assert config.ner_model == "dslim/bert-base-NER"
    assert config.risk_threshold == 0.3
    assert config.ner_threshold == 0.5
    assert config.order == 800
    assert config.k == 50
    assert config.text_mode == "context"
    assert "rader" not in available_retrievers()


def test_resolve_model_artifact_from_existing_outputs():
    repo_root = Path(__file__).resolve().parents[1]
    expected_dir = (
        repo_root
        / "outputs"
        / "12_risk_outputs"
        / "contriever_ratio_unrelevant_below_k_50_o_800_k_50_sampled_average"
        / "models"
    )
    if not expected_dir.exists():
        pytest.skip("Local trained model outputs are not available.")

    artifact = resolve_model_artifact(ArgusTextConfig(workspace_root=repo_root))

    assert artifact.exists()
    assert artifact.suffix == ".joblib"
    assert artifact.parent == expected_dir


def test_extract_entities_filters_by_ner_threshold():
    def fake_ner(_text):
        return [
            {"word": "Switzerland", "entity_group": "LOC", "start": 10, "end": 21, "score": 0.9},
            {"word": "weak", "entity_group": "MISC", "start": 30, "end": 34, "score": 0.1},
        ]

    entities = extract_entities(
        "Text about Switzerland and weak signal.",
        ArgusTextConfig(ner_threshold=0.5),
        ner_pipeline=fake_ner,
    )

    assert len(entities) == 1
    assert entities[0].text == "Switzerland"
    assert entities[0].entity_type == "LOC"


def test_analyze_text_with_mocked_ner_retriever_and_model(tmp_path):
    artifact = tmp_path / "dummy.joblib"
    joblib.dump(DummyModel(), artifact)

    def fake_ner(_text):
        return [
            {"word": "Zurich", "entity_group": "LOC", "start": 0, "end": 6, "score": 0.95},
            {"word": "ETH Zurich", "entity_group": "ORG", "start": 18, "end": 28, "score": 0.98},
        ]

    results = analyze_text(
        "Zurich is home to ETH Zurich.",
        ArgusTextConfig(risk_threshold=0.3),
        ner_pipeline=fake_ner,
        retriever=DummyRetriever(),
        model_artifact=artifact,
    )

    assert [row["entity"] for row in results] == ["Zurich", "ETH Zurich"]
    assert [row["above_threshold"] for row in results] == [False, True]
    assert all(row["retriever"] == "contriever" for row in results)


def test_empty_text_and_no_entities_return_empty_results(tmp_path):
    artifact = tmp_path / "dummy.joblib"
    joblib.dump(DummyModel(), artifact)

    assert analyze_text("", ArgusTextConfig(), ner_pipeline=lambda _text: []) == []
    assert (
        analyze_text(
            "No entities here.",
            ArgusTextConfig(),
            ner_pipeline=lambda _text: [],
            retriever=DummyRetriever(),
            model_artifact=artifact,
        )
        == []
    )


def test_legacy_utils_aliases_are_installed_for_saved_artifacts():
    import sys

    for name in list(sys.modules):
        if name == "utils" or name.startswith("utils."):
            sys.modules.pop(name, None)

    _install_legacy_utils_aliases()

    assert sys.modules["utils.models.mlp"].__name__ == "with_argus_eyes.utils.models.mlp"
    assert sys.modules["utils.models.baselines"].__name__ == "with_argus_eyes.utils.models.baselines"
