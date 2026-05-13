<p align="center">
  <img src="Argus_Logo.png" width="140" alt="ARGUS logo" />
</p>

# With Argus Eyes: Assessing Retrieval Gaps via Uncertainty Scoring to Detect and Remedy Retrieval Blind Spots

> Official code + data + model releases for the **With Argus Eyes** paper.
>
> **This repo provides:** (1) the full dataset construction pipeline (Wikidata/Wikipedia), (2) released processed artifacts for reproducibility, (3) released model checkpoints, and (4) utilities to score **Retrieval Probability Score (RPS)** for your own inputs.

---

## 🔎 Overview
This repository is the reference implementation for **With Argus Eyes**. It is designed so that an external user can:

1. **Build the dataset from scratch** starting from Wikidata/Wikipedia (heavy).
2. **Reuse released processed artifacts** for quick, reproducible runs.
3. **Use released model checkpoints** (or plug in your own model).
4. **Reproduce the paper figures** from stored experiment outputs.

**Scope note.** This repo focuses on the dataset + experimental pipeline used in the paper (not a full production RAG system).

---

## ✅ What we release
### Code
- **Dataset construction pipeline** (raw → aligned entities → RPS computation → risk-scored subsets).
- **Training / evaluation scripts** used in the paper.
- **RPS scoring utility (Script 14)** to compute RPS for your own inputs using any released model **or your own**.

### Data
Because the full pipeline starts from large external resources (Wikidata/Wikipedia), we separate **raw sources** from **released artifacts**:

- **Raw sources (external):** Wikidata + Wikipedia (not hosted here).
- **Released processed artifacts:** selected outputs needed for reproducibility (tracked under `data/` and/or made available as downloads).
  - For example, the **risk-scored entity artifacts** produced after assigning RPS (e.g., the output of *Script 11*) are provided so users can skip the heaviest stages.
  * Interim data release: we provide a zipped `interim/` artifact containing the outputs of the first four dataset-construction stages. This includes processed entity-level resources derived from the large-scale Wikidata/Wikipedia pipeline, covering approximately 7M processed entities. Users can unzip this artifact under `data/` and continue the remaining pipeline stages to reconstruct the final dataset. Download the artifact here: [interim.zip](https://drive.google.com/file/d/1Q3ona9pY5PHAygH_VYzVwSOqYNS_8L8A/view?usp=sharing).
- **Full/large artifacts:** the largest intermediate corpora are made available via **Hugging Face** releases rather than GitHub.

### Using the interim data release

If you do not want to rerun the heaviest preprocessing stages, download the released `interim.zip` artifact and unzip it under `data/`:

```bash
unzip interim.zip -d data/
```

### Models
We release the **model checkpoints used in the paper** (and fine-tuned variants). These are distributed via **Hugging Face** so users can reproduce results or swap in a new model.

> **Links:** The Hugging Face dataset/model identifiers are listed in the docs (see `docs/DATASET.md` and `docs/MODELS.md`).

---

## 🧭 Repository structure
```
.
├── configs/               # Runnable experiment + pipeline configs (dataset, training, etc.)
├── scripts/               # CLI entry points (pipeline stages, training, evaluation, scoring)
├── src/with_argus_eyes/   # Importable package (core implementation)
├── data/                  # Data layout (raw/intermediate/processed) + policies
├── outputs/               # Reproducible artifacts: metrics, logs, plots, (optional) checkpoints
├── docs/                  # Full instructions (dataset, models, reproduce)
├── paper/                 # Paper sources (LaTeX, figures, bibliography)
└── utils/                 # Small shared helpers
```

---

## 🧰 Requirements
- Python **≥ 3.9**

We recommend using a clean environment (venv/conda) to avoid dependency conflicts.

---

## ⚙️ Installation
Clone and install in editable mode:

```bash
git clone https://github.com/ZeinabTaghavi/With_Argus_Eyes.git
cd With_Argus_Eyes
pip install -e .
```

If you prefer pinned dependencies:

```bash
pip install -r requirements.txt
```

---

## 🏗️ Dataset pipeline
The dataset pipeline builds curated labels and context pairs from Wikidata/Wikipedia and computes RPS to identify high-risk entities.

- **Where to look:** `docs/DATASET.md`
- **Runnable configs:** `configs/dataset/`
- **Implementations:** `scripts/dataset/`

Typical usage:

```bash
bash configs/dataset/run_all.sh
```

> The pipeline writes intermediate/processed artifacts under `data/` (see `data/README.md` for the exact layout and what is tracked vs. generated).

---

## 🧪 Training and analysis
Training and analysis are orchestrated via configuration files and implemented as scripts.

- **Where to look:** `docs/MODELS.md`
- **Configs:** `configs/training/`
- **Scripts:** `scripts/training/`

Typical usage:

```bash
bash configs/training/run_all.sh
```

---

## 🧾 RPS scoring utility (Script 14)
If you have your own list of examples and want to compute RPS for entities under:
- one of the **released models**, or
- **your own** retriever/embedding model,

use the scoring utility in *Script 14*.

This is intended for “plug-and-play” analysis: provide the **context**, **entity label/mention**, and any required metadata, then compute the entity’s RPS (or predicted risk) under a chosen model.

- **Where to look:** `docs/MODELS.md` (usage + expected input format)
- **Entry point:** `scripts/` (Script 14)

### Interactive raw-text RPS notebook
For a user-friendly raw-text workflow, use:

- Notebook: `notebooks/argus_text_risk_demo.ipynb`
- Debug script with the same procedure: `notebooks/argus_text_risk_demo.py`

The notebook lets users enter raw text, configure the retriever, NER model, GPU IDs, Hugging Face cache paths, and the minimum acceptable **Retrieval Probability Score (RPS)**. It then extracts named entities, encodes them with the selected retriever, predicts RPS, highlights low-RPS entities, and prints a compact JSON view containing only entity names and RPS values.

Typical notebook usage:

```bash
jupyter notebook notebooks/argus_text_risk_demo.ipynb
```

For terminal debugging with step-by-step prints:

```bash
python notebooks/argus_text_risk_demo.py --help
python notebooks/argus_text_risk_demo.py --text "Zurich is home to ETH Zurich."
python notebooks/argus_text_risk_demo.py --text-file my_text.txt --retriever contriever
```

The first run may download the configured NER model (`dslim/bert-base-NER` by default) or the selected embedding model if they are not already cached.

---

## 📈 Reproducing figures
Figure mappings and exact reproduction steps are documented in:

- `docs/REPRODUCE.md`

By design, figure reproduction reads from artifacts under `outputs/`.

---

## 📦 Data notes (large files)
This project can include large processed artifacts under `data/`. If you only want the code, consider sparse checkout (example):

```bash
git clone --filter=blob:none --no-checkout https://github.com/ZeinabTaghavi/With_Argus_Eyes.git
cd With_Argus_Eyes
git sparse-checkout init --cone
git sparse-checkout set configs docs scripts src utils pyproject.toml requirements.txt README.md LICENSE
git checkout
```

---

## 📚 Citation
If you use this code, data, or models, please cite the paper:

```bibtex
@article{taghavi_with_argus_eyes,
  title   = {With Argus Eyes},
  author  = {Taghavi, ZeinabSadat and collaborators},
  journal = {arXiv/venue TBD},
  year    = {2026},
}
```

> Replace the BibTeX entry above with the final (camera-ready) BibTeX once available.

---

## 🙏 Acknowledgments
This project builds on public knowledge bases including **Wikidata** and **Wikipedia**.

**Funding (optional but recommended if required by the grant).**
If applicable, include your funding statement here, e.g.:

- This research was supported by the **German Research Foundation (Deutsche Forschungsgemeinschaft, DFG)** under grant **[GRANT_ID]**.

---

## 📝 License
MIT. See `LICENSE`.

---

## 💬 Contact / Questions
- Please use the GitHub **Issues** tab for questions, bug reports, or feature requests.

 
