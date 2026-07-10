# README.md: Main project documentation and setup guide for deal-intelligence-engine.

# Deal Intelligence Engine

The **Deal Intelligence Engine** is a multimodal deep learning system designed to predict B2B sales deal outcomes and infer causal loss reasons from seller-side communication data (emails, CRM activity logs).

This repository is currently in **Phase 0: Project Scaffolding and Data Acquisition**. No ML model architecture or training code is implemented in this phase.

---

## Why Poetry?
For environment reproducibility and dependency resolution, this project uses **Poetry** instead of a simple `pip` + `requirements.txt`.
*   **Deterministic Environments:** By resolving and pinning transitive dependencies in `poetry.lock`, all developers, CI tests, and production environments execute the exact same packages.
*   **Conflict Prevention:** Poetry checks all package requirements in `pyproject.toml` to prevent version collisions.

---

## Directory Structure
```
deal-intelligence-engine/
├── .github/workflows/
│   └── ci.yml              # CI configuration (linters + tests)
├── configs/
│   ├── data.yaml           # Data configuration keys
│   ├── model.yaml          # Model baseline architectures
│   └── train.yaml          # Hyperparameters and trainer config
├── dashboard/              # [Empty] Streamlit/Web dashboard UI (Phase 2+)
├── data/
│   ├── raw/                # gitignored raw data files
│   │   └── README.md       # Download links and dataset instructions
│   └── processed/          # gitignored pre-processed deal timelines
├── notebooks/              # [Empty] Analysis notebooks
├── scripts/
│   └── download_data.py    # Raw data integrity validator script
├── src/                    # Core source packages
│   ├── ingestion/          # Data loading pipelines (empty subpackage)
│   ├── features/           # Feature extractor modules (empty subpackage)
│   ├── fusion/             # Multimodal fusion layers (empty subpackage)
│   ├── models/             # PyTorch model code (empty subpackage)
│   ├── explainability/     # Causal explanation algorithms (empty subpackage)
│   ├── eval/               # Validation/metric evaluations (empty subpackage)
│   ├── __init__.py
│   └── config.py           # OmegaConf-based AppConfig loader
├── tests/
│   ├── test_config.py      # Config loading and override tests
│   └── test_smoke.py       # Import and basic load sanity tests
├── .gitignore
├── .pre-commit-config.yaml # Pre-commit hook configuration
├── pyproject.toml          # Poetry dependencies and tools definition
└── README.md               # Main project documentation
```

### Empty Packages (Awaiting Phase 1+ Code)
Per Phase 0 constraints, the following directories only contain an empty package marker `__init__.py` and will receive code in later phases:
*   `src/ingestion/`
*   `src/features/`
*   `src/fusion/`
*   `src/models/`
*   `src/explainability/`
*   `src/eval/`

---

## Installation & Setup

### Prerequisites
*   Python 3.11
*   Poetry (install via `curl -sSL https://install.python-poetry.org | python3 -` or homebrew)

### 1. Install Dependencies
Run the installation command to set up the virtual environment:
```bash
poetry install
```

### 2. Set Up Pre-commit Hooks
Register pre-commit hooks to automate formatting (`black`) and linting (`ruff`) on commit:
```bash
poetry run pre-commit install
```

### 3. Verify Scaffolding
Verify that the tests compile and run green:
```bash
poetry run pytest
```

---

## Data Acquisition & Validation
Detailed download instructions can be found in [data/raw/README.md](file:///Users/janhavi/deal-intelligence/data/raw/README.md).

Once you manually place the files, validate their presence and location by running:
```bash
poetry run python scripts/download_data.py
```

---

## Data Ingestion & Features Pipeline

To run the ingestion pipeline to parse Enron emails, link pseudo-deals with CRM metadata, and save timeline JSONs:
```bash
poetry run python src/ingestion/pipeline.py
```

### Fast Debugging (Max Deals Limit)
For faster iterations on a subset of the dataset, you can limit the pipeline to process only the first $N$ deals:
* **Via command-line option:**
  ```bash
  poetry run python src/ingestion/pipeline.py --max-deals 10
  ```
* **Via configuration file (`configs/data.yaml`):**
  Set `max_deals_debug: 10`

---

## Known Limitations

### 1. Synthetic Deal-Linking
> [!IMPORTANT]
> **Enron emails and HubSpot Kaggle CRM data are not natively linked by deal_id.**
> We use a deterministic pseudo-deal clustering heuristic (grouping emails by partner domain, subject similarity, and time gap), and then sample HubSpot CRM metadata to generate a synthetic deal profile. This limitation must be explicitly noted in system reports.

### 2. Behavioral Outcome Proxy (Label & Feature Leakage)
> [!WARNING]
> **The current "won/lost" label is a BEHAVIORAL PROXY, NOT a real sales outcome.**
> - **"won"** corresponds to active back-and-forth communication (at least one domain-switch and reply from an external party).
> - **"lost"** corresponds to a ghosted, one-way thread or a single-message email thread.
>
> This is a **meaningfully different and narrower claim** than the original goal of predicting validated business sales-loss outcomes and inferring causal sales-loss reasons. It should not be implied or treated as equivalent to genuine sales victory or defeat.
>
> **Leakage Sanity Check Results:**
> To verify what the LSTM model is learning, we trained a trivial baseline classifier (Logistic Regression) using **only two structural features** per deal sequence:
> 1. Total message count
> 2. Binary `has_external_reply` flag
>
> | Model | Validation AUC |
> |---|---|
> | Trivial Baseline Classifier | **0.9583** |
> | LSTM Sequence Encoder | **0.9676** |
>
> Because the trivial baseline achieves a validation AUC of **0.9583**, it **confirms that the LSTM is primarily learning to reconstruct the structural rule used to define the behavioral proxy label** rather than detecting complex semantic or sales-negotiation signals in the SBERT embeddings.
