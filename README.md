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

### 2. Synthetic Labeling Approach & Leakage Mitigation
> [!IMPORTANT]
> **Labeling Design and Sanity Check Findings:**
> Originally, Enron email content and HubSpot outcomes were independent. In a prior phase, we explored a **behavioral-proxy labeling approach** (defining a "won" deal based on communication structure: $\ge 2$ emails, at least one external reply, and at least one domain switch).
>
> However, a leakage sanity check revealed that a trivial classifier using **only two structural features** (total message count and presence of an external reply) achieved a **validation AUC of 0.9583**, virtually identical to the full LSTM sequence encoder's **0.9676 AUC**. This confirmed that the LSTM was not learning meaningful semantic signal from SBERT text embeddings, but was simply rediscovering the structural definition of the proxy label.
>
> **Mitigation:**
> To eliminate this label leakage, we **reverted to distribution-sampled synthetic labels** from HubSpot's real `deals.csv` dataset. The labels are sampled independently from HubSpot's stage/close statistics, ensuring the model trains on non-leaked targets (yielding a real, baseline AUC of **0.5613**). All advanced temporal features, content relevance filters, and LSTM attention pooling improvements have been retained as clean, non-leaking ML infrastructure.

### 3. Time-to-Close Regression Trivial Variance & Feature-Target Representation
> [!IMPORTANT]
> **Regression Diagnostic Findings & Target Leak Fix:**
> In Phase 4, the Time-to-Close regression head (XGBoost) achieved a low prediction error (MAE=`0.2583` days, RMSE=`0.6283` days). A diagnostic audit identified:
> *   **Trivial Target Variance:** The synthetic `days_to_close` target has a standard deviation of only `1.63` days and is heavily peaked (median=`3.0` days).
> *   **Velocity Feature Target Leak:** The stage velocity features (`velocity_prospecting`, `velocity_demo_scheduled`, etc.) sum up exactly to the target `days_to_close` for closed deals by definition (mean difference = `3.9e-17` days).
> *   **Exclusion Fix:** To resolve this structural leak, we modified the training pipeline to exclude all `velocity_*` features from the regression head.
> *   **Post-Exclusion Evaluation:** With velocities excluded, the XGBoost model achieves an MAE of `0.2585` days and RMSE of `0.6283` days, while the naive mean baseline achieves an MAE of `0.3284` days and RMSE of `0.6318` days. This head's low predictive value stems primarily from the near-constant target distribution in the synthetic labels, not from the (now-removed) structural leak, based on the trivial-baseline comparison already performed.
