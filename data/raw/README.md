# data/raw/README.md: Data acquisition documentation for raw sales communications and CRM datasets.

# Raw Data Acquisition

This directory holds the raw dataset inputs required by the Deal Intelligence Engine. Both datasets require manual download steps.

---

## 1. Enron Email Corpus

The Enron email corpus provides realistic sales communication records (emails, attachments, and metadata).

*   **Source URL:** [https://www.cs.cmu.edu/~./enron/](https://www.cs.cmu.edu/~./enron/)
*   **Download File:** `enron_mail_20150507.tar.gz`
*   **Approximate Compressed Size:** ~1.7 GB
*   **Approximate Uncompressed Size:** ~4.0 GB
*   **Destination Path:** `data/raw/enron/`
*   **Unpacking Instruction:** Extract the tarball inside `data/raw/enron/`. The root directory must contain the `maildir/` folder structure containing individual user folders (e.g., `data/raw/enron/maildir/lay-k/`).

---

## 2. HubSpot Kaggle CRM Dataset

The HubSpot CRM dataset provides structural sales activity records, specifically deal pipeline stages, close status, companies, and timeline events.

*   **Source URL:** [https://www.kaggle.com/datasets/kagglesocial/hubspot-crm-dataset](https://www.kaggle.com/datasets/kagglesocial/hubspot-crm-dataset)
*   **Download Files:** `deals.csv` and `companies.csv`
*   **Approximate Size:** ~10 MB
*   **Destination Path:** `data/raw/hubspot/`

---

## CRITICAL NOTE: Deal-Linking Limitation

> [!WARNING]
> **Known Limitation — Deal-Linking:**
> The Enron email corpus and HubSpot Kaggle CRM dataset are completely separate datasets and are **not natively linked by any standard identifier** (such as `deal_id` or common contact emails).
>
> **Phase 1 Strategy:**
> To bridge these sources, Phase 1 will require designing and implementing a documented synthetic-linking strategy (e.g., clustering email threads by domain name counterparties, mapping timeline event timestamps to CRM entries, and injecting synthetic `deal_id` metadata into the datasets). This is a known dataset limitation that must be explicitly discussed in the evaluation and limitation section of the final system reports.
