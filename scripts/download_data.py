# scripts/download_data.py: Validates manual raw data downloads.

import os
import sys

# Define base paths relative to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
ENRON_DIR = os.path.join(RAW_DATA_DIR, "enron")
HUBSPOT_DIR = os.path.join(RAW_DATA_DIR, "hubspot")

# Expected files/folders to validate dataset presence
# For Enron, we look for either the extracted 'maildir' directory or the raw tarball
ENRON_EXPECTED = ["maildir", "enron_mail_20150507.tar.gz"]
# For HubSpot, we expect the unzipped CSV export files (e.g., deals.csv, companies.csv)
HUBSPOT_EXPECTED = ["deals.csv", "companies.csv"]

ENRON_INSTRUCTIONS = """
--- Enron Email Corpus Download Instructions ---
1. Download the Enron email dataset from the official Carnegie Mellon University host:
   URL: https://www.cs.cmu.edu/~./enron/enron_mail_20150507.tar.gz (~1.7 GB compressed)
2. Extract the file or place the tarball in:
   Path: data/raw/enron/
3. If extracted, ensure the 'maildir' folder is located directly under data/raw/enron/
"""

HUBSPOT_INSTRUCTIONS = """
--- HubSpot Kaggle CRM Dataset Download Instructions ---
1. Download the HubSpot Kaggle CRM dataset CSVs (Kaggle account required):
   URL: https://www.kaggle.com/datasets/kagglesocial/hubspot-crm-dataset
2. Place the CSV export files (specifically 'deals.csv' and 'companies.csv') in:
   Path: data/raw/hubspot/
"""


def check_dataset(directory, expected_items, dataset_name, instructions):
    """Checks if any of the expected files/folders are present in the target directory.
    Creates the target directory if it does not exist.
    """
    os.makedirs(directory, exist_ok=True)

    # List contents of directory
    contents = os.listdir(directory)
    # Remove hidden files/folders (e.g. .DS_Store, .gitkeep)
    contents = [item for item in contents if not item.startswith(".")]

    if not contents:
        print(f"\n[ERROR] {dataset_name} dataset is MISSING.")
        print(
            f"Expected at least one of the following in {directory}: {expected_items}"
        )
        print(instructions)
        return False

    # Check if at least one of the expected items exists
    found_any = any(
        os.path.exists(os.path.join(directory, item)) for item in expected_items
    )
    if not found_any:
        print(
            f"\n[WARNING] {dataset_name} directory contains files, "
            "but none of the expected structure:"
        )
        print(f"Expected files/folders: {expected_items}")
        print(f"Found in directory: {contents}")
        print(instructions)
        return False

    print(f"[OK] Found expected files for {dataset_name} in {directory}")
    return True


def main():
    print("Validating raw data directories...")

    enron_ok = check_dataset(
        ENRON_DIR, ENRON_EXPECTED, "Enron Email Corpus", ENRON_INSTRUCTIONS
    )
    hubspot_ok = check_dataset(
        HUBSPOT_DIR, HUBSPOT_EXPECTED, "HubSpot Kaggle CRM", HUBSPOT_INSTRUCTIONS
    )

    if not enron_ok or not hubspot_ok:
        print(
            "\nValidation failed: one or more datasets are missing "
            "or incorrectly placed."
        )
        sys.exit(1)

    print("\nAll raw datasets verified successfully!")
    sys.exit(0)


if __name__ == "__main__":
    main()
