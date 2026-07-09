# src/ingestion/pipeline.py: Ingestion pipeline execution CLI for deal intelligence.

import logging
import os
import sys

from src.config import load_config
from src.ingestion.deal_linker import link_deals
from src.ingestion.email_parser import crawl_enron_emails
from src.ingestion.timeline_builder import build_deal_timeline, save_deal_timeline

# Configure logging to write to console and a pipeline log file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def run_ingestion_pipeline() -> None:
    """Runs the end-to-end ingestion pipeline.

    Steps:
    1. Loads configuration from YAML files.
    2. Crawls and parses raw Enron emails.
    3. Runs the deterministic synthetic linking engine with CRM data.
    4. Generates and validates timeline documents for each pseudo-deal.
    5. Saves timeline documents to data/processed/deals/ directory.
    """
    logger.info("Initializing ingestion pipeline...")

    import argparse

    parser = argparse.ArgumentParser(description="Ingestion pipeline CLI")
    parser.add_argument(
        "config_dir", nargs="?", default=None, help="Directory containing config files"
    )
    parser.add_argument(
        "--max-deals",
        type=int,
        default=None,
        help="Limit processing to the first N deals",
    )
    args = parser.parse_args()

    config_dir = args.config_dir
    try:
        config = load_config(config_dir=config_dir)
        data_cfg = config.data
    except Exception as e:
        logger.error(f"Failed to load configurations: {e}")
        sys.exit(1)

    # Override config key with CLI argument if provided
    if args.max_deals is not None:
        data_cfg.max_deals_debug = args.max_deals

    logger.info(f"Loaded configuration. Enron path: {data_cfg.enron_raw_dir}")

    # Step 1: Parse emails
    if not os.path.exists(data_cfg.enron_raw_dir):
        logger.error(f"Enron raw directory does not exist: {data_cfg.enron_raw_dir}")
        sys.exit(1)

    logger.info("Parsing raw Enron emails...")
    emails = list(
        crawl_enron_emails(
            data_cfg.enron_raw_dir, max_deals_debug=data_cfg.max_deals_debug
        )
    )
    if not emails:
        logger.warning("No emails parsed. Exiting pipeline.")
        sys.exit(0)

    logger.info(f"Parsed {len(emails)} emails. Proceeding to linking...")

    # Step 2: Link emails to CRM data (pseudo-deals)
    logger.info("Linking emails to synthetic CRM profiles...")
    try:
        deals = link_deals(emails, data_cfg)
    except Exception as e:
        logger.error(f"Failed during deal linking stage: {e}")
        sys.exit(1)

    # Step 3: Build timelines and save to disk
    logger.info("Building and saving deal timeline JSONs...")
    save_count = 0
    for deal_data in deals:
        if (
            data_cfg.max_deals_debug is not None
            and save_count >= data_cfg.max_deals_debug
        ):
            logger.info(
                f"Reached max_deals_debug limit of {data_cfg.max_deals_debug}. Stopping timeline creation."
            )
            break
        try:
            timeline = build_deal_timeline(deal_data)
            save_deal_timeline(timeline, data_cfg.processed_deals_dir)
            save_count += 1
        except Exception as e:
            logger.error(
                f"Validation or write failure for deal {deal_data['deal_id']}: {e}"
            )
            # Fail loudly on invalid data as requested
            raise

    logger.info(
        f"Pipeline finished successfully. Saved {save_count} deal timelines to {data_cfg.processed_deals_dir}"
    )


if __name__ == "__main__":
    run_ingestion_pipeline()
