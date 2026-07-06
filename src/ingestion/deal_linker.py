# src/ingestion/deal_linker.py: Links parsed emails into pseudo-deals and associates them with synthetic CRM metadata.

import os
import re
import logging
import difflib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

from src.config import DataConfig

logger = logging.getLogger(__name__)


def normalize_subject(subj: str) -> str:
    """Normalizes an email subject by stripping prefixes and non-alphanumeric characters.

    Args:
        subj: Raw email subject line.

    Returns:
        Normalized subject line.
    """
    s = subj.lower().strip()
    while True:
        changed = False
        for prefix in ["re:", "fw:", "fwd:"]:
            if s.startswith(prefix):
                s = s[len(prefix):].strip()
                changed = True
        if not changed:
            break
    # Remove non-alphanumeric characters but keep spaces
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return " ".join(s.split())


def get_partner_domain(sender: str, recipients: List[str]) -> str:
    """Identifies the primary external partner domain from an email's participants.

    Heuristic:
    - Collects all domains from sender and recipients.
    - Excludes 'enron.com' if other domains exist to avoid grouping internal emails.
    - Returns the alphabetically first remaining domain.

    Args:
        sender: Sender email address.
        recipients: List of recipient email addresses.

    Returns:
        The partner domain string.
    """
    domains = set()
    for addr in [sender] + recipients:
        if "@" in addr:
            domain = addr.split("@")[-1].strip().lower()
            if domain:
                domains.add(domain)

    if not domains:
        return "unknown"

    non_enron = {d for d in domains if d != "enron.com"}
    if non_enron:
        return sorted(list(non_enron))[0]
    return "enron.com"


def cluster_emails(
    emails: List[Dict[str, Any]],
    similarity_threshold: float,
    window_days: int,
) -> List[List[Dict[str, Any]]]:
    """Clusters emails into pseudo-deals based on partner domain, subject similarity, and time.

    Args:
        emails: List of parsed email dictionaries.
        similarity_threshold: Subject line similarity threshold (0.0 to 1.0).
        window_days: Maximum days allowed between consecutive emails in a cluster.

    Returns:
        A list of email clusters, where each cluster is a list of email dicts.
    """
    # Group emails by partner domain
    domain_groups: Dict[str, List[Dict[str, Any]]] = {}
    for email_rec in emails:
        domain = get_partner_domain(email_rec["sender"], email_rec["recipients"])
        domain_groups.setdefault(domain, []).append(email_rec)

    clusters: List[List[Dict[str, Any]]] = []

    # Sort each domain group by timestamp and perform gap-based clustering
    for domain, group in domain_groups.items():
        sorted_group = sorted(group, key=lambda x: x["timestamp"])

        # Active clusters for this domain
        active_clusters: List[Dict[str, Any]] = []

        for email_rec in sorted_group:
            normalized_subj = normalize_subject(email_rec["subject"])
            ts = email_rec["timestamp"]

            best_cluster_idx = -1
            best_similarity = -1.0

            for i, cl in enumerate(active_clusters):
                # Check time proximity
                time_diff = ts - cl["last_timestamp"]
                if time_diff > timedelta(days=window_days):
                    continue

                # Check subject similarity
                sim = difflib.SequenceMatcher(
                    None, normalized_subj, cl["rep_subject"]
                ).ratio()
                if sim >= similarity_threshold:
                    if sim > best_similarity:
                        best_similarity = sim
                        best_cluster_idx = i

            if best_cluster_idx != -1:
                # Add to existing cluster
                active_clusters[best_cluster_idx]["emails"].append(email_rec)
                active_clusters[best_cluster_idx]["last_timestamp"] = ts
            else:
                # Create a new cluster
                active_clusters.append(
                    {
                        "rep_subject": normalized_subj,
                        "last_timestamp": ts,
                        "emails": [email_rec],
                    }
                )

        for cl in active_clusters:
            clusters.append(cl["emails"])

    return clusters


def link_deals(
    emails: List[Dict[str, Any]], data_config: DataConfig
) -> List[Dict[str, Any]]:
    """Clusters emails and links them with sampled CRM data.

    ASSUMPTION:
    The Enron emails and HubSpot CSV datasets are independent and not natively linked.
    This module uses a deterministic pseudo-deal clustering heuristic, then draws
    real CRM distributions (stage, close_date, company, contact details) by sampling
    rows from HubSpot data to synthesize realistic B2B deal profiles. This is a synthetic
    construction and should be documented as a limitation in evaluation reports.

    Args:
        emails: List of parsed email dictionaries.
        data_config: Application configuration containing data paths and parameters.

    Returns:
        A list of synthesized pseudo-deal structures.
    """
    # 1. Cluster emails into pseudo-deals
    raw_clusters = cluster_emails(
        emails,
        data_config.subject_similarity_threshold,
        data_config.time_proximity_window_days,
    )

    if not raw_clusters:
        logger.warning("No email clusters formed. Check raw email inputs.")
        return []

    logger.info(f"Formed {len(raw_clusters)} raw email clusters.")

    # 2. Load HubSpot CSV databases
    try:
        deals_df = pd.read_csv(data_config.hubspot_deals_csv)
        companies_df = pd.read_csv(data_config.hubspot_companies_csv)
        contacts_df = pd.read_csv(data_config.hubspot_contacts_csv)
    except Exception as e:
        logger.error(f"Error loading HubSpot CSV files: {e}")
        raise

    # Initialize deterministic random number generator
    rng = np.random.default_rng(data_config.deal_linker_seed)

    processed_deals: List[Dict[str, Any]] = []

    # Stage progression logic
    stage_sequences = {
        "Prospecting": ["Prospecting"],
        "Demo Scheduled": ["Prospecting", "Demo Scheduled"],
        "Negotiation": ["Prospecting", "Demo Scheduled", "Negotiation"],
        "Closed Won": ["Prospecting", "Demo Scheduled", "Negotiation", "Closed Won"],
        "Closed Lost": ["Prospecting", "Demo Scheduled", "Negotiation", "Closed Lost"],
    }

    # Generate statistic variables
    sizes = []

    for idx, cluster in enumerate(raw_clusters):
        deal_id = idx + 1
        sizes.append(len(cluster))

        # Sample a deal row from deals.csv
        deal_idx = rng.choice(len(deals_df))
        sampled_deal = deals_df.iloc[deal_idx]

        # Extract basic deal attributes
        stage = str(sampled_deal["stage"])
        amount = float(sampled_deal["amount"])

        # Determine outcome from stage
        if stage == "Closed Won":
            outcome = "won"
        elif stage == "Closed Lost":
            outcome = "lost"
        else:
            outcome = "open"

        # Lookup company information
        company_id = sampled_deal["company_id"]
        company_rows = companies_df[companies_df["company_id"] == company_id]
        if not company_rows.empty:
            company_info = company_rows.iloc[0]
        else:
            # Fallback
            company_info = companies_df.iloc[rng.choice(len(companies_df))]
            company_id = int(company_info["company_id"])

        company_name = str(company_info["company_name"])
        industry = str(company_info["industry"])
        annual_revenue = float(company_info["annual_revenue"])
        num_employees = int(company_info["num_employees"])
        country = str(company_info["country"])

        # Lookup and associate contacts
        candidate_contacts = contacts_df[contacts_df["company_id"] == company_id]
        if candidate_contacts.empty:
            candidate_contacts = contacts_df

        # Identify unique email addresses in the cluster thread
        unique_emails = set()
        for e in cluster:
            unique_emails.add(e["sender"])
            unique_emails.update(e["recipients"])

        # Create mappings for participants to contacts.csv records
        contacts_list: List[Dict[str, Any]] = []
        unique_emails_sorted = sorted(list(unique_emails))

        # Check if the primary deal contact exists
        primary_contact_id = sampled_deal["contact_id"]
        primary_rows = contacts_df[contacts_df["contact_id"] == primary_contact_id]
        primary_contact = primary_rows.iloc[0] if not primary_rows.empty else None

        for p_idx, p_email in enumerate(unique_emails_sorted):
            # Attempt to associate first participant with primary contact, if valid
            if p_idx == 0 and primary_contact is not None:
                contact_row = primary_contact
            else:
                contact_row = candidate_contacts.iloc[
                    rng.choice(len(candidate_contacts))
                ]

            contacts_list.append(
                {
                    "contact_id": int(contact_row["contact_id"]),
                    "first_name": str(contact_row["first_name"]),
                    "last_name": str(contact_row["last_name"]),
                    "email": str(contact_row["email"]),
                    "phone": str(contact_row["phone"]),
                    "job_title": str(contact_row["job_title"]),
                }
            )

        # Generate stage transitions distributed along the email timeline
        t_start = min(e["timestamp"] for e in cluster)
        t_end = max(e["timestamp"] for e in cluster)

        stages = stage_sequences.get(stage, ["Prospecting"])
        K = len(stages)
        stage_transitions: List[Dict[str, Any]] = []

        total_seconds = (t_end - t_start).total_seconds()
        if total_seconds <= 0:
            t_stages = [t_start + timedelta(days=i) for i in range(K)]
        else:
            fractions = sorted(
                [0.0] + [float(rng.uniform(0.1, 0.9)) for _ in range(K - 2)] + [1.0]
            )
            t_stages = [
                t_start + timedelta(seconds=int(frac * total_seconds))
                for frac in fractions
            ]

        # Form transitions
        for i in range(K):
            from_stage = stages[i - 1] if i > 0 else None
            to_stage = stages[i]
            stage_transitions.append(
                {
                    "timestamp": t_stages[i],
                    "from_stage": from_stage,
                    "to_stage": to_stage,
                }
            )

        # Align close date to the final stage transition timestamp (if closed)
        close_date = t_stages[-1] if stage in ["Closed Won", "Closed Lost"] else None

        processed_deals.append(
            {
                "deal_id": deal_id,
                "stage": stage,
                "outcome": outcome,
                "amount": amount,
                "close_date": close_date,
                "company_id": company_id,
                "company_name": company_name,
                "industry": industry,
                "annual_revenue": annual_revenue,
                "num_employees": num_employees,
                "country": country,
                "contacts": contacts_list,
                "emails": cluster,
                "stage_transitions": stage_transitions,
            }
        )

    # Log statistics about the clusters
    if sizes:
        sizes_arr = np.array(sizes)
        logger.info(f"Deal linking finished. Linked {len(processed_deals)} deals.")
        logger.info(
            f"Deal email size stats - Min: {sizes_arr.min()}, Max: {sizes_arr.max()}, "
            f"Mean: {sizes_arr.mean():.2f}, Median: {np.median(sizes_arr):.2f}"
        )

    return processed_deals
