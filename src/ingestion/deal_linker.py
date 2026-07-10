# src/ingestion/deal_linker.py: Links parsed emails into pseudo-deals and associates them with synthetic CRM metadata.

import difflib
import logging
import os
import re
from datetime import timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

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
                s = s[len(prefix) :].strip()
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


def load_deal_relevance_keywords(path: str) -> List[str]:
    """Loads a list of deal-relevance keywords/phrases from a flat text file.

    Lines starting with '#' and blank lines are ignored. Keywords are stripped
    and lowercased for case-insensitive matching.

    Args:
        path: Path to the keywords text file.

    Returns:
        A list of cleaned keyword strings.
    """
    fallback = [
        "proposal",
        "contract",
        "pricing",
        "quote",
        "purchase",
        "agreement",
        "nda",
        "invoice",
        "bid",
        "rfp",
    ]
    if not os.path.exists(path):
        logger.warning(
            f"Deal relevance keywords file not found at '{path}'. Using built-in fallback list."
        )
        return fallback

    with open(path, "r", encoding="utf-8") as f:
        keywords = [
            line.strip().lower()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

    if not keywords:
        logger.warning(
            f"Deal relevance keywords file at '{path}' is empty. Using built-in fallback list."
        )
        return fallback

    logger.info(f"Loaded {len(keywords)} deal relevance keywords from '{path}'.")
    return keywords


def is_external_email(sender: str, recipients: List[str]) -> bool:
    """Returns True if the email is between parties from different domains.

    Internal-only emails (all participants from the same domain, typically
    '@enron.com') are considered not deal-relevant since they do not represent
    communications with an external customer or vendor.

    Args:
        sender: Sender email address string.
        recipients: List of recipient email address strings.

    Returns:
        True if at least one recipient is from a different domain than the sender.
    """
    if "@" not in sender:
        return False
    sender_domain = sender.split("@")[-1].strip().lower()
    for recipient in recipients:
        if "@" in recipient:
            rcpt_domain = recipient.split("@")[-1].strip().lower()
            if rcpt_domain != sender_domain:
                return True
    return False


def is_deal_relevant(email_rec: Dict[str, Any], keywords: List[str]) -> bool:
    """Returns True if the email's subject or body contains any deal-relevance keyword.

    Matching is case-insensitive substring search against the combined subject
    and body content.

    Args:
        email_rec: Parsed email dictionary with 'subject' and 'content' keys.
        keywords: List of lowercase keyword strings to match against.

    Returns:
        True if any keyword is found in the subject or content.
    """
    subject = (email_rec.get("subject") or "").lower()
    content = (email_rec.get("content") or "").lower()
    combined = subject + " " + content
    return any(kw in combined for kw in keywords)


def cluster_emails(
    emails: List[Dict[str, Any]],
    similarity_threshold: float,
    window_days: int,
    max_deals_debug: Optional[int] = None,
) -> List[List[Dict[str, Any]]]:
    """Clusters emails into pseudo-deals based on partner domain, subject similarity, and time.

    Args:
        emails: List of parsed email dictionaries.
        similarity_threshold: Subject line similarity threshold (0.0 to 1.0).
        window_days: Maximum days allowed between consecutive emails in a cluster.
        max_deals_debug: Optional limit on the number of clusters to form.

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
        if max_deals_debug is not None and len(clusters) >= max_deals_debug:
            break

        sorted_group = sorted(group, key=lambda x: x["timestamp"])

        # Active clusters for this domain
        active_clusters: List[Dict[str, Any]] = []

        for email_rec in sorted_group:
            # Stop early if we have already formed enough clusters
            if (
                max_deals_debug is not None
                and (len(clusters) + len(active_clusters)) >= max_deals_debug
            ):
                for cl in active_clusters:
                    if len(clusters) >= max_deals_debug:
                        break
                    clusters.append(cl["emails"])
                active_clusters = []
                break

            normalized_subj = normalize_subject(email_rec["subject"])
            ts = email_rec["timestamp"]

            # Prune active clusters that have expired (last_timestamp is too old)
            # Since sorted_group is chronologically sorted, any cluster that is expired now
            # will remain expired for all subsequent emails in sorted_group.
            still_active = []
            for cl in active_clusters:
                if ts - cl["last_timestamp"] <= timedelta(days=window_days):
                    still_active.append(cl)
                else:
                    clusters.append(cl["emails"])
            active_clusters = still_active

            best_cluster_idx = -1
            best_similarity = -1.0

            # 1. Pre-check: Exact match check is extremely fast and very common
            for i, cl in enumerate(active_clusters):
                if normalized_subj == cl["rep_subject"]:
                    best_cluster_idx = i
                    best_similarity = 1.0
                    break

            # 2. Heuristic check: Fallback to SequenceMatcher only if lengths are comparable
            if best_cluster_idx == -1:
                len_subj = len(normalized_subj)
                for i, cl in enumerate(active_clusters):
                    len_rep = len(cl["rep_subject"])
                    # SequenceMatcher ratio has upper bound: 2 * min(L1, L2) / (L1 + L2)
                    max_possible_ratio = (2.0 * min(len_subj, len_rep)) / max(
                        1, len_subj + len_rep
                    )
                    if max_possible_ratio < similarity_threshold:
                        continue

                    # Check subject similarity (time proximity is guaranteed by pruning above)
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
            if max_deals_debug is not None and len(clusters) >= max_deals_debug:
                break
            clusters.append(cl["emails"])

    return clusters


def compute_behavioral_outcome(cluster: List[Dict[str, Any]]) -> str:
    """Derives a proxy outcome label from observable email thread engagement.

    Instead of randomly assigning HubSpot outcomes (which have no causal
    connection to the email content), this function derives a label from
    measurable communication behavior within the cluster:

    - "won"  (positive engagement): thread has ≥2 messages AND at least one
      message was sent by the external party (they replied, not just received)
      AND there is at least one domain-switch (genuine back-and-forth).
    - "lost" (no engagement / ghosted): Enron only sent messages with no
      external reply, or the cluster is a single isolated message.

    These labels are causally tied to the email content — engaged threads
    look structurally different from ghosted ones — giving the LSTM real
    signal to learn from.

    Args:
        cluster: List of parsed email dicts in the cluster.

    Returns:
        "won" or "lost".
    """
    if len(cluster) < 2:
        return "lost"

    enron_senders = 0
    external_senders = 0
    prev_domain: Optional[str] = None
    domain_switches = 0

    sorted_cluster = sorted(cluster, key=lambda x: x["timestamp"])
    for email_rec in sorted_cluster:
        sender = email_rec.get("sender", "")
        if "@" in sender:
            domain = sender.split("@")[-1].strip().lower()
            if domain == "enron.com":
                enron_senders += 1
            else:
                external_senders += 1

            if prev_domain is not None and domain != prev_domain:
                domain_switches += 1
            prev_domain = domain

    # Genuine back-and-forth: external party replied at least once
    if external_senders >= 1 and enron_senders >= 1 and domain_switches >= 1:
        return "won"

    return "lost"


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

    Content Relevance Filter:
    Before clustering, emails are screened for (1) external correspondence —
    sender and at least one recipient from different domains — and (2) deal-relevant
    keywords in subject or body. Only emails passing both filters are clustered.
    This prevents internal Enron admin chatter from polluting pseudo-deal timelines.

    Args:
        emails: List of parsed email dictionaries.
        data_config: Application configuration containing data paths and parameters.

    Returns:
        A list of synthesized pseudo-deal structures.
    """
    n_raw = len(emails)

    # 1a. Load deal-relevance keywords
    keywords = load_deal_relevance_keywords(data_config.deal_relevance_keywords_path)

    # 1b. Apply external-domain filter
    external_emails = [
        e for e in emails if is_external_email(e["sender"], e["recipients"])
    ]
    n_external = len(external_emails)
    logger.info(
        f"Content filter: {n_raw} raw emails → {n_external} after external-domain filter "
        f"({n_raw - n_external} internal-only removed)."
    )

    # 1c. Apply keyword relevance filter
    relevant_emails = [e for e in external_emails if is_deal_relevant(e, keywords)]
    n_relevant = len(relevant_emails)
    logger.info(
        f"Content filter: {n_external} external emails → {n_relevant} after keyword relevance filter "
        f"({n_external - n_relevant} non-deal-relevant removed)."
    )

    if n_relevant == 0:
        logger.warning(
            "No emails survived the content relevance filter. "
            "The corpus may not contain enough external deal-relevant communications. "
            "Consider broadening the keyword list or removing filters."
        )
        return []

    # 1d. Cluster filtered emails into pseudo-deals
    raw_clusters = cluster_emails(
        relevant_emails,
        data_config.subject_similarity_threshold,
        data_config.time_proximity_window_days,
        max_deals_debug=data_config.max_deals_debug,
    )

    if not raw_clusters:
        logger.warning(
            "No email clusters formed from filtered emails. Check filter settings."
        )
        return []

    logger.info(
        f"Formed {len(raw_clusters)} raw email clusters from {n_relevant} filtered emails."
    )

    # Warn if pseudo-deal count is too low for a meaningful signal check
    if len(raw_clusters) < 200:
        logger.warning(
            f"WARNING: Only {len(raw_clusters)} pseudo-deals formed after filtering — "
            "below the recommended minimum of 200 for statistically meaningful validation. "
            "This is an important finding: the Enron corpus may be unsuitable for this task "
            "as configured. Consider expanding the keyword list or removing the domain filter."
        )

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
