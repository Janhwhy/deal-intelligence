# src/ingestion/email_parser.py: Parses raw Enron emails and cleans their body contents.

import email
import email.utils
import logging
import os
import re
from email.policy import default
from typing import Any, Dict, Generator, Optional

logger = logging.getLogger(__name__)


def clean_body(body: str) -> str:
    """Strips signature blocks, quoted replies, and forwarded messages from an email body.

    Args:
        body: The raw text body of the email.

    Returns:
        The cleaned body text.
    """
    if not body:
        return ""

    lines = body.splitlines()
    cleaned_lines = []

    # 1. Strip quoted reply headers and inline quotes
    for i, line in enumerate(lines):
        stripped = line.strip()
        lower_line = stripped.lower()

        # Check for original message markers (standard in Outlook/Enron mail clients)
        if (
            "-----original message-----" in lower_line
            or "----- original message -----" in lower_line
        ):
            break
        if "_____________________________________________" in lower_line:
            break
        if "-----forwarded by" in lower_line or "----- forwarded by" in lower_line:
            break

        # Check for inline header pattern starting with 'From:'
        if stripped.startswith("From:") and i < len(lines) - 2:
            subsequent = "".join(lines[i : i + 4])
            if "To:" in subsequent or "Sent:" in subsequent or "Subject:" in subsequent:
                break

        # Skip inline quotes starting with '>'
        if stripped.startswith(">"):
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines).rstrip()
    lines = text.splitlines()

    # 2. Heuristically strip trailing signature blocks
    # We examine the last 10 lines for common salutations or contact markers
    sig_start_idx = len(lines)
    for i in range(max(0, len(lines) - 10), len(lines)):
        stripped = lines[i].strip()
        lower_line = stripped.lower()

        # Standard signature delimiters
        if stripped == "--" or stripped == "---":
            sig_start_idx = i
            break

        # Common greetings/salutations that indicate signature transition
        salutations = {
            "thanks",
            "thanks,",
            "thank you",
            "thank you,",
            "regards",
            "regards,",
            "best regards",
            "best regards,",
            "best",
            "best,",
            "sincerely",
            "sincerely,",
            "cheers",
            "cheers,",
            "warmly",
            "warmly,",
        }
        if lower_line in salutations:
            sig_start_idx = i
            break

        # Contact detail indicators (e.g. "Phone: 123-456-7890")
        if re.search(r"\b(phone|tel|cell|fax|office|dir):\s*[\d\-()+\s]+", lower_line):
            sig_start_idx = min(sig_start_idx, i)
            break

    if sig_start_idx < len(lines):
        lines = lines[:sig_start_idx]

    return "\n".join(lines).strip()


def parse_email_file(file_path: str) -> Optional[Dict[str, Any]]:
    """Parses a single raw Enron email file into a normalized dictionary.

    Args:
        file_path: Absolute path to the raw email file.

    Returns:
        A dictionary of parsed fields or None if parsing fails.
    """
    try:
        with open(file_path, "rb") as f:
            msg = email.message_from_binary_file(f, policy=default)

        # Parse message ID, fallback to filepath basename if missing
        message_id = msg.get("message-id")
        if message_id:
            message_id = message_id.strip()
        else:
            message_id = f"gen-{os.path.basename(file_path)}"

        # Parse date and convert to datetime object
        date_str = msg.get("date")
        if not date_str:
            logger.warning(f"Skipping email without Date header: {file_path}")
            return None
        try:
            timestamp = email.utils.parsedate_to_datetime(date_str)
        except Exception as e:
            logger.warning(
                f"Skipping email due to unparseable Date header '{date_str}': {e}"
            )
            return None

        # Extract subject line
        subject = msg.get("subject", "")
        if subject:
            subject = subject.strip()
        else:
            subject = ""

        # Extract sender
        sender = msg.get("from", "")
        if sender:
            sender = sender.strip().lower()
        else:
            sender = ""

        # Extract recipients from To, Cc, Bcc
        recipients_list = []
        for header in ["to", "cc", "bcc"]:
            vals = msg.get_all(header, [])
            for name, addr in email.utils.getaddresses(vals):
                if addr:
                    recipients_list.append(addr.strip().lower())
        # Deduplicate list while preserving order
        recipients = list(dict.fromkeys(recipients_list))

        # Extract body text
        body = ""
        if msg.is_multipart():
            # Standard RFC 822 emails can have multiple parts
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disp = part.get("content-disposition", "")
                if content_type == "text/plain" and "attachment" not in content_disp:
                    try:
                        body = part.get_content()
                    except Exception:
                        # Fallback for decoding errors
                        payload = part.get_payload(decode=True)
                        if payload:
                            body = payload.decode("utf-8", errors="ignore")
                    break
        else:
            try:
                body = msg.get_content()
            except Exception:
                payload = msg.get_payload(decode=True)
                if payload:
                    body = payload.decode("utf-8", errors="ignore")

        # Clean body
        cleaned_body = clean_body(body)

        return {
            "message_id": message_id,
            "sender": sender,
            "recipients": recipients,
            "subject": subject,
            "timestamp": timestamp,
            "cleaned_body": cleaned_body,
        }
    except Exception as e:
        logger.error(f"Failed to parse email file {file_path}: {e}")
        return None


def crawl_enron_emails(enron_raw_dir: str) -> Generator[Dict[str, Any], None, None]:
    """Recursively walks the raw Enron directory and yields parsed email records.

    Args:
        enron_raw_dir: Root directory of raw Enron data.

    Yields:
        Parsed email dictionaries.
    """
    target_dir = enron_raw_dir
    maildir_path = os.path.join(enron_raw_dir, "maildir")
    if os.path.isdir(maildir_path):
        target_dir = maildir_path

    logger.info(f"Crawling Enron emails in: {target_dir}")
    count = 0
    for root, _, files in os.walk(target_dir):
        for file in files:
            # Enron mail files are typically numbers or strings, avoid hidden files
            if file.startswith("."):
                continue

            file_path = os.path.join(root, file)
            parsed = parse_email_file(file_path)
            if parsed:
                count += 1
                yield parsed

    logger.info(f"Completed crawling. Successfully parsed {count} email messages.")
