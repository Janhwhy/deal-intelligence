# src/ingestion/timeline_builder.py: Builds chronologically ordered deal timelines validated against Pydantic models.

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# --- Pydantic Schema Definitions ---


class ContactModel(BaseModel):
    contact_id: int
    first_name: str
    last_name: str
    email: str
    phone: str
    job_title: str


class EmailMetadataModel(BaseModel):
    sender: str
    recipients: List[str]
    subject: str
    message_id: str


class EmailEventModel(BaseModel):
    timestamp: datetime
    type: Literal["email"] = "email"
    content: str
    metadata: EmailMetadataModel


class StageChangeMetadataModel(BaseModel):
    from_stage: Optional[str]
    to_stage: str


class StageChangeEventModel(BaseModel):
    timestamp: datetime
    type: Literal["stage_change"] = "stage_change"
    content: str
    metadata: StageChangeMetadataModel


class DealTimelineModel(BaseModel):
    deal_id: int
    stage: str
    outcome: str
    amount: float
    close_date: Optional[datetime] = None
    company_id: int
    company_name: str
    industry: str
    annual_revenue: float
    num_employees: int
    country: str
    contacts: List[ContactModel]
    events: List[Union[EmailEventModel, StageChangeEventModel]]


# --- Logic ---


def build_deal_timeline(deal_data: Dict[str, Any]) -> DealTimelineModel:
    """Consolidates emails and stage transitions of a deal into a sorted, validated timeline.

    Args:
        deal_data: A deal structure containing raw emails, stage transitions, and CRM data.

    Returns:
        A validated DealTimelineModel instance.
    """
    events: List[Union[EmailEventModel, StageChangeEventModel]] = []

    # Process email events
    for email_rec in deal_data["emails"]:
        events.append(
            EmailEventModel(
                timestamp=email_rec["timestamp"],
                type="email",
                content=email_rec["cleaned_body"],
                metadata=EmailMetadataModel(
                    sender=email_rec["sender"],
                    recipients=email_rec["recipients"],
                    subject=email_rec["subject"],
                    message_id=email_rec["message_id"],
                ),
            )
        )

    # Process stage transition events
    for trans in deal_data["stage_transitions"]:
        from_st = trans["from_stage"]
        to_st = trans["to_stage"]
        from_str = from_st if from_st else "Start"
        content_desc = f"CRM Stage changed from {from_str} to {to_st}"

        events.append(
            StageChangeEventModel(
                timestamp=trans["timestamp"],
                type="stage_change",
                content=content_desc,
                metadata=StageChangeMetadataModel(
                    from_stage=from_st,
                    to_stage=to_st,
                ),
            )
        )

    # Sort all events chronologically
    sorted_events = sorted(events, key=lambda x: x.timestamp)

    # Build primary timeline model
    timeline = DealTimelineModel(
        deal_id=deal_data["deal_id"],
        stage=deal_data["stage"],
        outcome=deal_data["outcome"],
        amount=deal_data["amount"],
        close_date=deal_data["close_date"],
        company_id=deal_data["company_id"],
        company_name=deal_data["company_name"],
        industry=deal_data["industry"],
        annual_revenue=deal_data["annual_revenue"],
        num_employees=deal_data["num_employees"],
        country=deal_data["country"],
        contacts=[ContactModel(**c) for c in deal_data["contacts"]],
        events=sorted_events,
    )

    return timeline


def save_deal_timeline(timeline: DealTimelineModel, output_dir: str) -> str:
    """Saves a DealTimelineModel as a JSON file in the output directory.

    Args:
        timeline: The validated DealTimelineModel to write.
        output_dir: Destination folder path.

    Returns:
        The absolute filepath where the timeline was written.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{timeline.deal_id}.json")

    # Serialize using Pydantic's JSON dumping capabilities
    # Pydantic v2 dump_model / model_dump_json handles datetimes/literal correctly
    json_str = timeline.model_dump_json(indent=2)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(json_str)

    logger.debug(f"Saved timeline for deal {timeline.deal_id} to {file_path}")
    return file_path
