"""Email processing agents using OpenAI Agents SDK with Ollama."""
from pathlib import Path
from pydantic import BaseModel
from agents import Agent, function_tool, ModelSettings
from config import get_model, NOTIFICATION_EMAIL
from gmail_tools import send_email


# ============ Prompt Loading ============

PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str) -> str:
    """Load an agent prompt from the prompts directory."""
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").strip()


# ============ Output Schemas ============

class TriageResult(BaseModel):
    """Output schema for the Triage Agent."""
    action: str  # "respond", "ignore", or "flag_for_human"
    urgency: str  # "high", "medium", "low"
    category: str  # e.g., "business", "personal", "spam", "newsletter"
    reasoning: str  # Brief explanation of the decision
    summary: str  # One-line summary of the email


class DraftResult(BaseModel):
    """Output schema for the Draft Agent."""
    subject: str  # Reply subject line
    body: str  # The drafted reply
    tone: str  # e.g., "formal", "casual", "friendly"
    key_points_addressed: list[str]  # What the reply covers


class QualityResult(BaseModel):
    """Output schema for the Quality Checker Agent."""
    approved: bool  # Whether the draft is ready to send
    needs_human_review: bool  # Flag for human review
    issues: list[str]  # Any issues found
    suggestions: list[str]  # Improvement suggestions
    final_draft: str  # The final (possibly revised) draft
    quality_score: int  # 1-10 rating


# ============ Function Tools ============

@function_tool
def send_flagged_notification(
    original_subject: str,
    original_sender: str,
    reason: str,
    email_summary: str
) -> str:
    """Send a notification email about a flagged email that needs human review."""
    notification_body = f"""
FLAGGED EMAIL NOTIFICATION
==========================

An email has been flagged for your review.

Original Email Details:
- From: {original_sender}
- Subject: {original_subject}

Summary:
{email_summary}

Reason for Flagging:
{reason}

Please review this email in your inbox and respond manually.
"""
    
    result = send_email(
        to=NOTIFICATION_EMAIL,
        subject=f"FLAGGED EMAIL: {original_subject}",
        body=notification_body
    )
    return f"Notification sent: {result}"


@function_tool
def send_reply_email(
    to: str,
    subject: str,
    body: str,
    original_message_id: str
) -> str:
    """Send a reply email."""
    result = send_email(
        to=to,
        subject=subject,
        body=body,
        reply_to_message_id=original_message_id
    )
    return f"Reply sent: {result}"


# ============ Agent Definitions ============

def create_triage_agent() -> Agent:
    """Create the Triage Agent that analyzes incoming emails."""
    return Agent(
        name="Triage Agent",
        instructions=load_prompt("triage_agent.txt"),
        model=get_model(),
        model_settings=ModelSettings(extra_body={"reasoning_effort": "none"}),
        output_type=TriageResult,
    )


def create_draft_agent() -> Agent:
    """Create the Draft Agent that generates email replies."""
    return Agent(
        name="Draft Agent",
        instructions=load_prompt("draft_agent.txt"),
        model=get_model(),
        model_settings=ModelSettings(extra_body={"reasoning_effort": "none"}),
        output_type=DraftResult,
    )


def create_quality_checker_agent() -> Agent:
    """Create the Quality Checker Agent that reviews drafts."""
    return Agent(
        name="Quality Checker",
        instructions=load_prompt("quality_checker_agent.txt"),
        model=get_model(),
        model_settings=ModelSettings(extra_body={"reasoning_effort": "none"}),
        output_type=QualityResult,
        tools=[send_flagged_notification],
    )
