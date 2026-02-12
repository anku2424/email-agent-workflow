"""Email processing agents using OpenAI Agents SDK with Ollama."""
from pydantic import BaseModel
from agents import Agent, function_tool
from config import get_model, NOTIFICATION_EMAIL
from gmail_tools import send_email


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
        instructions="""You are an email triage specialist. Analyze incoming emails and decide how to handle them.

For each email, determine:
1. ACTION - One of:
   - "respond": Email requires a reply (questions, requests, important communications)
   - "ignore": No action needed (newsletters, promotions, automated notifications)
   - "flag_for_human": Complex/sensitive matters requiring human judgment

2. URGENCY - Rate as "high", "medium", or "low" based on:
   - Deadlines mentioned
   - Sender importance
   - Content criticality

3. CATEGORY - Classify as: business, personal, spam, newsletter, support, inquiry, etc.

Be concise but thorough in your reasoning. When in doubt about sensitive topics (legal, financial, emotional), flag for human review.""",
        model=get_model(),
        output_type=TriageResult,
    )


def create_draft_agent() -> Agent:
    """Create the Draft Agent that generates email replies."""
    return Agent(
        name="Draft Agent",
        instructions="""You are a professional email writer. Generate clear, helpful email replies.

Guidelines:
1. Match the tone to the original email (formal for business, friendly for personal)
2. Address all questions/points raised in the original email
3. Be concise but complete - avoid unnecessary filler
4. Use appropriate greetings and sign-offs
5. If the email is a question, provide a helpful answer
6. If it's a request, acknowledge and respond appropriately
7. Never make up information - if unsure, suggest the person contact support

Keep replies professional and to the point.""",
        model=get_model(),
        output_type=DraftResult,
    )


def create_quality_checker_agent() -> Agent:
    """Create the Quality Checker Agent that reviews drafts."""
    return Agent(
        name="Quality Checker",
        instructions="""You are a quality assurance specialist for email communications.

Review email drafts for:
1. TONE - Is it appropriate for the context? Professional enough?
2. ACCURACY - Does it address the original email's concerns?
3. CLARITY - Is it easy to understand? No ambiguity?
4. COMPLETENESS - Are all points addressed?
5. SENSITIVITY - Any content that could be misunderstood or offensive?

Flag for human review if:
- The topic is legally sensitive
- Financial commitments are mentioned
- The tone seems off or potentially offensive
- Complex technical accuracy can't be verified
- The original email was emotionally charged

Provide a quality score (1-10):
- 8-10: Ready to send
- 5-7: Minor issues, can send with revisions
- 1-4: Needs human review

If approved, you may make minor edits to improve the draft.""",
        model=get_model(),
        output_type=QualityResult,
        tools=[send_flagged_notification],
    )
