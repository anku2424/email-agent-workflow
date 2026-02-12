"""
Email Agent Workflow using OpenAI Agents SDK with Ollama (qwen3:8b)

Workflow:
1. Fetch new emails from Gmail
2. Triage Agent analyzes each email
3. Draft Agent creates reply (if needed)
4. Quality Checker reviews and approves/flags
"""
import asyncio
import json
from typing import Optional
from agents import Runner, set_tracing_disabled

from config import NOTIFICATION_EMAIL
from gmail_tools import get_unread_emails, send_email, create_draft, mark_as_read
from agents_workflow import (
    create_triage_agent,
    create_draft_agent,
    create_quality_checker_agent,
    TriageResult,
    DraftResult,
    QualityResult,
)

# Disable tracing for local Ollama models
set_tracing_disabled(True)


async def process_email(email: dict) -> dict:
    """Process a single email through the agent workflow."""
    print(f"\n{'='*60}")
    print(f"Processing: {email['subject']}")
    print(f"From: {email['sender']}")
    print(f"{'='*60}")
    
    result = {
        "email_id": email["id"],
        "subject": email["subject"],
        "sender": email["sender"],
        "triage": None,
        "draft": None,
        "quality": None,
        "final_action": None,
    }
    
    # ============ Step 1: Triage ============
    print("\n📋 Step 1: Triage Agent analyzing email...")
    
    triage_agent = create_triage_agent()
    triage_input = f"""
Analyze this email:

FROM: {email['sender']}
SUBJECT: {email['subject']}
DATE: {email['date']}

BODY:
{email['body']}
"""
    
    triage_result = await Runner.run(triage_agent, triage_input)
    triage_output: TriageResult = triage_result.final_output
    
    result["triage"] = {
        "action": triage_output.action,
        "urgency": triage_output.urgency,
        "category": triage_output.category,
        "reasoning": triage_output.reasoning,
        "summary": triage_output.summary,
    }
    
    print(f"   Action: {triage_output.action}")
    print(f"   Urgency: {triage_output.urgency}")
    print(f"   Category: {triage_output.category}")
    print(f"   Summary: {triage_output.summary}")
    
    # Handle triage decisions
    if triage_output.action == "ignore":
        print("\n⏭️  Email marked to ignore. Skipping.")
        result["final_action"] = "ignored"
        mark_as_read(email["id"])
        return result
    
    if triage_output.action == "flag_for_human":
        print("\n🚩 Email flagged for human review.")
        await send_flag_notification(email, triage_output.reasoning, triage_output.summary)
        result["final_action"] = "flagged_at_triage"
        return result
    
    # ============ Step 2: Draft Reply ============
    print("\n✍️  Step 2: Draft Agent generating reply...")
    
    draft_agent = create_draft_agent()
    draft_input = f"""
Generate a reply to this email:

FROM: {email['sender']}
SUBJECT: {email['subject']}

ORIGINAL EMAIL:
{email['body']}

TRIAGE CONTEXT:
- Category: {triage_output.category}
- Urgency: {triage_output.urgency}
- Summary: {triage_output.summary}
"""
    
    draft_result = await Runner.run(draft_agent, draft_input)
    draft_output: DraftResult = draft_result.final_output
    
    result["draft"] = {
        "subject": draft_output.subject,
        "body": draft_output.body,
        "tone": draft_output.tone,
        "key_points": draft_output.key_points_addressed,
    }
    
    print(f"   Subject: {draft_output.subject}")
    print(f"   Tone: {draft_output.tone}")
    print(f"   Key points: {', '.join(draft_output.key_points_addressed)}")
    
    # ============ Step 3: Quality Check ============
    print("\n🔍 Step 3: Quality Checker reviewing draft...")
    
    quality_agent = create_quality_checker_agent()
    quality_input = f"""
Review this email draft:

ORIGINAL EMAIL:
From: {email['sender']}
Subject: {email['subject']}
Body: {email['body']}

DRAFT REPLY:
Subject: {draft_output.subject}
Body: {draft_output.body}

CONTEXT:
- Email category: {triage_output.category}
- Urgency level: {triage_output.urgency}
- Tone used: {draft_output.tone}
"""
    
    quality_result = await Runner.run(quality_agent, quality_input)
    quality_output: QualityResult = quality_result.final_output
    
    result["quality"] = {
        "approved": quality_output.approved,
        "needs_human_review": quality_output.needs_human_review,
        "score": quality_output.quality_score,
        "issues": quality_output.issues,
        "suggestions": quality_output.suggestions,
    }
    
    print(f"   Approved: {quality_output.approved}")
    print(f"   Score: {quality_output.quality_score}/10")
    print(f"   Needs human review: {quality_output.needs_human_review}")
    
    if quality_output.issues:
        print(f"   Issues: {', '.join(quality_output.issues)}")
    
    # ============ Final Action ============
    if quality_output.needs_human_review:
        print("\n🚩 Draft needs human review. Sending notification...")
        await send_flag_notification(
            email,
            f"Quality check flagged this draft. Issues: {', '.join(quality_output.issues)}",
            triage_output.summary
        )
        result["final_action"] = "flagged_at_quality"
    elif quality_output.approved:
        print("\n✅ Draft approved! Saving to drafts...")
        # Extract sender email address
        sender_email = extract_email_address(email["sender"])
        draft_result = create_draft(
            to=sender_email,
            subject=draft_output.subject,
            body=quality_output.final_draft,
            reply_to_message_id=email["id"]
        )
        mark_as_read(email["id"])
        result["final_action"] = "draft_created"
        result["draft_id"] = draft_result.get("draft_id")
        print(f"   Draft saved! Check your Gmail Drafts folder to review and send.")
    else:
        print("\n⚠️  Draft not approved but not flagged. Marking for review.")
        await send_flag_notification(
            email,
            f"Draft not approved. Suggestions: {', '.join(quality_output.suggestions)}",
            triage_output.summary
        )
        result["final_action"] = "flagged_low_quality"
    
    return result


async def send_flag_notification(email: dict, reason: str, summary: str):
    """Send a flagged email notification."""
    notification_body = f"""
FLAGGED EMAIL NOTIFICATION
==========================

An email has been flagged for your review.

Original Email Details:
- From: {email['sender']}
- Subject: {email['subject']}

Summary:
{summary}

Reason for Flagging:
{reason}

Please review this email in your inbox and respond manually.
"""
    
    send_email(
        to=NOTIFICATION_EMAIL,
        subject=f"FLAGGED EMAIL: {email['subject']}",
        body=notification_body
    )


def extract_email_address(sender: str) -> str:
    """Extract email address from 'Name <email@domain.com>' format."""
    if "<" in sender and ">" in sender:
        start = sender.index("<") + 1
        end = sender.index(">")
        return sender[start:end]
    return sender


async def run_workflow(max_emails: int = 5):
    """Main workflow: fetch and process emails."""
    print("🚀 Starting Email Agent Workflow")
    print(f"📧 Fetching up to {max_emails} unread emails...")
    
    try:
        emails = get_unread_emails(max_results=max_emails)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nTo set up Gmail API:")
        print("1. Go to https://console.cloud.google.com/")
        print("2. Create a project and enable Gmail API")
        print("3. Create OAuth 2.0 credentials (Desktop app)")
        print("4. Download credentials.json to this directory")
        return
    
    if not emails:
        print("📭 No unread emails found.")
        return
    
    print(f"📬 Found {len(emails)} unread email(s)")
    
    results = []
    for email in emails:
        try:
            result = await process_email(email)
            results.append(result)
        except Exception as e:
            print(f"\n❌ Error processing email '{email['subject']}': {e}")
            results.append({
                "email_id": email["id"],
                "subject": email["subject"],
                "error": str(e)
            })
    
    # Summary
    print("\n" + "="*60)
    print("📊 WORKFLOW SUMMARY")
    print("="*60)
    
    actions = {"draft_created": 0, "ignored": 0, "flagged": 0, "error": 0}
    for r in results:
        action = r.get("final_action", "error")
        if "flagged" in action:
            actions["flagged"] += 1
        elif action in actions:
            actions[action] += 1
        else:
            actions["error"] += 1
    
    print(f"📝 Drafts created: {actions['draft_created']}")
    print(f"⏭️  Ignored: {actions['ignored']}")
    print(f"🚩 Flagged for review: {actions['flagged']}")
    print(f"❌ Errors: {actions['error']}")
    
    return results


def main():
    """Entry point."""
    asyncio.run(run_workflow())


if __name__ == "__main__":
    main()
