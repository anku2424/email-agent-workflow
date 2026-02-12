"""
Test the agents locally without Gmail integration.
This lets you verify Ollama + agents work before setting up Gmail.
"""
import asyncio
from agents import Runner, set_tracing_disabled
from agents_workflow import (
    create_triage_agent,
    create_draft_agent,
    create_quality_checker_agent,
)

# Disable tracing for local models
set_tracing_disabled(True)

# Sample test email
SAMPLE_EMAIL = {
    "sender": "john.smith@company.com",
    "subject": "Question about project timeline",
    "date": "2024-02-08",
    "body": """Hi,

I wanted to follow up on our meeting yesterday. We discussed the project 
deadline being moved to March 15th, but I haven't received the updated 
schedule yet.

Could you please send me:
1. The revised project timeline
2. Updated task assignments
3. Any changes to the deliverables

This is quite urgent as I need to brief my team by end of week.

Thanks,
John"""
}


async def test_triage_agent():
    """Test the Triage Agent."""
    print("\n" + "="*60)
    print("🧪 Testing Triage Agent")
    print("="*60)
    
    agent = create_triage_agent()
    
    input_text = f"""
Analyze this email:

FROM: {SAMPLE_EMAIL['sender']}
SUBJECT: {SAMPLE_EMAIL['subject']}
DATE: {SAMPLE_EMAIL['date']}

BODY:
{SAMPLE_EMAIL['body']}
"""
    
    result = await Runner.run(agent, input_text)
    output = result.final_output
    
    print(f"\n📋 Triage Results:")
    print(f"   Action: {output.action}")
    print(f"   Urgency: {output.urgency}")
    print(f"   Category: {output.category}")
    print(f"   Summary: {output.summary}")
    print(f"   Reasoning: {output.reasoning}")
    
    return output


async def test_draft_agent(triage_output):
    """Test the Draft Agent."""
    print("\n" + "="*60)
    print("🧪 Testing Draft Agent")
    print("="*60)
    
    agent = create_draft_agent()
    
    input_text = f"""
Generate a reply to this email:

FROM: {SAMPLE_EMAIL['sender']}
SUBJECT: {SAMPLE_EMAIL['subject']}

ORIGINAL EMAIL:
{SAMPLE_EMAIL['body']}

TRIAGE CONTEXT:
- Category: {triage_output.category}
- Urgency: {triage_output.urgency}
- Summary: {triage_output.summary}
"""
    
    result = await Runner.run(agent, input_text)
    output = result.final_output
    
    print(f"\n✍️  Draft Results:")
    print(f"   Subject: {output.subject}")
    print(f"   Tone: {output.tone}")
    print(f"   Key Points: {', '.join(output.key_points_addressed)}")
    print(f"\n   Draft Body:\n{'-'*40}")
    print(output.body)
    print(f"{'-'*40}")
    
    return output


async def test_quality_checker(triage_output, draft_output):
    """Test the Quality Checker Agent."""
    print("\n" + "="*60)
    print("🧪 Testing Quality Checker Agent")
    print("="*60)
    
    agent = create_quality_checker_agent()
    
    input_text = f"""
Review this email draft:

ORIGINAL EMAIL:
From: {SAMPLE_EMAIL['sender']}
Subject: {SAMPLE_EMAIL['subject']}
Body: {SAMPLE_EMAIL['body']}

DRAFT REPLY:
Subject: {draft_output.subject}
Body: {draft_output.body}

CONTEXT:
- Email category: {triage_output.category}
- Urgency level: {triage_output.urgency}
- Tone used: {draft_output.tone}
"""
    
    result = await Runner.run(agent, input_text)
    output = result.final_output
    
    print(f"\n🔍 Quality Check Results:")
    print(f"   Approved: {output.approved}")
    print(f"   Score: {output.quality_score}/10")
    print(f"   Needs Human Review: {output.needs_human_review}")
    
    if output.issues:
        print(f"   Issues: {', '.join(output.issues)}")
    if output.suggestions:
        print(f"   Suggestions: {', '.join(output.suggestions)}")
    
    print(f"\n   Final Draft:\n{'-'*40}")
    print(output.final_draft)
    print(f"{'-'*40}")
    
    return output


async def run_full_test():
    """Run all agents in sequence."""
    print("\n🚀 Starting Agent Test Suite")
    print("="*60)
    print(f"Test Email: '{SAMPLE_EMAIL['subject']}'")
    print(f"From: {SAMPLE_EMAIL['sender']}")
    
    # Test Triage
    triage_output = await test_triage_agent()
    
    if triage_output.action == "ignore":
        print("\n⏭️  Triage says ignore. Stopping here.")
        return
    
    if triage_output.action == "flag_for_human":
        print("\n🚩 Triage flagged for human. Stopping here.")
        return
    
    # Test Draft
    draft_output = await test_draft_agent(triage_output)
    
    # Test Quality
    quality_output = await test_quality_checker(triage_output, draft_output)
    
    # Final verdict
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    if quality_output.approved and not quality_output.needs_human_review:
        print("✅ Email would be AUTO-SENT")
    elif quality_output.needs_human_review:
        print("🚩 Email would be FLAGGED for human review")
    else:
        print("⚠️  Email would need revisions")


if __name__ == "__main__":
    print("Make sure Ollama is running with qwen3:8b loaded!")
    print("  ollama pull qwen3:8b")
    print("  ollama serve")
    asyncio.run(run_full_test())
