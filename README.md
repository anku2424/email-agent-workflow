# Email Agent Workflow

An automated email processing system using OpenAI Agents SDK with Ollama (qwen3:8b).

## Architecture

```
New email arrives to <INSERT EMAIL>
         ↓
Agent 1: "Triage Agent"
- Analyzes email importance/urgency
- Decides: respond, ignore, or flag for human
         ↓
Agent 2: "Draft Agent" (if respond)
- Generates reply draft
- Passes to Agent 3
         ↓
Agent 3: "Quality Checker"
- Reviews draft for tone, accuracy
- Flags if needs human review (sends email with title "FLAGGED EMAIL")
```

## Prerequisites

1. **Python 3.9+**
2. **Ollama** installed and running
3. **Gmail API credentials**

## Setup

### 1. Install Ollama and pull the model

```bash
# Install Ollama from https://ollama.com
# Then pull qwen3:8b
ollama pull qwen3:8b
```

### 2. Install Python dependencies

```bash
cd email-agent-workflow
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set up Gmail API

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or select existing)
3. Enable the **Gmail API**:
   - Go to "APIs & Services" → "Enable APIs and Services"
   - Search for "Gmail API" and enable it
4. Create OAuth 2.0 credentials:
   - Go to "APIs & Services" → "Credentials"
   - Click "Create Credentials" → "OAuth client ID"
   - Choose "Desktop app"
   - Download the credentials
5. Rename the downloaded file to `credentials.json` and place it in this directory

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env if needed (defaults should work)
```

### 5. First run (Gmail authorization)

```bash
python main.py
```

On first run, a browser window will open for Gmail authorization. Grant the requested permissions.

## Usage

```bash
# Ensure Ollama is running
ollama serve

# Run the workflow
python main.py
```

## How It Works

### Agent 1: Triage Agent
- Analyzes each unread email
- Classifies by category (business, personal, spam, etc.)
- Rates urgency (high, medium, low)
- Decides action:
  - **respond**: Pass to Draft Agent
  - **ignore**: Mark as read, skip
  - **flag_for_human**: Send notification, skip auto-reply

### Agent 2: Draft Agent
- Creates a contextually appropriate reply
- Matches tone to original email
- Addresses all questions/points raised

### Agent 3: Quality Checker
- Reviews draft for:
  - Appropriate tone
  - Accuracy
  - Completeness
- If approved: sends the reply
- If issues found: saves draft for review

## Files

- `main.py` - Workflow orchestration
- `agents_workflow.py` - Agent definitions and tools
- `gmail_tools.py` - Gmail API functions
- `config.py` - Configuration (Ollama, email settings)

## Customization

### Change the model
Edit `.env`:
```
OLLAMA_MODEL=llama3.2  # or any other Ollama model
```

### Adjust agent behavior
Modify the `instructions` in `agents_workflow.py` for each agent.

## Troubleshooting

**"Ollama connection refused"**
- Ensure Ollama is running: `ollama serve`

**"Missing credentials.json"**
- Complete Gmail API setup (Step 3 above)

**"Model not found"**
- Pull the model: `ollama pull qwen3:8b`
