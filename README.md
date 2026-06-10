# AI Study Assistant

A conversational AI study agent built for GSB 570 (Generative AI). It combines RAG-powered Q&A, deadline-aware study planning, and Google Calendar integration into a single Streamlit web app backed by Amazon Bedrock.

---

## Features

- **RAG Q&A** — upload your course materials (PDF or TXT) and ask concept questions grounded in your own notes
- **Study planner** — generates a day-by-day plan from your topics, hours available, and a deadline
- **Calendar-aware scheduling** — connects to Google Calendar to avoid conflicts and schedule study sessions as real events
- **Smart Study Planner tab** — review, edit, and selectively push AI-proposed sessions to Google Calendar
- **Session memory** — maintains multi-turn conversation history with automatic compression

---

## Project Structure

```
study_assistant.py    # Core logic: RAG, planner, calendar agent, supervisor loop
app.py                # Streamlit web UI
google_auth/
  auth.py             # Google OAuth2 flow and service helpers
  calendar_tools.py   # Calendar read/write tools
  credentials.json    # ← your Google OAuth client credentials (not committed)
  token.json          # ← auto-generated OAuth token (not committed)
tests/
  test_examples.py    # Example-based unit tests
  test_properties.py  # Property-based tests (Hypothesis)
requirements.txt      # Python dependencies
```

---

## Prerequisites

- Python 3.12+
- An AWS account with Amazon Bedrock access in `us-east-1` (or your chosen region)
- Model access enabled for:
  - `anthropic.claude-3-5-haiku-20241022-v1:0` (supervisor / chat)
  - `amazon.titan-embed-text-v2:0` (embeddings)
- A Google Cloud project with the Calendar API enabled and OAuth 2.0 credentials downloaded as `google_auth/credentials.json`

---

## Setup

**1. Clone the repo and install dependencies**

```bash
git clone <your-repo-url>
cd Study_Assistant_Project
pip install -r requirements.txt
```

**2. Configure AWS credentials**

The app uses `boto3`, which reads credentials from the standard AWS credential chain. The easiest option is to configure the AWS CLI:

```bash
aws configure
```

Or set environment variables:

```bash
set AWS_ACCESS_KEY_ID=your_key
set AWS_SECRET_ACCESS_KEY=your_secret
set AWS_DEFAULT_REGION=us-east-1
```

> **Never commit your AWS credentials or any `.env` file to the repository.**

**3. Add Google Calendar credentials**

1. Go to [Google Cloud Console](https://console.cloud.google.com/) → APIs & Services → Credentials
2. Create an OAuth 2.0 Client ID (Desktop app)
3. Download the JSON file and save it as `google_auth/credentials.json`

The first time you connect the calendar in the app, a browser window will open for OAuth authorization. The token is saved automatically to `google_auth/token.json`.

> **Do not commit `credentials.json` or `token.json` to the repository.**

**4. (Optional) Override defaults via environment variables**

| Variable | Default | Description |
|---|---|---|
| `AWS_REGION` | `us-east-1` | AWS region for Bedrock API calls |
| `STUDY_ASSISTANT_MODEL_ID` | `anthropic.claude-3-5-haiku-20241022-v1:0` | Bedrock model ID |
| `VECTOR_STORE_PATH` | `study_assistant.db` | Path to the SQLite vector store |

---

## Running the App

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

**Alternatively**, use the CLI for a terminal-based chat session:

```bash
python study_assistant.py                        # starts with default prompt
python study_assistant.py "Help me study RAG"    # starts with a custom prompt
python study_assistant.py --ingest notes.pdf     # ingest a document then chat
python study_assistant.py --debug                # print Bedrock request payloads
```

---

## How to Use (Web UI)

### 1. Load course materials
Upload a PDF or TXT file in the sidebar and click **Ingest Document**. The document is chunked, embedded with Amazon Titan, and stored in a local SQLite vector store.

### 2. Chat with the assistant
In the **Study Assistant** tab, ask questions like:
- *"Explain chunking strategies from my notes"*
- *"Quiz me on RAG concepts"*
- *"I have a quiz on June 12 — create a 3-day study plan"*

Study plans use the format `Month Day: Topic (X hours)` so sessions are automatically parsed and sent to the Smart Study Planner tab.

### 3. Connect Google Calendar
Click **Connect Google Calendar** in the sidebar to authorize access. The app reads your upcoming events to avoid scheduling conflicts.

### 4. Review and schedule in the Smart Study Planner tab
Switch to the **Smart Study Planner** tab to:
- Review the proposed sessions
- Edit topic labels
- Adjust scheduling preferences (earliest/latest study hour, session length)
- Select which sessions to add, then click **Add Selected to Google Calendar**

---

## Running Tests

```bash
pytest tests/ -v
```

The test suite includes 16 example-based unit tests and property-based tests covering chunking invariants, ingestion idempotence, RAG retrieval sort order, z-score filtering, deadline enforcement, calendar hour reduction, and vector store persistence.

---

## Security Notes

- AWS credentials are read from the environment — never hardcoded
- Google OAuth tokens are stored locally only (`google_auth/token.json`)
- The following files should be in `.gitignore` and never committed:
  - `google_auth/credentials.json`
  - `google_auth/token.json`
  - `study_assistant.db` (contains your personal document embeddings)
  - Any `.env` files

---

## Architecture

```
User prompt
    │
    ▼
run_supervisor_turn()          ← Claude Sonnet 4.5 via Bedrock converse()
    │
    ├── dispatch_tool("answer_concept_question")
    │       └── run_qa_agent()
    │               ├── search_chunks()        ← cosine sim + z-score filter
    │               └── Bedrock converse()     ← grounded answer
    │
    └── dispatch_tool("generate_study_plan")
            ├── run_calendar_agent()           ← Google Calendar events
            └── run_study_planner()            ← day-by-day DayEntry list
                        │
                        ▼
            build_study_blocks_from_plan_entries()
                        │
                        ▼
            create_selected_study_sessions()   ← writes to Google Calendar
```

Vector store: SQLite with embeddings stored as binary BLOBs (`study_assistant.db`)  
Embedding model: Amazon Titan Text Embeddings V2 (1024 dimensions, normalized)

---

## Dependencies

See `requirements.txt`. Key packages:

| Package | Purpose |
|---|---|
| `boto3` | AWS SDK — Bedrock `converse()` API and Titan embeddings |
| `numpy` | Embedding vector math (cosine similarity, z-score filtering) |
| `PyPDF2` | PDF text extraction for document ingestion |
| `streamlit` | Web UI |
| `google-api-python-client` | Google Calendar API |
| `google-auth-oauthlib` | OAuth 2.0 flow for Google |
