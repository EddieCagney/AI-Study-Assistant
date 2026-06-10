"""
AI Study Assistant
==================
A multi-capability conversational study agent for GSB 570 (Generative AI).

Composes the following patterns from the existing codebase:
  - Raw boto3 converse() agentic loops (study_coach_agent.py)
  - Amazon Titan Text Embeddings V2 + cosine similarity + z-score filtering (rag-overview.py)
  - Supervisor + specialist multi-agent pattern (multi_agent_research.py)
  - SQLite vector store with binary blob embeddings (rag-overview.py)

Capabilities:
  1. RAG-powered Q&A — answers concept questions grounded in ingested course materials
  2. Deadline-aware study planning — generates day-by-day plans using calendar data
  3. Multi-agent orchestration — routes requests to specialist sub-agents automatically
  4. Session memory — maintains conversation history and compresses it when it grows large

Usage:
  python study_assistant.py [prompt] [--ingest <file_path>] [--debug]

Environment variables:
  AWS_REGION                  AWS region (default: us-east-1)
  STUDY_ASSISTANT_MODEL_ID    Bedrock model ID (default: us.anthropic.claude-sonnet-4-5-20250929-v1:0)
  VECTOR_STORE_PATH           SQLite DB path (default: study_assistant.db)
"""

import datetime
import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Optional

import argparse

import boto3
import numpy as np
import PyPDF2


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """
    Runtime configuration for the Study Assistant.

    All values are read from environment variables with documented defaults.
    The ``debug`` flag is set via the ``--debug`` CLI argument rather than
    an environment variable, so it defaults to False here and is toggled
    by ``main()`` after parsing args.

    Environment variables:
        AWS_REGION                  → region          (default: "us-east-1")
        STUDY_ASSISTANT_MODEL_ID    → model_id        (default: "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
        VECTOR_STORE_PATH           → vector_store_path (default: "study_assistant.db")
    """

    # AWS region used for all Bedrock API calls.
    region: str = field(
        default_factory=lambda: os.environ.get("AWS_REGION", "us-east-1")
    )

    # Bedrock model ID for the supervisor agent and history summarizer.
    # Uses Claude 3.5 Haiku which supports tool use and runs in us-west-2
    # without a cross-region inference profile. Override with STUDY_ASSISTANT_MODEL_ID.
    model_id: str = field(
        default_factory=lambda: os.environ.get(
            "STUDY_ASSISTANT_MODEL_ID",
            "anthropic.claude-3-5-haiku-20241022-v1:0",
        )
    )

    # Path to the SQLite database file used as the persistent vector store.
    vector_store_path: str = field(
        default_factory=lambda: os.environ.get("VECTOR_STORE_PATH", "study_assistant.db")
    )

    # Target size of each text chunk in characters (used by chunk_text()).
    chunk_size: int = 1024

    # Number of characters to overlap between consecutive chunks.
    chunk_overlap: int = 256

    # Dimensionality of Titan Text Embeddings V2 vectors.
    embedding_dimensions: int = 1024

    # When True, the full Bedrock request payload is printed before each API call.
    # Set to True by passing --debug on the command line.
    debug: bool = False


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class DayEntry:
    """
    A single day's entry in a generated study plan.

    Fields:
        date         ISO-format date string ("YYYY-MM-DD") for this study day.
        focus_topic  The topic the student should focus on for this day.
        hours        Allocated study hours (may be halved for calendar-busy days).
        outcome      A concrete deliverable or takeaway expected by end of day.
    """

    date: str          # ISO format: "YYYY-MM-DD"
    focus_topic: str   # Topic for the day
    hours: float       # Allocated study hours (may be halved for busy days)
    outcome: str       # Concrete deliverable or takeaway


@dataclass
class SessionMemory:
    """
    In-memory state for a single conversation session.

    Fields:
        messages         Full converse() message history in Bedrock format.
                         Each entry is a dict with "role" and "content" keys,
                         following the Bedrock converse() message schema exactly.
        last_study_plan  The most recently generated study plan dict, or None
                         if no plan has been generated in this session yet.
                         Stored so the supervisor can reference it on follow-up turns
                         without the user repeating it (Requirement 6.3).
    """

    messages: list = field(default_factory=list)
    last_study_plan: Optional[dict] = None


# =============================================================================
# Vector Store
# =============================================================================

def init_vector_store(db_path: str) -> sqlite3.Connection:
    """
    Open (or create) the SQLite vector store and ensure the schema exists.

    Creates the ``rag_chunks`` table and the ``idx_source_doc`` index if they
    do not already exist.  Passing ``":memory:"`` creates a transient in-memory
    database, which is useful for tests and one-off ingestion jobs.

    Schema
    ------
    rag_chunks
        id          INTEGER PRIMARY KEY AUTOINCREMENT
        chunk_text  TEXT    NOT NULL
        source_doc  TEXT    NOT NULL   -- original file name, e.g. "attention.pdf"
        embedding   BLOB    NOT NULL   -- numpy float64 array serialized via .tobytes()
        created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP

    Index
    -----
    idx_source_doc ON rag_chunks(source_doc)
        Speeds up document-level queries (list, dedup check, deletion).

    Args:
        db_path: Filesystem path for the SQLite database file, or ``":memory:"``
                 for a transient in-memory database.

    Returns:
        An open ``sqlite3.Connection`` with the schema initialized.

    Requirements: 2.8, 4.3
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS rag_chunks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_text  TEXT      NOT NULL,
            source_doc  TEXT      NOT NULL,
            embedding   BLOB      NOT NULL,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_source_doc ON rag_chunks(source_doc)
    """)

    conn.commit()
    return conn


# =============================================================================
# Embedding
# =============================================================================

def embed_text(text: str, config: Config) -> np.ndarray:
    """
    Embed a text string using Amazon Titan Text Embeddings V2.

    Calls Bedrock's ``invoke_model`` directly (no LangChain) with the Titan
    Text Embeddings V2 model.  The request asks Titan to normalize the output
    vector (``"normalize": True``), so the returned vector is already a unit
    vector.  A safety re-normalization step is applied afterward in case of
    floating-point drift.

    Args:
        text:   The input text to embed (a word, sentence, or paragraph).
        config: Runtime configuration supplying ``region`` and
                ``embedding_dimensions``.

    Returns:
        A 1-D ``np.ndarray`` of shape ``(config.embedding_dimensions,)`` with
        dtype ``float32``.  The vector is normalized to unit length.

    Raises:
        botocore.exceptions.ClientError: If the Bedrock API call fails.
        KeyError: If the response body does not contain an ``"embedding"`` field.

    Requirements: 2.8
    """
    bedrock = boto3.client("bedrock-runtime", region_name=config.region)

    body = json.dumps({
        "inputText": text,
        "dimensions": config.embedding_dimensions,
        "normalize": True,
    })

    response = bedrock.invoke_model(
        body=body,
        modelId="amazon.titan-embed-text-v2:0",
        accept="application/json",
        contentType="application/json",
    )

    response_body = json.loads(response["body"].read())
    vector = np.array(response_body["embedding"], dtype=np.float32)

    # Safety re-normalization: Titan returns a unit vector when normalize=True,
    # but floating-point serialization/deserialization can introduce tiny drift.
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm

    return vector


# =============================================================================
# Text Chunking
# =============================================================================

def chunk_text(text: str, chunk_size: int = 1024, overlap: int = 256) -> list[str]:
    """
    Split text into overlapping character-based chunks.

    Uses a simple sliding-window algorithm: a window of ``chunk_size`` characters
    slides across the text, stepping forward by ``(chunk_size - overlap)`` characters
    on each iteration.  This guarantees that the last ``overlap`` characters of
    chunk N appear at the start of chunk N+1, preserving context across boundaries.

    Consistent with the chunking approach in ``Embeddings/rag-overview.py``.

    Args:
        text:       The full text to chunk.
        chunk_size: Target size of each chunk in characters (default 1024).
        overlap:    Number of characters to overlap between consecutive chunks
                    (default 256).  Must be less than ``chunk_size``.

    Returns:
        A list of non-empty, stripped text chunks.  If ``text`` is shorter than
        ``chunk_size`` (or empty), a list containing the single stripped text is
        returned (empty string yields ``[""]`` only if text is non-empty after
        stripping; an all-whitespace input yields ``[]``).

    Examples:
        >>> chunks = chunk_text("hello world", chunk_size=5, overlap=2)
        >>> # window 0-5 → "hello", window 3-8 → "lo wo", window 6-11 → "world"
    """
    # Short-circuit: text fits in a single chunk
    if len(text) <= chunk_size:
        stripped = text.strip()
        return [stripped] if stripped else []

    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0

    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step

    return chunks


# =============================================================================
# Document Ingestion
# =============================================================================

def ingest_document(file_path: str, conn: sqlite3.Connection, config: Config) -> str:
    """
    Validate, parse, chunk, embed, and store a course material file.

    Supports ``.pdf`` and ``.txt`` files (case-insensitive).  If the document
    has already been ingested (identified by its basename), the function returns
    early without adding duplicate chunks.

    Args:
        file_path: Absolute or relative path to the file to ingest.
        conn:      Open SQLite connection to the vector store (must have the
                   ``rag_chunks`` table created by ``init_vector_store()``).
        config:    Runtime configuration (used by ``embed_text()`` for the
                   Bedrock region and embedding dimensions, and by
                   ``chunk_text()`` for chunk size and overlap).

    Returns:
        A descriptive string indicating the outcome:

        - ``"Ingested {n} chunks from {name}"``          — success
        - ``"Document already ingested: {name}"``        — duplicate, skipped
        - ``"File not found: {file_path}"``              — path does not exist
        - ``"Unsupported file type: {ext}. Supported: .pdf, .txt"``
        - ``"Failed to parse PDF: {e}"``                 — PyPDF2 error
        - ``"Database write failed: {e}"``               — SQLite error

    Requirements: 2.6, 2.7, 2.8, 2.9, 4.1, 4.2, 4.3
    """
    try:
        # ------------------------------------------------------------------
        # 1. Validate file existence
        # ------------------------------------------------------------------
        if not os.path.exists(file_path):
            return f"File not found: {file_path}"

        # ------------------------------------------------------------------
        # 2. Validate extension
        # ------------------------------------------------------------------
        _, ext = os.path.splitext(file_path)
        ext_lower = ext.lower()
        if ext_lower not in (".pdf", ".txt"):
            return f"Unsupported file type: {ext}. Supported: .pdf, .txt"

        # ------------------------------------------------------------------
        # 3. Deduplication check — use just the filename as source_doc
        # ------------------------------------------------------------------
        name = os.path.basename(file_path)
        cursor = conn.execute(
            "SELECT COUNT(*) FROM rag_chunks WHERE source_doc = ?", (name,)
        )
        if cursor.fetchone()[0] > 0:
            return f"Document already ingested: {name}"

        # ------------------------------------------------------------------
        # 4. Parse the document into raw text
        # ------------------------------------------------------------------
        if ext_lower == ".pdf":
            try:
                with open(file_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    pages_text = [page.extract_text() or "" for page in reader.pages]
                text = "\n".join(pages_text)
            except Exception as e:
                return f"Failed to parse PDF: {e}"
        else:  # .txt
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

        # ------------------------------------------------------------------
        # 5. Chunk → embed → insert
        # ------------------------------------------------------------------
        chunks = chunk_text(text, chunk_size=config.chunk_size, overlap=config.chunk_overlap)

        try:
            for chunk in chunks:
                embedding = embed_text(chunk, config)
                conn.execute(
                    "INSERT INTO rag_chunks (chunk_text, source_doc, embedding) VALUES (?, ?, ?)",
                    (chunk, name, embedding.tobytes()),
                )
            conn.commit()
        except sqlite3.Error as e:
            return f"Database write failed: {e}"

        return f"Ingested {len(chunks)} chunks from {name}"

    except Exception as e:
        # Catch-all so the function always returns a string, never raises.
        return f"Database write failed: {e}"


# =============================================================================
# Study Planner Specialist
# =============================================================================

def run_study_planner(
    goal: str,
    topics: list[str],
    hours_per_week: float,
    days: int,
    deadline: str | None,
    calendar_events: list[dict],
) -> dict:
    """
    Build a day-by-day study plan for the given goal and topics.

    Planning logic:
      1. Compute ``hours_per_day = hours_per_week / days``.
      2. If ``deadline`` is provided (ISO string "YYYY-MM-DD"), count *backward*
         from the deadline date, assigning one topic per day so that the last
         topic lands on (or before) the deadline.
      3. If no deadline is provided, start from today and go *forward*, one
         topic per day.
      4. For each day, halve the hours if ``calendar_events`` contains an event
         whose ``"date"`` key matches that day's date.
      5. If ``hours_per_week < len(topics)``, include a warning in the result.

    Args:
        goal:            The overarching study goal (e.g. "Prepare for midterm").
        topics:          Ordered list of topics to cover, one per day.
        hours_per_week:  Total study hours available per week.
        days:            Number of study days per week (used to compute daily hours).
        deadline:        Optional ISO date string ("YYYY-MM-DD") for the last study day.
                         When provided, topics are assigned counting backward from this date.
        calendar_events: List of event dicts, each with at least a ``"date"`` key
                         (ISO string "YYYY-MM-DD").  Days matching an event get
                         half the normal daily hours.

    Returns:
        A dict with two keys:
          - ``"plan"``:    A list of :class:`DayEntry` objects, one per topic.
          - ``"warning"``: A warning string if ``hours_per_week < len(topics)``,
                           otherwise ``None``.

    Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6
    """
    # --- 1. Compute base daily hours ---
    hours_per_day: float = hours_per_week / days

    # --- 2. Build the set of event dates for O(1) lookup ---
    event_dates: set[str] = {
        event["date"] for event in calendar_events if "date" in event
    }

    # --- 3. Determine start date and direction ---
    n = len(topics)

    today = datetime.date.today()
    dated_topics: list[tuple[datetime.date, str]]

    if deadline is not None:
        deadline_date = datetime.date.fromisoformat(deadline)
        available_days = (deadline_date - today).days + 1
        if available_days <= 0:
            return {"plan": [], "warning": "Warning: the deadline has already passed."}

        if available_days >= n:
            start_date = deadline_date - datetime.timedelta(days=n - 1)
            dated_topics = [
                (start_date + datetime.timedelta(days=i), topic)
                for i, topic in enumerate(topics)
            ]
        else:
            # When the requested timeline has already started, keep the plan on or after today
            # by grouping multiple topics into each remaining study day.
            dated_topics = []
            group_size = n / available_days
            start_idx = 0
            for day_index in range(available_days):
                end_idx = round((day_index + 1) * group_size)
                group = topics[start_idx:max(start_idx + 1, end_idx)]
                start_idx = max(start_idx + 1, end_idx)
                grouped_topic = "; ".join(t.strip() for t in group if isinstance(t, str) and t.strip())
                if not grouped_topic:
                    grouped_topic = f"Topic group {day_index + 1}"
                dated_topics.append((today + datetime.timedelta(days=day_index), grouped_topic))
    else:
        dated_topics = [
            (today + datetime.timedelta(days=i), topic)
            for i, topic in enumerate(topics)
        ]

    # --- 4. Build plan entries ---
    plan: list[DayEntry] = []
    for i, (day_date, topic) in enumerate(dated_topics):
        day_str = day_date.isoformat()  # "YYYY-MM-DD"

        # Halve hours on calendar-busy days (Requirement 3.2)
        hours = hours_per_day / 2 if day_str in event_dates else hours_per_day

        # Normalize topic: strip whitespace; fall back to a placeholder so
        # every entry always has a non-empty focus_topic (Requirement 3.4).
        normalized_topic = topic.strip() if isinstance(topic, str) else str(topic)
        if not normalized_topic:
            normalized_topic = f"Topic {i + 1}"

        entry = DayEntry(
            date=day_str,
            focus_topic=normalized_topic,
            hours=hours,
            outcome=f"Complete study of {normalized_topic}",
        )
        plan.append(entry)

    # --- 5. Insufficient-hours warning (Requirement 3.5) ---
    warning: str | None = None
    if hours_per_week < n:
        warning = (
            f"Warning: insufficient hours ({hours_per_week}h/week) "
            f"to cover all {n} topics"
        )

    return {"plan": plan, "warning": warning}


# =============================================================================
# Calendar Agent Specialist
# =============================================================================

def run_calendar_agent(days_ahead: int = 14) -> list[dict]:
    """
    Retrieve Google Calendar events for the next ``days_ahead`` days.
    Returns events with full time information for accurate free-slot detection.
    """
    try:
        from google_auth.auth import get_calendar_service
        from datetime import datetime as _dt, timedelta as _td, timezone as _tz

        service = get_calendar_service()
        now = _dt.now(_tz.utc)
        time_min = now.isoformat()
        time_max = (now + _td(days=days_ahead)).isoformat()

        result = service.events().list(
            calendarId="primary",
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy="startTime",
        ).execute()

        events = []
        for item in result.get("items", []):
            start_raw = item["start"].get("dateTime", item["start"].get("date", ""))
            end_raw = item["end"].get("dateTime", item["end"].get("date", ""))
            date = start_raw[:10] if start_raw else ""
            summary = item.get("summary", "(No title)")

            # Extract HH:MM times if available (dateTime events have times)
            if "T" in start_raw:
                # Parse local time from ISO string (e.g. "2026-05-18T14:00:00-07:00")
                start_time = start_raw[11:16]  # "HH:MM"
                end_time = end_raw[11:16] if "T" in end_raw else "23:59"
                all_day = False
            else:
                start_time = "00:00"
                end_time = "23:59"
                all_day = True

            if date:
                events.append({
                    "date": date,
                    "summary": summary,
                    "start_time": start_time,
                    "end_time": end_time,
                    "all_day": all_day,
                })

        return events

    except Exception:
        return []


def create_study_sessions(
    plan: list,
    start_hour: int = 9,
    session_duration_hours: float = 1.5,
) -> list[dict]:
    """
    Create Google Calendar events for each entry in a study plan.
    """
    try:
        from google_auth.auth import get_calendar_service
        service = get_calendar_service()
    except Exception as e:
        return [{"date": "N/A", "status": "failed", "detail": f"Calendar not available: {e}"}]

    results = []
    for entry in plan:
        try:
            duration = min(entry.hours, session_duration_hours)
            end_hour = start_hour + duration
            start_str = f"{int(start_hour):02d}:{int((start_hour % 1) * 60):02d}"
            end_str = f"{int(end_hour):02d}:{int((end_hour % 1) * 60):02d}"

            event = {
                "summary": f"Study: {entry.focus_topic}",
                "description": f"Outcome: {entry.outcome}\n\nGenerated by AI Study Assistant.",
                "start": {
                    "dateTime": f"{entry.date}T{start_str}:00",
                    "timeZone": "America/Los_Angeles",
                },
                "end": {
                    "dateTime": f"{entry.date}T{end_str}:00",
                    "timeZone": "America/Los_Angeles",
                },
                "colorId": "2",
            }

            created = service.events().insert(calendarId="primary", body=event).execute()
            results.append({
                "date": entry.date,
                "status": "created",
                "detail": f"Created: {created.get('summary')} on {entry.date} {start_str}-{end_str}"
            })
        except Exception as e:
            results.append({"date": entry.date, "status": "failed", "detail": str(e)})

    return results


# =============================================================================
# RAG Search
# =============================================================================

def search_chunks(
    query: str,
    conn: sqlite3.Connection,
    config: Config,
    top_n: int = 50,
) -> list[tuple[str, str, float]]:
    """
    Embed a query, search the vector store by cosine similarity, and return
    the most relevant chunks after z-score filtering.

    Algorithm
    ---------
    1. Embed ``query`` using Titan Text Embeddings V2 (unit vector).
    2. Fetch ALL rows from ``rag_chunks`` (no SQL-level ordering — embeddings
       are BLOBs so similarity must be computed in Python).
    3. For each row, deserialize the stored embedding blob and compute the
       dot product with the query vector.  Because both vectors are unit
       vectors (Titan normalizes on ingest and ``embed_text`` re-normalizes),
       the dot product equals the cosine similarity.
    4. Sort all (chunk_text, source_doc, score) triples in descending order
       of cosine similarity and keep the top ``top_n``.
    5. Apply z-score filtering on the top-``top_n`` scores:
         z_scores = (scores - mean) / std
       Keep only items whose z-score is ≥ ``max_z / 2``.
       If ``std == 0`` (all scores identical), return all ``top_n`` items
       unchanged (no filtering possible).
    6. Return the surviving triples in descending score order.

    Args:
        query:  The user's natural-language question or search string.
        conn:   Open SQLite connection to the vector store (must have the
                ``rag_chunks`` table created by ``init_vector_store()``).
        config: Runtime configuration passed to ``embed_text()``.
        top_n:  Maximum number of candidates to consider before z-score
                filtering (default 50).  All rows are fetched from the DB;
                only the top ``top_n`` by cosine similarity are kept before
                the filter is applied.

    Returns:
        A list of ``(chunk_text, source_doc, cosine_similarity)`` tuples,
        sorted in descending order of cosine similarity.  Returns ``[]`` if
        the vector store contains no chunks.

    Requirements: 2.1, 2.2
    """
    # ------------------------------------------------------------------
    # 1. Embed the query
    # ------------------------------------------------------------------
    query_vec = embed_text(query, config)

    # ------------------------------------------------------------------
    # 2. Fetch all rows from the vector store
    # ------------------------------------------------------------------
    cursor = conn.execute("SELECT chunk_text, source_doc, embedding FROM rag_chunks")
    rows = cursor.fetchall()

    if not rows:
        return []

    # ------------------------------------------------------------------
    # 3. Compute cosine similarity for every chunk
    # ------------------------------------------------------------------
    # Both query_vec and chunk_vec are unit vectors, so dot product == cosine sim.
    results: list[tuple[str, str, float]] = []
    for row in rows:
        chunk_text_val: str = row[0]
        source_doc_val: str = row[1]
        chunk_vec = np.frombuffer(row[2], dtype=np.float32)
        score = float(np.dot(query_vec, chunk_vec))
        results.append((chunk_text_val, source_doc_val, score))

    # ------------------------------------------------------------------
    # 4. Sort descending by cosine similarity and take top_n
    # ------------------------------------------------------------------
    results.sort(key=lambda x: x[2], reverse=True)
    results = results[:top_n]

    # ------------------------------------------------------------------
    # 5. Z-score filtering (Requirement 2.2)
    #    Keep items where z_score >= max_z / 2.
    #    If std == 0, all scores are identical — return all results.
    # ------------------------------------------------------------------
    scores = np.array([r[2] for r in results], dtype=np.float64)
    mean = float(np.mean(scores))
    std = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0

    if std == 0.0:
        # Cannot discriminate — return all candidates unchanged.
        return results

    z_scores = (scores - mean) / std
    max_z = float(np.max(z_scores))

    # Guard against floating-point artifacts: if std is a near-zero artifact
    # (all scores effectively identical), max_z will be <= 0 and the threshold
    # max_z / 2 would exclude everything.  Return all results in that case.
    if max_z <= 0.0:
        return results

    filtered = [
        results[i]
        for i in range(len(results))
        if z_scores[i] >= max_z / 2
    ]

    return filtered


# =============================================================================
# Document Management
# =============================================================================

def list_documents(conn: sqlite3.Connection) -> list[str]:
    """
    Return a sorted list of all distinct document names in the vector store.

    Queries the ``source_doc`` column of ``rag_chunks`` for distinct values and
    returns them in ascending alphabetical order.

    Args:
        conn: Open SQLite connection to the vector store (must have the
              ``rag_chunks`` table created by ``init_vector_store()``).

    Returns:
        A sorted list of unique source document names (strings).  Returns an
        empty list if no documents have been ingested yet.

    Requirements: 4.4
    """
    cursor = conn.execute(
        "SELECT DISTINCT source_doc FROM rag_chunks ORDER BY source_doc"
    )
    return [row[0] for row in cursor.fetchall()]


def remove_document(source_name: str, conn: sqlite3.Connection) -> None:
    """
    Delete all chunks belonging to a given source document from the vector store.

    Removes every row in ``rag_chunks`` whose ``source_doc`` matches
    ``source_name`` exactly, then commits the transaction.

    Args:
        source_name: The document name to remove (must match the ``source_doc``
                     value used during ingestion, i.e. the file's basename).
        conn:        Open SQLite connection to the vector store (must have the
                     ``rag_chunks`` table created by ``init_vector_store()``).

    Returns:
        None.  If ``source_name`` does not exist in the store, the function
        completes silently without error.

    Requirements: 4.5
    """
    conn.execute(
        "DELETE FROM rag_chunks WHERE source_doc = ?",
        (source_name,),
    )
    conn.commit()


def format_time_12h(time_str: str) -> str:
    """
    Convert a 24-hour ``HH:MM`` time string to 12-hour ``H:MM AM/PM`` format.

    Returns the original value unchanged if it cannot be parsed cleanly.
    """
    try:
        hour_str, minute_str = time_str.split(":")
        hour = int(hour_str)
        minute = int(minute_str)
        suffix = "AM" if hour < 12 else "PM"
        hour_12 = hour % 12 or 12
        return f"{hour_12}:{minute:02d} {suffix}"
    except Exception:
        return time_str


def format_calendar_event_line(event: dict) -> str:
    """
    Format a calendar event for display in prompts and chat responses.
    """
    if event.get("all_day", False):
        time_range = "All day"
    else:
        start = format_time_12h(event.get("start_time", "?"))
        end = format_time_12h(event.get("end_time", "?"))
        time_range = f"{start}-{end}"
    return f"  {event['date']} {time_range}: {event['summary']}"


def _parse_time_to_float(time_str: str) -> float:
    """Convert HH:MM to float hours."""
    hour_str, minute_str = time_str.split(":")
    return int(hour_str) + int(minute_str) / 60.0


def _format_hour_float(hour_value: float) -> str:
    """Convert float hours to 12-hour display format."""
    total_minutes = int(round(hour_value * 60))
    hour = (total_minutes // 60) % 24
    minute = total_minutes % 60
    return format_time_12h(f"{hour:02d}:{minute:02d}")


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping time intervals."""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def build_calendar_summary_response(
    calendar_events: list[dict],
    days_ahead: int = 7,
    day_start: float = 8.0,
    day_end: float = 21.0,
) -> str:
    """
    Build a direct natural-language summary of the user's connected calendar.
    """
    today = datetime.date.today()
    end_date = today + datetime.timedelta(days=days_ahead - 1)

    relevant_events = []
    for event in calendar_events:
        try:
            event_date = datetime.date.fromisoformat(event["date"])
        except Exception:
            continue
        if today <= event_date <= end_date:
            relevant_events.append(event)

    relevant_events.sort(key=lambda e: (e["date"], e.get("start_time", "00:00")))

    lines = [f"Here’s your Google Calendar for the next {days_ahead} days:"]
    if relevant_events:
        for event in relevant_events:
            lines.append(format_calendar_event_line(event).strip())
    else:
        lines.append("No events are scheduled in that window.")

    lines.append("")
    lines.append("Study gaps you could use:")

    suggestions = []
    for offset in range(days_ahead):
        current_date = today + datetime.timedelta(days=offset)
        date_str = current_date.isoformat()
        day_events = [e for e in relevant_events if e["date"] == date_str]

        if any(e.get("all_day", False) for e in day_events):
            continue

        busy = []
        for event in day_events:
            try:
                busy.append((
                    _parse_time_to_float(event.get("start_time", "00:00")),
                    _parse_time_to_float(event.get("end_time", "23:59")),
                ))
            except Exception:
                continue

        merged_busy = _merge_intervals(busy)
        cursor = day_start
        day_slots = []
        for start, end in merged_busy:
            if start > cursor and (start - cursor) >= 1.0:
                day_slots.append((cursor, start))
            cursor = max(cursor, end)
        if day_end > cursor and (day_end - cursor) >= 1.0:
            day_slots.append((cursor, day_end))

        for start, end in day_slots[:2]:
            duration = end - start
            if duration >= 1.0:
                label = "good for a 1-2 hour study block"
                if duration >= 2.5:
                    label = "good for a longer focused study block"
                suggestions.append(
                    f"- {date_str} {_format_hour_float(start)}-{_format_hour_float(end)}: {label}"
                )

    if suggestions:
        lines.extend(suggestions[:8])
    else:
        lines.append("- I don’t see any 1+ hour gaps between 8:00 AM and 9:00 PM this week.")

    lines.append("")
    lines.append(
        "If you want, I can also turn the best gaps into a concrete study plan for your Finance Analytics exam "
        "using realistic 1-2 hour study sessions."
    )
    return "\n".join(lines)


def build_calendar_write_limitation_response() -> str:
    """
    Explain that chat can suggest sessions but does not directly create them.
    """
    return (
        "I haven’t added anything to your Google Calendar from chat. "
        "I can suggest study sessions here, but actual calendar creation happens through the app controls.\n\n"
        "To add study sessions to your Google Calendar:\n"
        "1. Generate a study plan in chat.\n"
        "2. In the chat plan box, click 'Use This Plan in Smart Study Planner'.\n"
        "3. Review the proposed sessions.\n"
        "4. Click the Add to Google Calendar button in the app.\n\n"
        "If you want, I can help you turn your open gaps into a concrete study schedule first."
    )


def _sanitize_conversation_messages(messages: list[dict]) -> list[dict]:
    """
    Remove malformed or empty Bedrock message content blocks before converse().
    """
    sanitized: list[dict] = []
    for msg in messages:
        role = msg.get("role")
        if role not in {"user", "assistant"}:
            continue

        content_blocks = msg.get("content", [])
        cleaned_blocks: list[dict] = []
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            if "text" in block:
                text = str(block.get("text", "")).strip()
                if text:
                    cleaned_blocks.append({"text": text})
            elif "toolUse" in block or "toolResult" in block:
                cleaned_blocks.append(block)

        if cleaned_blocks:
            sanitized.append({"role": role, "content": cleaned_blocks})

    return sanitized


def _parse_duration_text_to_hours(duration_text: str) -> float | None:
    """
    Parse duration text like ``1-1.5 hours`` or ``45-60 mins`` into hours.
    """
    cleaned = duration_text.strip().lower()
    number_matches = re.findall(r"\d+(?:\.\d+)?", cleaned)
    if not number_matches:
        return None

    values = [float(val) for val in number_matches]
    value = max(values)

    if "min" in cleaned:
        return value / 60.0
    return value


def parse_time_12h_to_24h(time_text: str) -> str | None:
    """
    Convert ``H:MM AM/PM`` into ``HH:MM``.
    """
    cleaned = time_text.strip().upper()
    match = re.match(r"^(\d{1,2}):(\d{2})\s*([AP]M)$", cleaned)
    if not match:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2))
    meridiem = match.group(3)

    if hour == 12:
        hour = 0
    if meridiem == "PM":
        hour += 12

    return f"{hour:02d}:{minute:02d}"


def parse_natural_date_to_iso(date_text: str, default_year: int | None = None) -> str | None:
    """
    Convert dates like ``May 26th`` into ``YYYY-MM-DD``.
    """
    if default_year is None:
        default_year = datetime.date.today().year

    cleaned = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", date_text.strip(), flags=re.IGNORECASE)
    for fmt in ("%B %d", "%b %d"):
        try:
            parsed = datetime.datetime.strptime(cleaned, fmt)
            return datetime.date(default_year, parsed.month, parsed.day).isoformat()
        except ValueError:
            continue
    return None


def parse_month_day_to_iso(month_day_text: str, default_year: int | None = None) -> str | None:
    """
    Convert dates like ``5/25`` into ``YYYY-MM-DD``.
    """
    if default_year is None:
        default_year = datetime.date.today().year

    match = re.match(r"^\s*(\d{1,2})/(\d{1,2})\s*$", month_day_text)
    if not match:
        return None

    month = int(match.group(1))
    day = int(match.group(2))
    try:
        return datetime.date(default_year, month, day).isoformat()
    except ValueError:
        return None


def parse_compact_time_to_24h(time_text: str, default_meridiem: str | None = None) -> str | None:
    """
    Convert compact times like ``2``, ``2pm``, or ``2:30pm`` into ``HH:MM``.
    """
    cleaned = time_text.strip().lower().replace(" ", "")
    match = re.match(r"^(\d{1,2})(?::(\d{2}))?([ap]m)?$", cleaned)
    if not match:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2) or "00")
    meridiem = (match.group(3) or default_meridiem or "").upper()
    if meridiem not in {"AM", "PM"}:
        return None

    if hour == 12:
        hour = 0
    if meridiem == "PM":
        hour += 12

    return f"{hour:02d}:{minute:02d}"


def _normalize_study_session_window(
    start_24h: str,
    end_24h: str,
    max_session_hours: float = 2.0,
    min_session_hours: float = 0.5,
) -> tuple[str, str, float]:
    """
    Convert a broad availability window into one focused study session.

    If a chat response proposes a very large window, we keep the window's
    start time but cap the scheduled session duration to a realistic length.
    """
    start_float = _parse_time_to_float(start_24h)
    end_float = _parse_time_to_float(end_24h)
    raw_duration = max(0.0, end_float - start_float)

    duration_hours = max(min_session_hours, min(raw_duration, max_session_hours))
    duration_hours = round(duration_hours * 2) / 2
    normalized_end = start_float + duration_hours

    end_hour = int(normalized_end)
    end_minute = int(round((normalized_end % 1) * 60))
    if end_minute == 60:
        end_hour += 1
        end_minute = 0

    normalized_end_24h = f"{end_hour:02d}:{end_minute:02d}"
    return start_24h, normalized_end_24h, duration_hours


def parse_study_plan_entries_from_text(
    response_text: str,
    start_date: datetime.date | None = None,
    default_year: int | None = None,
) -> list[DayEntry]:
    """
    Extract structured study plan entries from free-form assistant text.

    Supports both:
    - ``Day 1: Topic (1.5 hours)``
    - ``Altman Z-Score (1-1.5 hours)``
    """
    if start_date is None:
        start_date = datetime.date.today()
    if default_year is None:
        default_year = start_date.year

    normalized_text = response_text.replace(" • ", "\n• ")
    normalized_text = normalized_text.replace("• ", "\n• ")

    inline_entries: list[DayEntry] = []
    natural_inline_matches = re.finditer(
        r"(?:^|[•\n])\s*(?:[A-Za-z]+,\s*)?([A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?)"
        r"(?:\s*\([^)]*\))?\s*:\s*([^()\n]+?)\s*\(([0-9.\-\s]+)\s*hours?\)",
        normalized_text,
        flags=re.IGNORECASE,
    )
    for match in natural_inline_matches:
        date_text, topic, hours_text = match.groups()
        entry_date = parse_natural_date_to_iso(date_text, default_year=default_year)
        if not entry_date:
            continue
        try:
            hours = float(hours_text.strip())
        except ValueError:
            continue
        topic = topic.strip().rstrip(".:")
        inline_entries.append(
            DayEntry(
                date=entry_date,
                focus_topic=topic,
                hours=hours,
                outcome=f"Complete study of {topic}",
            )
        )

    compact_inline_matches = re.finditer(
        r"(?:^|[•\n])\s*(?:[A-Za-z]+,\s*)?(\d{1,2}/\d{1,2})"
        r"(?:\s*\([^)]*\))?\s*:\s*([^()\n]+?)\s*\(([0-9.\-\s]+)\s*hours?\)",
        normalized_text,
        flags=re.IGNORECASE,
    )
    for match in compact_inline_matches:
        date_text, topic, hours_text = match.groups()
        entry_date = parse_month_day_to_iso(date_text, default_year=default_year)
        if not entry_date:
            continue
        try:
            hours = float(hours_text.strip())
        except ValueError:
            continue
        topic = topic.strip().rstrip(".:")
        inline_entries.append(
            DayEntry(
                date=entry_date,
                focus_topic=topic,
                hours=hours,
                outcome=f"Complete study of {topic}",
            )
        )

    if inline_entries:
        return inline_entries

    entries: list[DayEntry] = []
    next_day_offset = 0
    pending_entry_date: str | None = None
    pending_entry_topic: str | None = None

    for raw_line in normalized_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("•"):
            line = line[1:].strip()

        day_match = re.search(
            r"[Dd]ay\s*(\d+)[:\s\-]+([^(\n]+?)(?:\s*\(([0-9.\-\s]+)\s*hours?\))?$",
            line,
        )
        if day_match:
            day_index = max(int(day_match.group(1)) - 1, 0)
            topic = day_match.group(2).strip().rstrip(".:")
            hours = float(day_match.group(3).strip()) if day_match.group(3) else 1.5
            entry_date = (start_date + datetime.timedelta(days=day_index)).isoformat()
            entries.append(
                DayEntry(
                    date=entry_date,
                    focus_topic=topic,
                    hours=hours,
                    outcome=f"Complete study of {topic}",
                )
            )
            next_day_offset = max(next_day_offset, day_index + 1)
            continue

        dated_header_match = re.match(
            r"^(?:[A-Za-z]+,\s*)?([A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?)(?:\s*\([^)]*\))?\s*:\s*$",
            line,
            flags=re.IGNORECASE,
        )
        if dated_header_match:
            pending_entry_date = parse_natural_date_to_iso(
                dated_header_match.group(1),
                default_year=default_year,
            )
            pending_entry_topic = None
            continue

        compact_dated_header_match = re.match(
            r"^(?:[A-Za-z]+,\s*)?(\d{1,2}/\d{1,2})(?:\s*\([^)]*\))?\s*:\s*$",
            line,
            flags=re.IGNORECASE,
        )
        if compact_dated_header_match:
            pending_entry_date = parse_month_day_to_iso(
                compact_dated_header_match.group(1),
                default_year=default_year,
            )
            pending_entry_topic = None
            continue

        if pending_entry_date:
            focus_match = re.match(r"^Focus:\s*(.+)$", line, flags=re.IGNORECASE)
            if focus_match:
                pending_entry_topic = focus_match.group(1).strip().rstrip(".:")
                continue

            study_time_match = re.match(r"^Study Time:\s*~?\s*(.+)$", line, flags=re.IGNORECASE)
            if study_time_match and pending_entry_topic:
                hours = _parse_duration_text_to_hours(study_time_match.group(1))
                if hours is not None:
                    entries.append(
                        DayEntry(
                            date=pending_entry_date,
                            focus_topic=pending_entry_topic,
                            hours=hours,
                            outcome=f"Complete study of {pending_entry_topic}",
                        )
                    )
                pending_entry_date = None
                pending_entry_topic = None
                continue

        natural_date_match = re.match(
            r"^(?:[A-Za-z]+,\s*)?([A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?)(?:\s*\([^)]*\))?\s*:\s*"
            r"([^(\n]+?)\s*\(([0-9.\-\s]+)\s*hours?\)\s*$",
            line,
            flags=re.IGNORECASE,
        )
        if natural_date_match:
            date_text, topic, hours_text = natural_date_match.groups()
            entry_date = parse_natural_date_to_iso(date_text, default_year=default_year)
            if not entry_date:
                continue
            try:
                hours = float(hours_text.strip())
            except ValueError:
                continue
            topic = topic.strip().rstrip(".:")
            entries.append(
                DayEntry(
                    date=entry_date,
                    focus_topic=topic,
                    hours=hours,
                    outcome=f"Complete study of {topic}",
                )
            )
            continue

        compact_date_match = re.match(
            r"^(?:[A-Za-z]+,\s*)?(\d{1,2}/\d{1,2})(?:\s*\([^)]*\))?\s*:\s*"
            r"([^(\n]+?)\s*\(([0-9.\-\s]+)\s*hours?\)\s*$",
            line,
            flags=re.IGNORECASE,
        )
        if compact_date_match:
            date_text, topic, hours_text = compact_date_match.groups()
            entry_date = parse_month_day_to_iso(date_text, default_year=default_year)
            if not entry_date:
                continue
            try:
                hours = float(hours_text.strip())
            except ValueError:
                continue
            topic = topic.strip().rstrip(".:")
            entries.append(
                DayEntry(
                    date=entry_date,
                    focus_topic=topic,
                    hours=hours,
                    outcome=f"Complete study of {topic}",
                )
            )
            continue

        item_match = re.match(
            r"^(?:[-*•]\s*|\d+[.)]\s*)?([A-Za-z][^()]{2,}?)\s*\(([^)]+)\)\s*$",
            line,
        )
        if not item_match:
            continue

        topic = item_match.group(1).strip().rstrip(".:")
        duration_text = item_match.group(2)
        if "hour" not in duration_text.lower() and "min" not in duration_text.lower():
            continue
        hours = _parse_duration_text_to_hours(duration_text)
        if hours is None:
            continue

        lower_topic = topic.lower()
        if lower_topic.startswith("source:") or lower_topic.startswith("created:"):
            continue
        if any(token in line for token in ['"', "[", "]", "Source:"]):
            continue

        entry_date = (start_date + datetime.timedelta(days=next_day_offset)).isoformat()
        entries.append(
            DayEntry(
                date=entry_date,
                focus_topic=topic,
                hours=hours,
                outcome=f"Complete study of {topic}",
            )
        )
        next_day_offset += 1

    if not entries:
        natural_inline_matches = re.finditer(
            r"(?:^|[•\n])\s*(?:[A-Za-z]+,\s*)?([A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?)"
            r"(?:\s*\([^)]*\))?\s*:\s*([^()\n]+?)\s*\(([0-9.\-\s]+)\s*hours?\)",
            normalized_text,
            flags=re.IGNORECASE,
        )
        for match in natural_inline_matches:
            date_text, topic, hours_text = match.groups()
            entry_date = parse_natural_date_to_iso(date_text, default_year=default_year)
            if not entry_date:
                continue
            try:
                hours = float(hours_text.strip())
            except ValueError:
                continue
            topic = topic.strip().rstrip(".:")
            entries.append(
                DayEntry(
                    date=entry_date,
                    focus_topic=topic,
                    hours=hours,
                    outcome=f"Complete study of {topic}",
                )
            )

        compact_inline_matches = re.finditer(
            r"(?:^|[•\n])\s*(?:[A-Za-z]+,\s*)?(\d{1,2}/\d{1,2})"
            r"(?:\s*\([^)]*\))?\s*:\s*([^()\n]+?)\s*\(([0-9.\-\s]+)\s*hours?\)",
            normalized_text,
            flags=re.IGNORECASE,
        )
        for match in compact_inline_matches:
            date_text, topic, hours_text = match.groups()
            entry_date = parse_month_day_to_iso(date_text, default_year=default_year)
            if not entry_date:
                continue
            try:
                hours = float(hours_text.strip())
            except ValueError:
                continue
            topic = topic.strip().rstrip(".:")
            entries.append(
                DayEntry(
                    date=entry_date,
                    focus_topic=topic,
                    hours=hours,
                    outcome=f"Complete study of {topic}",
                )
            )

    return entries


def parse_study_blocks_from_text(response_text: str) -> list[dict]:
    """
    Extract explicit dated time-slot study blocks from assistant text.

    Supports lines like:
    ``2026-05-20 8:00 AM-2:00 PM: Focus on financial ratios``
    """
    blocks: list[dict] = []
    schedule_section_headers = [
        "now, let's fit these topics into your calendar",
        "now let's fit these topics into your calendar",
        "optimal study session windows",
        "here is a suggested study plan",
        "study plan:",
        "proposed study plan:",
        "recommended study plan:",
        "study schedule:",
        "suggested study schedule:",
        "recommended study schedule:",
        "proposed study schedule:",
    ]
    in_schedule_section = False
    pending_block: dict | None = None

    def _looks_like_schedule_header_line(text: str) -> bool:
        patterns = [
            r"^\d{4}-\d{2}-\d{2}\s+",
            r"^[A-Za-z]+(?:\s+\d{1,2}(?:st|nd|rd|th)?),\s*\d",
            r"^(?:[A-Za-z]+,\s*)?[A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?\s+from\s+\d",
            r"^(?:[A-Za-z]+day\s+)?\d{1,2}/\d{1,2},\s*\d",
        ]
        return any(re.match(pattern, text, flags=re.IGNORECASE) for pattern in patterns)

    def _finalize_pending_block() -> None:
        nonlocal pending_block
        if not pending_block:
            return

        start_24h = pending_block.get("start_time")
        end_24h = pending_block.get("end_time")
        topic = (pending_block.get("topic") or "").strip()
        if not start_24h or not end_24h or not topic:
            pending_block = None
            return

        try:
            start_24h, end_24h, duration_hours = _normalize_study_session_window(
                start_24h,
                end_24h,
            )
        except Exception:
            duration_hours = 1.0

        label = f"Study: {topic}"
        blocks.append({
            "date": pending_block["date"],
            "start_time": start_24h,
            "end_time": end_24h,
            "topic": label,
            "label": label,
            "duration_hours": duration_hours,
            "selected": True,
        })
        pending_block = None

    for raw_line in response_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        lower_line = line.lower()
        if any(header in lower_line for header in schedule_section_headers):
            in_schedule_section = True
            continue

        if not in_schedule_section:
            continue

        if pending_block:
            if _looks_like_schedule_header_line(line):
                _finalize_pending_block()
            elif pending_block.get("topic"):
                if lower_line not in {
                    "relevant excerpts:",
                    "suggested formats for computational exam questions, including calculating:",
                } and not line.startswith('"'):
                    pending_block["topic"] = f"{pending_block['topic']} {line}".strip()
                    continue
            else:
                if lower_line in {
                    "relevant excerpts:",
                    "suggested formats for computational exam questions, including calculating:",
                } or line.startswith('"'):
                    continue
                if not _looks_like_schedule_header_line(line):
                    pending_block["topic"] = line.strip().rstrip(".")
                    continue

        topic_line_match = re.match(r"^Topics:\s*(.+)$", line, flags=re.IGNORECASE)
        if topic_line_match and pending_block:
            pending_block["topic"] = topic_line_match.group(1).strip().rstrip(".")
            _finalize_pending_block()
            continue

        weekday_natural_from_line_match = re.match(
            r"^(?:[A-Za-z]+,\s*)?([A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?)\s+from\s+"
            r"(\d{1,2}:\d{2}\s*[AP]M)\s*-\s*(\d{1,2}:\d{2}\s*[AP]M):?\s*$",
            line,
            flags=re.IGNORECASE,
        )
        if weekday_natural_from_line_match:
            _finalize_pending_block()
            date_text, start_12h, end_12h = weekday_natural_from_line_match.groups()
            date_str = parse_natural_date_to_iso(date_text)
            start_24h = parse_time_12h_to_24h(start_12h)
            end_24h = parse_time_12h_to_24h(end_12h)
            if start_24h and end_24h and date_str:
                pending_block = {
                    "date": date_str,
                    "start_time": start_24h,
                    "end_time": end_24h,
                    "topic": "",
                }
            continue

        dated_block_header_match = re.match(
            r"^(\d{4}-\d{2}-\d{2})\s+"
            r"(\d{1,2}:\d{2}\s*[AP]M)\s*-\s*(\d{1,2}:\d{2}\s*[AP]M)"
            r"(?:\s*\(([^)]+)\))?\s*$",
            line,
            flags=re.IGNORECASE,
        )
        if dated_block_header_match:
            _finalize_pending_block()
            date_str, start_12h, end_12h, _duration_text = dated_block_header_match.groups()
            start_24h = parse_time_12h_to_24h(start_12h)
            end_24h = parse_time_12h_to_24h(end_12h)
            if start_24h and end_24h:
                pending_block = {
                    "date": date_str,
                    "start_time": start_24h,
                    "end_time": end_24h,
                    "topic": "",
                }
            continue

        natural_date_line_match = re.match(
            r"^([A-Za-z]+(?:\s+\d{1,2}(?:st|nd|rd|th)?)),\s*"
            r"(\d{1,2}:\d{2}\s*[AP]M)\s*-\s*(\d{1,2}:\d{2}\s*[AP]M)\s*:\s*(.+)$",
            line,
            flags=re.IGNORECASE,
        )
        if natural_date_line_match:
            date_text, start_12h, end_12h, topic = natural_date_line_match.groups()
            date_str = parse_natural_date_to_iso(date_text)
            start_24h = parse_time_12h_to_24h(start_12h)
            end_24h = parse_time_12h_to_24h(end_12h)
            if not date_str or not start_24h or not end_24h:
                continue

            try:
                start_24h, end_24h, duration_hours = _normalize_study_session_window(
                    start_24h,
                    end_24h,
                )
            except Exception:
                duration_hours = 1.0

            label = f"Study: {topic.strip()}"
            blocks.append({
                "date": date_str,
                "start_time": start_24h,
                "end_time": end_24h,
                "topic": label,
                "label": label,
                "duration_hours": duration_hours,
                "selected": True,
            })
            continue

        natural_date_compact_time_match = re.match(
            r"^([A-Za-z]+(?:\s+\d{1,2}(?:st|nd|rd|th)?)),\s*"
            r"(\d{1,2}(?::\d{2})?(?:\s*[ap]m)?)\s*-\s*"
            r"(\d{1,2}(?::\d{2})?\s*[ap]m)\s*:\s*(.+)$",
            line,
            flags=re.IGNORECASE,
        )
        if natural_date_compact_time_match:
            date_text, start_text, end_text, topic = natural_date_compact_time_match.groups()
            date_str = parse_natural_date_to_iso(date_text)
            end_meridiem_match = re.search(r"([ap]m)\s*$", end_text.strip(), flags=re.IGNORECASE)
            default_meridiem = end_meridiem_match.group(1).upper() if end_meridiem_match else None
            start_24h = parse_compact_time_to_24h(start_text, default_meridiem=default_meridiem)
            end_24h = parse_compact_time_to_24h(end_text)
            if not date_str or not start_24h or not end_24h:
                continue

            try:
                start_24h, end_24h, duration_hours = _normalize_study_session_window(
                    start_24h,
                    end_24h,
                )
            except Exception:
                duration_hours = 1.0

            label = f"Study: {topic.strip()}"
            blocks.append({
                "date": date_str,
                "start_time": start_24h,
                "end_time": end_24h,
                "topic": label,
                "label": label,
                "duration_hours": duration_hours,
                "selected": True,
            })
            continue

        weekday_natural_date_line_match = re.match(
            r"^(?:[A-Za-z]+),\s*([A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?),\s*"
            r"(\d{1,2}:\d{2}\s*[AP]M)\s*-\s*(\d{1,2}:\d{2}\s*[AP]M)\s*:\s*(.+)$",
            line,
            flags=re.IGNORECASE,
        )
        if weekday_natural_date_line_match:
            date_text, start_12h, end_12h, topic = weekday_natural_date_line_match.groups()
            date_str = parse_natural_date_to_iso(date_text)
            start_24h = parse_time_12h_to_24h(start_12h)
            end_24h = parse_time_12h_to_24h(end_12h)
            if not date_str or not start_24h or not end_24h:
                continue

            try:
                start_24h, end_24h, duration_hours = _normalize_study_session_window(
                    start_24h,
                    end_24h,
                )
            except Exception:
                duration_hours = 1.0

            label = f"Study: {topic.strip()}"
            blocks.append({
                "date": date_str,
                "start_time": start_24h,
                "end_time": end_24h,
                "topic": label,
                "label": label,
                "duration_hours": duration_hours,
                "selected": True,
            })
            continue

        compact_date_line_match = re.match(
            r"^(?:[A-Za-z]+day\s+)?(\d{1,2}/\d{1,2}),\s*"
            r"(\d{1,2}(?::\d{2})?(?:\s*[ap]m)?)\s*-\s*"
            r"(\d{1,2}(?::\d{2})?\s*[ap]m)\s*:\s*(.+)$",
            line,
            flags=re.IGNORECASE,
        )
        if compact_date_line_match:
            month_day_text, start_text, end_text, topic = compact_date_line_match.groups()
            date_str = parse_month_day_to_iso(month_day_text)
            end_meridiem_match = re.search(r"([ap]m)\s*$", end_text.strip(), flags=re.IGNORECASE)
            default_meridiem = end_meridiem_match.group(1).upper() if end_meridiem_match else None
            start_24h = parse_compact_time_to_24h(start_text, default_meridiem=default_meridiem)
            end_24h = parse_compact_time_to_24h(end_text)
            if not date_str or not start_24h or not end_24h:
                continue

            try:
                start_24h, end_24h, duration_hours = _normalize_study_session_window(
                    start_24h,
                    end_24h,
                )
            except Exception:
                duration_hours = 1.0

            cleaned_topic = topic.strip()
            cleaned_topic = re.sub(
                r"^\d+(?:\.\d+)?-hour session(?:\s+on)?\s+",
                "",
                cleaned_topic,
                flags=re.IGNORECASE,
            )
            label = f"Study: {cleaned_topic}"
            blocks.append({
                "date": date_str,
                "start_time": start_24h,
                "end_time": end_24h,
                "topic": label,
                "label": label,
                "duration_hours": duration_hours,
                "selected": True,
            })
            continue

        match = re.match(
            r"^(\d{4}-\d{2}-\d{2})\s+"
            r"(\d{1,2}:\d{2}\s*[AP]M)\s*-\s*(\d{1,2}:\d{2}\s*[AP]M)\s*:\s*(.+)$",
            line,
            flags=re.IGNORECASE,
        )
        if not match:
            continue

        date_str, start_12h, end_12h, topic = match.groups()
        start_24h = parse_time_12h_to_24h(start_12h)
        end_24h = parse_time_12h_to_24h(end_12h)
        if not start_24h or not end_24h:
            continue

        try:
            start_24h, end_24h, duration_hours = _normalize_study_session_window(
                start_24h,
                end_24h,
            )
        except Exception:
            duration_hours = 1.0

        label = f"Study: {topic.strip()}"
        blocks.append({
            "date": date_str,
            "start_time": start_24h,
            "end_time": end_24h,
            "topic": label,
            "label": label,
            "duration_hours": duration_hours,
            "selected": True,
        })

    _finalize_pending_block()
    return blocks


def extract_study_blocks_with_llm(
    response_text: str,
    bedrock_client,
    model_id: str,
    reference_year: int | None = None,
) -> list[dict]:
    """
    Use the LLM to normalize study-session recommendations into structured blocks.

    This avoids brittle dependence on the exact prose date/time format used in chat.
    Returns an empty list if no explicit study sessions are found or if parsing fails.
    """
    if reference_year is None:
        reference_year = datetime.date.today().year

    extraction_prompt = f"""
You extract structured study sessions from an assistant's study-plan response.

Task:
- Find only the explicit recommended study sessions in the text.
- Ignore explanatory paragraphs, quotes, source citations, greetings, and signatures.
- Normalize every session into JSON with these exact keys:
  - date: YYYY-MM-DD
  - start_time: HH:MM in 24-hour time (use "09:00" as default if no clock time is given)
  - end_time: HH:MM in 24-hour time (derive from duration if given, otherwise use "10:30")
  - topic: concise study-topic label
  - duration_hours: numeric hours as a float (e.g. 1.5)

Rules:
- Assume the year is {reference_year} when the text omits the year.
- Lines like "June 10: Fixed Size Chunking (1.5 hours)" are valid sessions — extract date, topic, and duration.
- Lines like "June 10 (Today): Topic (1.5 hours)" are also valid — strip the parenthetical from the date.
- If a line has explicit times like "from 12:30 PM - 1:30 PM", derive start_time/end_time from those.
- If only duration is given (e.g. "1.5 hours"), set start_time="09:00" and compute end_time by adding duration to 09:00.
- If multiple topic lines belong to one date, combine them into one concise topic string.
- If a study plan says "~40 minutes", convert to 0.67 hours.
- Return ONLY valid JSON with no commentary before or after.
- Output format must be a JSON array. Example:
  [
    {{"date":"{reference_year}-06-10","start_time":"09:00","end_time":"10:30","topic":"Fixed Size Chunking","duration_hours":1.5}}
  ]
- If there are genuinely no study sessions at all in the text, return [].

Assistant response to extract from:
{response_text}
""".strip()

    try:
        response = bedrock_client.converse(
            modelId=model_id,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": extraction_prompt}],
                }
            ],
        )
        content = response["output"]["message"]["content"]
        text = "\n".join(block["text"] for block in content if "text" in block).strip()
        # Strip markdown code fences if the LLM wraps its output
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            return []

        blocks: list[dict] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            date_str = item.get("date")
            start_time = item.get("start_time", "09:00")
            end_time = item.get("end_time")
            topic = item.get("topic")
            duration_hours_raw = item.get("duration_hours")

            if not all(isinstance(val, str) and val.strip() for val in [date_str, topic]):
                continue

            # Derive end_time from duration if it's missing or if we have a duration override
            if duration_hours_raw is not None:
                try:
                    dur = float(duration_hours_raw)
                    start_float = _parse_time_to_float(start_time or "09:00")
                    end_float = start_float + dur
                    end_hour = int(end_float)
                    end_minute = int(round((end_float % 1) * 60))
                    if end_minute == 60:
                        end_hour += 1
                        end_minute = 0
                    end_time = f"{end_hour:02d}:{end_minute:02d}"
                except Exception:
                    pass

            if not start_time:
                start_time = "09:00"
            if not end_time:
                end_time = "10:30"

            try:
                start_time, end_time, duration_hours = _normalize_study_session_window(
                    start_time.strip(),
                    end_time.strip(),
                )
            except Exception:
                continue

            label = f"Study: {topic.strip()}"
            blocks.append({
                "date": date_str.strip(),
                "start_time": start_time,
                "end_time": end_time,
                "topic": label,
                "label": label,
                "duration_hours": duration_hours,
                "selected": True,
            })
        return blocks
    except Exception:
        return []


def build_study_blocks_from_plan_entries(
    plan_entries: list[DayEntry],
    calendar_events: list[dict],
    preferred_start_hour: int = 9,
    preferred_end_hour: int = 21,
    max_session_hours: float = 2.0,
    min_session_hours: float = 0.5,
) -> tuple[list[dict], dict]:
    """
    Convert explicit study plan entries into Smart Study Planner session blocks.
    """
    busy_intervals: dict[str, list[tuple[float, float]]] = {}
    for event in calendar_events:
        date = event.get("date", "")
        if not date:
            continue
        if event.get("all_day", False):
            busy_intervals.setdefault(date, []).append((0.0, 24.0))
        else:
            try:
                start = _parse_time_to_float(event.get("start_time", "00:00"))
                end = _parse_time_to_float(event.get("end_time", "23:59"))
            except Exception:
                continue
            busy_intervals.setdefault(date, []).append((start, end))

    blocks: list[dict] = []
    unscheduled = 0

    for entry in plan_entries:
        date = entry.date
        busy = _merge_intervals(busy_intervals.get(date, []))
        cursor = float(preferred_start_hour)
        free_slots: list[tuple[float, float]] = []
        for start, end in busy:
            if start > cursor and (start - cursor) >= min_session_hours:
                free_slots.append((cursor, start))
            cursor = max(cursor, end)
        if float(preferred_end_hour) > cursor and (float(preferred_end_hour) - cursor) >= min_session_hours:
            free_slots.append((cursor, float(preferred_end_hour)))

        chosen_slot: tuple[float, float] | None = None
        for slot in free_slots:
            if (slot[1] - slot[0]) >= min_session_hours:
                chosen_slot = slot
                break

        if chosen_slot is None:
            unscheduled += 1
            continue

        start_hour = chosen_slot[0]
        duration = min(max(entry.hours, min_session_hours), max_session_hours, chosen_slot[1] - chosen_slot[0])
        duration = max(min_session_hours, round(duration * 2) / 2)
        end_hour = min(chosen_slot[1], start_hour + duration)

        start_str = f"{int(start_hour):02d}:{int((start_hour % 1) * 60):02d}"
        end_str = f"{int(end_hour):02d}:{int((end_hour % 1) * 60):02d}"
        label = f"Study: {entry.focus_topic}"

        blocks.append({
            "date": date,
            "start_time": start_str,
            "end_time": end_str,
            "topic": label,
            "label": label,
            "duration_hours": max(min_session_hours, end_hour - start_hour),
            "selected": True,
        })

    stats = {
        "source": "chat_plan",
        "total_topics": len(plan_entries),
        "total_sessions": len(blocks),
        "total_study_hours": sum(block["duration_hours"] for block in blocks),
        "unscheduled_topics": unscheduled,
    }
    return blocks, stats


# =============================================================================
# Q&A Agent Specialist
# =============================================================================

def run_qa_agent(question: str, conn: sqlite3.Connection, config: Config) -> dict:
    """
    Retrieve relevant course material chunks and generate a grounded answer.

    Internal flow:
      1. Call ``search_chunks()`` to embed the question and retrieve the most
         relevant chunks from the vector store (with z-score filtering applied).
      2. If no chunks are returned (empty store, or all scores non-positive after
         filtering), return the standard fallback message with an empty sources list.
      3. Build a prompt that includes each retrieved chunk's text and source name
         as context, followed by the user's question.
      4. Call Claude Sonnet 4.5 via ``boto3 converse()`` to generate the answer.
      5. Return the answer text and a deduplicated list of source document names.

    Args:
        question: The user's natural-language concept question.
        conn:     Open SQLite connection to the vector store (must have the
                  ``rag_chunks`` table created by ``init_vector_store()``).
        config:   Runtime configuration supplying ``region`` and ``model_id``.

    Returns:
        A dict with two keys:
          - ``"answer"``:  The generated answer string, or the fallback message
                           if no relevant chunks were found.
          - ``"sources"``: A deduplicated list of source document names used as
                           context.  Empty list when the fallback is returned.

    Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
    """
    # ------------------------------------------------------------------
    # 1. Retrieve relevant chunks via RAG search
    # ------------------------------------------------------------------
    chunks = search_chunks(question, conn, config)

    # ------------------------------------------------------------------
    # 2. Fallback: no relevant chunks found
    # ------------------------------------------------------------------
    if not chunks:
        return {
            "answer": "I don't have enough course material to answer that question.",
            "sources": [],
        }

    # ------------------------------------------------------------------
    # 3. Build the context prompt
    #    Format:
    #      You are a study assistant. Answer the question using ONLY the
    #      provided course material context.
    #
    #      Context:
    #      [chunk 1 text]
    #      Source: [source_doc_1]
    #
    #      [chunk 2 text]
    #      Source: [source_doc_2]
    #      ...
    #
    #      Question: {question}
    # ------------------------------------------------------------------
    context_blocks: list[str] = []
    for chunk_text_val, source_doc_val, _score in chunks:
        context_blocks.append(f"{chunk_text_val}\nSource: {source_doc_val}")

    context_section = "\n\n".join(context_blocks)

    prompt = (
        "You are a study assistant. Answer the question using ONLY the provided course material context.\n\n"
        f"Context:\n{context_section}\n\n"
        f"Question: {question}"
    )

    # ------------------------------------------------------------------
    # 4. Call Claude Sonnet 4.5 via boto3 converse()
    # ------------------------------------------------------------------
    bedrock_client = boto3.client("bedrock-runtime", region_name=config.region)

    response = bedrock_client.converse(
        modelId=config.model_id,
        messages=[
            {
                "role": "user",
                "content": [{"text": prompt}],
            }
        ],
    )

    answer: str = response["output"]["message"]["content"][0]["text"]

    # ------------------------------------------------------------------
    # 5. Deduplicate sources (preserve order of first appearance)
    # ------------------------------------------------------------------
    seen: set[str] = set()
    sources: list[str] = []
    for _chunk_text, source_doc_val, _score in chunks:
        if source_doc_val not in seen:
            seen.add(source_doc_val)
            sources.append(source_doc_val)

    return {"answer": answer, "sources": sources}


# =============================================================================
# Session Memory — History Compression
# =============================================================================

# Model ID for the history summarizer (Claude 3.5 Haiku).
# Used by compress_history() to summarize the oldest 10 turns when the
# conversation history exceeds 20 messages.
SUPERVISOR_MODEL_ID = "anthropic.claude-3-5-haiku-20241022-v1:0"


def compress_history(memory: SessionMemory, bedrock_client) -> SessionMemory:
    """
    Compress the conversation history when it grows beyond 20 messages.

    If ``len(memory.messages) <= 20``, the function returns ``memory``
    unchanged.  Otherwise, the oldest 10 messages are summarized into a
    single "context" message via Claude Sonnet 4.5, and the resulting
    ``SessionMemory`` has a history length of ``(original_length - 10 + 1)``.

    Algorithm
    ---------
    1. If ``len(memory.messages) <= 20``, return ``memory`` unchanged.
    2. Take the first 10 messages: ``oldest = memory.messages[:10]``.
    3. Build a summarization prompt that formats the 10 messages as a
       conversation transcript and asks Claude to summarize the key points.
    4. Call ``bedrock_client.converse()`` with the summarization prompt.
    5. Extract the summary text from the response.
    6. Create a new "context" message:
       ``{"role": "user", "content": [{"text": "[Previous conversation summary]: {summary}"}]}``
    7. Build the new messages list: ``[context_message] + memory.messages[10:]``.
    8. Return ``SessionMemory(messages=new_messages, last_study_plan=memory.last_study_plan)``.

    Args:
        memory:         The current session memory containing the full message
                        history and the most recently generated study plan.
        bedrock_client: An active ``boto3`` Bedrock Runtime client (the same
                        client used by the supervisor agent).  Must support
                        the ``converse()`` API.

    Returns:
        A new ``SessionMemory`` instance.  If compression was applied, the
        ``messages`` list has length ``(original_length - 10 + 1)``.  If
        compression was not needed (history ≤ 20 messages), the original
        ``memory`` object is returned unchanged.

    Error handling:
        If the Bedrock call fails for any reason, the function logs a warning
        and returns the original ``memory`` unchanged so the session can
        continue with the uncompressed history.

    Requirements: 6.4
    """
    # Step 1: Guard — only compress when history exceeds 20 messages.
    if len(memory.messages) <= 20:
        return memory

    # Step 2: Isolate the oldest 10 messages.
    oldest = memory.messages[:10]

    # Step 3: Build a summarization prompt from the transcript.
    transcript_lines: list[str] = []
    for msg in oldest:
        role = msg.get("role", "unknown").capitalize()
        # Extract text from the content list (Bedrock converse() format).
        content_blocks = msg.get("content", [])
        text_parts: list[str] = []
        for block in content_blocks:
            if isinstance(block, dict):
                if "text" in block:
                    text_parts.append(block["text"])
                elif "toolUse" in block:
                    tool = block["toolUse"]
                    text_parts.append(
                        f"[Tool call: {tool.get('name', 'unknown')} "
                        f"with input {json.dumps(tool.get('input', {}))}]"
                    )
                elif "toolResult" in block:
                    result = block["toolResult"]
                    result_content = result.get("content", [])
                    result_texts = [
                        r.get("text", json.dumps(r.get("json", {})))
                        for r in result_content
                        if isinstance(r, dict)
                    ]
                    text_parts.append(f"[Tool result: {' '.join(result_texts)}]")
        transcript_lines.append(f"{role}: {' '.join(text_parts)}")

    transcript = "\n".join(transcript_lines)

    summary_prompt = (
        "The following is the beginning of a conversation between a student and an AI study "
        "assistant. Please summarize the key points, topics discussed, decisions made, and any "
        "study plans or goals mentioned. Be concise but preserve all important context that "
        "would help the assistant continue the conversation coherently.\n\n"
        f"Conversation transcript:\n{transcript}\n\n"
        "Summary:"
    )

    # Step 4: Call Claude Sonnet 4.5 to generate the summary.
    try:
        response = bedrock_client.converse(
            modelId=SUPERVISOR_MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [{"text": summary_prompt}],
                }
            ],
        )

        # Step 5: Extract the summary text from the response.
        output_message = response["output"]["message"]
        summary_parts = [
            block["text"]
            for block in output_message.get("content", [])
            if "text" in block
        ]
        summary = "\n".join(summary_parts).strip()

    except Exception as exc:
        # Graceful degradation: if summarization fails, keep the original history.
        import warnings
        warnings.warn(
            f"compress_history: summarization failed, keeping uncompressed history. "
            f"Reason: {exc}"
        )
        return memory

    # Step 6: Build the replacement context message.
    context_message: dict = {
        "role": "user",
        "content": [{"text": f"[Previous conversation summary]: {summary}"}],
    }

    # Step 7: Assemble the new messages list.
    new_messages = [context_message] + memory.messages[10:]

    # Step 8: Return the updated SessionMemory.
    return SessionMemory(
        messages=new_messages,
        last_study_plan=memory.last_study_plan,
    )


# =============================================================================
# Output Formatters
# =============================================================================

def format_study_plan(plan: list) -> str:
    """
    Format a list of DayEntry objects as a human-readable bulleted study plan.

    Each entry is rendered on its own line using the template::

        • {date} | {focus_topic} | {hours}h | {outcome}

    Args:
        plan: A list of :class:`DayEntry` objects (or any objects with
              ``date``, ``focus_topic``, ``hours``, and ``outcome`` attributes).

    Returns:
        A multi-line string with one bullet per plan entry, joined by newlines.
        Returns an empty string if ``plan`` is empty.

    Requirements: 7.1
    """
    lines: list[str] = []
    for entry in plan:
        lines.append(
            f"• {entry.date} | {entry.focus_topic} | {entry.hours}h | {entry.outcome}"
        )
    return "\n".join(lines)


def format_qa_response(answer: str, sources: list[str]) -> str:
    """
    Format a Q&A answer with optional source citations.

    If ``sources`` is non-empty, each source is appended on its own line
    prefixed with ``"Source: "``, separated from the answer by a blank line::

        {answer}

        Source: {source1}
        Source: {source2}
        ...

    If ``sources`` is empty, only the answer text is returned (no citation
    block, no trailing newline).

    Args:
        answer:  The answer text generated by the Q&A agent.
        sources: A list of source document names to cite.  Pass an empty list
                 to omit the citation block entirely.

    Returns:
        A formatted string combining the answer and (optionally) citations.

    Requirements: 7.2
    """
    if not sources:
        return answer

    citation_lines = "\n".join(f"Source: {src}" for src in sources)
    return f"{answer}\n\n{citation_lines}"


def format_turn(user_input: str, assistant_response: str) -> str:
    """
    Format a single conversation turn with labelled prefixes and a divider.

    Renders the turn as::

        You: {user_input}

        Assistant: {assistant_response}

        ----------------------------------------

    The divider is exactly 60 hyphens (≥ 40 characters as required).

    Args:
        user_input:          The raw text entered by the user.
        assistant_response:  The assistant's reply for this turn.

    Returns:
        A formatted string representing the full turn, ending with a divider
        line followed by a trailing newline.

    Requirements: 7.3, 7.4
    """
    divider = "-" * 60
    return (
        f"You: {user_input}\n"
        f"\n"
        f"Assistant: {assistant_response}\n"
        f"\n"
        f"{divider}\n"
    )


# =============================================================================
# Supervisor Agent
# =============================================================================
# The supervisor receives the user's message and orchestrates the three
# specialist agents by exposing them as tools in a boto3 converse() agentic
# loop — identical in structure to the SUPERVISOR_TOOL_CONFIG in
# multi_agent_research.py.
#
# Tools:
#   1. answer_concept_question  → run_qa_agent()
#   2. generate_study_plan      → run_study_planner() (after run_calendar_agent())
#   3. get_calendar_events      → run_calendar_agent()
#
# Requirements: 5.1, 5.3

SUPERVISOR_TOOL_CONFIG = {
    "tools": [
        {
            "toolSpec": {
                "name": "answer_concept_question",
                "description": (
                    "Answer a concept question using ingested course materials. "
                    "Call this for any question about AI concepts, course topics, "
                    "or subject-matter content that can be grounded in the course materials."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The concept question to answer",
                            }
                        },
                        "required": ["question"],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "generate_study_plan",
                "description": (
                    "Generate a deadline-aware, day-by-day study plan for a given goal "
                    "and list of topics. Automatically accounts for calendar events by "
                    "halving study hours on busy days. Call this for any study planning "
                    "or scheduling request."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "goal": {
                                "type": "string",
                                "description": "The overarching study goal (e.g. 'Prepare for midterm')",
                            },
                            "topics": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Ordered list of topics to cover, one per study day",
                            },
                            "hours_per_week": {
                                "type": "number",
                                "description": "Total study hours available per week",
                            },
                            "days": {
                                "type": "integer",
                                "description": "Number of study days per week",
                            },
                            "deadline": {
                                "type": "string",
                                "description": "Optional ISO date string YYYY-MM-DD for the last study day",
                            },
                        },
                        "required": ["goal", "topics", "hours_per_week", "days"],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "get_calendar_events",
                "description": (
                    "Retrieve upcoming Google Calendar events for the next N days. "
                    "Returns a list of events with their dates and summaries. "
                    "Call this to check the user's schedule before generating a study plan, "
                    "or when the user asks about their upcoming calendar."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "days_ahead": {
                                "type": "integer",
                                "description": "Number of days ahead to look (default 14)",
                            }
                        },
                        "required": [],
                    }
                },
            }
        },
        {
            "toolSpec": {
                "name": "ingest_file",
                "description": (
                    "Load a course material file (PDF or TXT) from the user's computer "
                    "into the study assistant's knowledge base. Call this whenever the user "
                    "wants to upload, import, or load their notes, slides, textbook, or any "
                    "document so the assistant can answer questions about it. "
                    "Ask the user for the full file path if they haven't provided one."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Full path to the PDF or TXT file to load, e.g. C:\\Users\\Eddie\\notes.pdf",
                            }
                        },
                        "required": ["file_path"],
                    }
                },
            }
        },
    ],
    "toolChoice": {"auto": {}},
}


def dispatch_tool(
    name: str,
    inputs: dict,
    conn: sqlite3.Connection,
    config: "Config",
) -> dict:
    """
    Route a tool call from the supervisor to the appropriate specialist agent.

    This is the "tool router" — the bridge between the supervisor's tool calls
    (from the ``boto3 converse()`` agentic loop) and the actual specialist agent
    functions.  It mirrors the ``dispatch_tool()`` pattern from
    ``multi_agent_research.py``, adapted for the three study-assistant tools.

    Routing table
    -------------
    ``answer_concept_question``
        → ``run_qa_agent(inputs["question"], conn, config)``
        Returns ``{"answer": str, "sources": list[str]}``

    ``generate_study_plan``
        → ``run_calendar_agent()`` first to fetch live calendar events, then
          ``run_study_planner(goal, topics, hours_per_week, days, deadline,
          calendar_events)``
        Returns ``{"plan": list[dict], "warning": str | None}`` where each
        plan entry is a plain dict (DayEntry objects are serialized so the
        result is JSON-safe).

    ``get_calendar_events``
        → ``run_calendar_agent(days_ahead=inputs.get("days_ahead", 14))``
        Returns ``{"events": list[dict]}``

    Error handling
    --------------
    - Each specialist call is wrapped in a ``try/except``.  On any exception,
      the function returns ``{"error": "Tool {name} failed: {e}"}`` so the
      supervisor can surface a user-friendly message without crashing.
    - If the specialist itself returns a dict containing an ``"error"`` key,
      that error is passed through unchanged so the supervisor can handle it.
    - Unknown tool names return ``{"error": "Unknown tool: {name}"}`` immediately.

    Args:
        name:   The tool name requested by the supervisor (one of
                ``"answer_concept_question"``, ``"generate_study_plan"``,
                ``"get_calendar_events"``).
        inputs: The tool input dict from the supervisor's ``converse()``
                response (already parsed from JSON by boto3).
        conn:   Open SQLite connection to the vector store, passed through to
                ``run_qa_agent()`` and ``search_chunks()``.
        config: Runtime configuration, passed through to ``run_qa_agent()``
                and ``embed_text()``.

    Returns:
        A dict containing the specialist's output, or an error dict of the
        form ``{"error": str}`` if the call failed or the tool name is unknown.

    Requirements: 5.1, 5.2, 5.4
    """
    # ------------------------------------------------------------------
    # Route: answer_concept_question → run_qa_agent()
    # ------------------------------------------------------------------
    if name == "answer_concept_question":
        try:
            result = run_qa_agent(inputs["question"], conn, config)
            return result
        except Exception as e:
            return {"error": f"Tool {name} failed: {e}"}

    # ------------------------------------------------------------------
    # Route: generate_study_plan → run_calendar_agent() + run_study_planner()
    # ------------------------------------------------------------------
    elif name == "generate_study_plan":
        try:
            # Fetch live calendar events first so the planner can halve hours
            # on busy days (Requirement 3.2).
            calendar_events = run_calendar_agent()

            # Extract planner inputs from the tool call.
            goal: str = inputs["goal"]
            topics: list = inputs["topics"]
            hours_per_week: float = inputs["hours_per_week"]
            days: int = inputs["days"]
            deadline: str | None = inputs.get("deadline")

            planner_result = run_study_planner(
                goal=goal,
                topics=topics,
                hours_per_week=hours_per_week,
                days=days,
                deadline=deadline,
                calendar_events=calendar_events,
            )

            # Serialize DayEntry objects to plain dicts so the result is
            # JSON-safe when the supervisor sends it back to Bedrock.
            raw_plan = planner_result.get("plan", [])
            serialized_plan: list[dict] = []
            for entry in raw_plan:
                if isinstance(entry, DayEntry):
                    serialized_plan.append(
                        {
                            "date": entry.date,
                            "focus_topic": entry.focus_topic,
                            "hours": entry.hours,
                            "outcome": entry.outcome,
                        }
                    )
                elif isinstance(entry, dict):
                    # Already a dict (e.g. from tests or future refactors).
                    serialized_plan.append(entry)
                else:
                    # Fallback: convert via __dict__ if available.
                    serialized_plan.append(vars(entry))

            return {
                "plan": serialized_plan,
                "warning": planner_result.get("warning"),
            }

        except Exception as e:
            return {"error": f"Tool {name} failed: {e}"}

    # ------------------------------------------------------------------
    # Route: get_calendar_events → run_calendar_agent()
    # ------------------------------------------------------------------
    elif name == "get_calendar_events":
        try:
            days_ahead: int = inputs.get("days_ahead", 14)
            events = run_calendar_agent(days_ahead=days_ahead)
            return {"events": events}
        except Exception as e:
            return {"error": f"Tool {name} failed: {e}"}

    # ------------------------------------------------------------------
    # Route: ingest_file → ingest_document()
    # ------------------------------------------------------------------
    elif name == "ingest_file":
        try:
            file_path: str = inputs["file_path"]
            result = ingest_document(file_path, conn, config)
            return {"result": result}
        except Exception as e:
            return {"error": f"Tool {name} failed: {e}"}

    # ------------------------------------------------------------------
    # Unknown tool name
    # ------------------------------------------------------------------
    else:
        return {"error": f"Unknown tool: {name}"}


def generate_smart_study_plan(
    conn: sqlite3.Connection,
    config: Config,
    exam_date: str,
    calendar_events: list[dict],
    preferred_start_hour: int = 9,
    preferred_end_hour: int = 21,
    max_session_hours: float = 2.0,
    min_session_hours: float = 0.5,
    target_hours_per_week: float = 0.0,  # 0 = use recommended
) -> list[dict]:
    """
    Generate a smart study plan by finding free time slots in the calendar
    and distributing topics from ingested documents across them.

    Research-backed defaults:
    - 2-4 hours of focused study per day is optimal (Pomodoro / spaced repetition)
    - Sessions should be 25-90 minutes with breaks
    - Heavier review sessions 2-3 days before exam
    - Lighter review on day before exam
    """
    import datetime as _dt

    today = _dt.date.today()
    exam = _dt.date.fromisoformat(exam_date)
    days_until_exam = (exam - today).days

    if days_until_exam <= 0:
        return []

    # --- Research-backed recommended hours ---
    # Based on cognitive science: 2h/day for light material, 3h for moderate, 4h for heavy
    # We estimate based on number of topics and document count
    docs = list_documents(conn)
    if not docs:
        return []

    all_chunks = search_chunks("main topics concepts key ideas overview", conn, config, top_n=50)

    doc_chunks: dict[str, list[str]] = {}
    for chunk_text, source_doc, _score in all_chunks:
        if source_doc not in doc_chunks:
            doc_chunks[source_doc] = []
        doc_chunks[source_doc].append(chunk_text)

    # Estimate complexity: more chunks = more complex material
    total_chunks = sum(len(c) for c in doc_chunks.values())
    if total_chunks < 10:
        recommended_daily_hours = 1.5
        complexity = "light"
    elif total_chunks < 30:
        recommended_daily_hours = 2.5
        complexity = "moderate"
    else:
        recommended_daily_hours = 3.5
        complexity = "heavy"

    # Use target if provided, otherwise use recommended
    daily_target = target_hours_per_week / 7.0 if target_hours_per_week > 0 else recommended_daily_hours
    # Cap at 4 hours/day (cognitive science limit for effective studying)
    daily_target = min(daily_target, 4.0)

    # Build topic list with weights (more chunks = more time needed)
    topics = []
    for doc, chunks in doc_chunks.items():
        n_topics = max(1, len(chunks) // 3)
        for i in range(n_topics):
            chunk_preview = chunks[i * 3].strip()[:80].replace('\n', ' ')
            # Weight: topics near exam get review sessions (shorter, lighter)
            topics.append({"doc": doc, "hint": chunk_preview, "weight": 1.0})

    # Add review sessions for last 2 days before exam
    if days_until_exam >= 3:
        topics.append({"doc": "Review", "hint": "Comprehensive review of all topics", "weight": 0.5})
        topics.append({"doc": "Review", "hint": "Light review and practice questions", "weight": 0.3})

    if not topics:
        return []

    # --- Build busy intervals from calendar events ---
    def _to_float(t: str) -> float:
        parts = t.split(":")
        return int(parts[0]) + int(parts[1]) / 60.0

    busy_intervals: dict[str, list[tuple]] = {}
    for event in calendar_events:
        date = event.get("date", "")
        if not date:
            continue
        if event.get("all_day", False):
            busy_intervals.setdefault(date, []).append((0.0, 24.0))
        else:
            s = _to_float(event.get("start_time", "00:00"))
            e = _to_float(event.get("end_time", "00:00"))
            # Add 30-min buffer before and after events
            busy_intervals.setdefault(date, []).append((max(0, s - 0.5), min(24, e + 0.5)))

    def _find_free_slots(date_str, busy, day_start, day_end, min_slot):
        intervals = sorted(busy, key=lambda x: x[0])
        merged: list[list] = []
        for s, e in intervals:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        free = []
        cursor = day_start
        for s, e in merged:
            if s > cursor and (s - cursor) >= min_slot:
                free.append((cursor, s))
            cursor = max(cursor, e)
        if day_end > cursor and (day_end - cursor) >= min_slot:
            free.append((cursor, day_end))
        return free

    # --- Schedule study blocks ---
    study_blocks = []
    topic_idx = 0
    total_topics = len(topics)

    for day_offset in range(days_until_exam):
        if topic_idx >= total_topics:
            break

        study_date = today + _dt.timedelta(days=day_offset)
        date_str = study_date.isoformat()
        days_remaining = days_until_exam - day_offset

        # Ramp up intensity as exam approaches
        if days_remaining <= 1:
            day_cap = min(daily_target * 0.5, 1.5)  # light day before exam
        elif days_remaining <= 3:
            day_cap = min(daily_target * 1.2, 4.0)  # heavier review near exam
        else:
            day_cap = daily_target

        day_busy = busy_intervals.get(date_str, [])
        free_slots = _find_free_slots(
            date_str, day_busy,
            float(preferred_start_hour), float(preferred_end_hour),
            min_session_hours,
        )

        hours_scheduled_today = 0.0

        for slot_start, slot_end in free_slots:
            if topic_idx >= total_topics:
                break
            if hours_scheduled_today >= day_cap:
                break

            current_hour = slot_start
            while (current_hour + min_session_hours <= slot_end
                   and topic_idx < total_topics
                   and hours_scheduled_today < day_cap):

                topic = topics[topic_idx]
                # Shorter sessions for review topics, longer for new material
                ideal_session = max_session_hours * topic["weight"]
                remaining_today = day_cap - hours_scheduled_today
                session_hours = min(ideal_session, slot_end - current_hour, remaining_today)
                session_hours = max(min_session_hours, round(session_hours * 2) / 2)

                start_str = f"{int(current_hour):02d}:{int((current_hour % 1) * 60):02d}"
                end_hour = current_hour + session_hours
                end_str = f"{int(end_hour):02d}:{int((end_hour % 1) * 60):02d}"

                label = f"Study: {topic['doc']} — {topic['hint']}"
                if topic["doc"] == "Review":
                    label = f"Review: {topic['hint']}"

                study_blocks.append({
                    "date": date_str,
                    "start_time": start_str,
                    "end_time": end_str,
                    "topic": label,
                    "duration_hours": session_hours,
                    "selected": True,
                })

                hours_scheduled_today += session_hours
                current_hour = end_hour + 0.5  # 30-min break
                topic_idx += 1

    return study_blocks, {
        "complexity": complexity,
        "recommended_daily_hours": recommended_daily_hours,
        "actual_daily_target": daily_target,
        "total_topics": len(topics),
        "total_sessions": len(study_blocks),
        "total_study_hours": sum(b["duration_hours"] for b in study_blocks),
    }
    """
    Generate a smart study plan by finding free time slots in the calendar
    and distributing topics from ingested documents across them.

    Algorithm:
    1. Extract all topics from ingested documents using the LLM
    2. Build a map of busy time slots from calendar events
    3. Find free slots between now and exam_date
    4. Distribute topics across free slots, heavier material gets more time
    5. Return a list of proposed study blocks with date, start_time, end_time, topic

    Args:
        conn:                  SQLite connection to the vector store
        config:                Runtime configuration
        exam_date:             ISO date string YYYY-MM-DD for the exam/deadline
        calendar_events:       List of {"date": str, "summary": str} dicts
        preferred_start_hour:  Earliest hour to schedule study (default 9am)
        preferred_end_hour:    Latest hour to end study (default 9pm)
        max_session_hours:     Maximum length of a single study session
        min_session_hours:     Minimum length of a single study session

    Returns:
        List of study block dicts:
        {"date": str, "start_time": str, "end_time": str, "topic": str,
         "duration_hours": float, "selected": bool}
    """
    import datetime as _dt

    today = _dt.date.today()
    exam = _dt.date.fromisoformat(exam_date)
    days_until_exam = (exam - today).days

    if days_until_exam <= 0:
        return []

    # --- 1. Get topics from ingested documents ---
    docs = list_documents(conn)
    if not docs:
        return []

    # Use a broad query to retrieve diverse chunks covering all topics
    all_chunks = search_chunks("main topics concepts key ideas overview", conn, config, top_n=50)

    # Extract unique source documents and estimate topic count per doc
    doc_chunks: dict[str, list[str]] = {}
    for chunk_text, source_doc, _score in all_chunks:
        if source_doc not in doc_chunks:
            doc_chunks[source_doc] = []
        doc_chunks[source_doc].append(chunk_text)

    # Build topic list: one entry per ~3 chunks (rough topic granularity)
    topics = []
    for doc, chunks in doc_chunks.items():
        n_topics = max(1, len(chunks) // 3)
        for i in range(n_topics):
            # Use chunk text as topic hint
            chunk_preview = chunks[i * 3].strip()[:60].replace('\n', ' ')
            topics.append({"doc": doc, "hint": chunk_preview, "weight": 1.0})

    if not topics:
        return []

    # --- 2. Build busy time map from calendar events ---
    # Build a dict: date -> list of (start_hour_float, end_hour_float) busy intervals
    busy_intervals: dict[str, list[tuple[float, float]]] = {}
    for event in calendar_events:
        date = event.get("date", "")
        if not date:
            continue
        if event.get("all_day", False):
            # All-day event: mark entire day as busy
            busy_intervals.setdefault(date, []).append((0.0, 24.0))
        else:
            # Parse HH:MM to float hours
            def _to_float(t: str) -> float:
                parts = t.split(":")
                return int(parts[0]) + int(parts[1]) / 60.0

            s = _to_float(event.get("start_time", "00:00"))
            e = _to_float(event.get("end_time", "00:00"))
            busy_intervals.setdefault(date, []).append((s, e))

    # --- 3. Find free slots and distribute topics ---
    study_blocks = []
    topic_idx = 0
    total_topics = len(topics)

    def _to_float(t: str) -> float:
        parts = t.split(":")
        return int(parts[0]) + int(parts[1]) / 60.0

    def _find_free_slots(
        date_str: str,
        busy: list[tuple],
        day_start: float,
        day_end: float,
        min_slot: float,
    ) -> list[tuple]:
        intervals = sorted(busy, key=lambda x: x[0])
        merged: list[list] = []
        for s, e in intervals:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        free = []
        cursor = day_start
        for s, e in merged:
            if s > cursor and (s - cursor) >= min_slot:
                free.append((cursor, s))
            cursor = max(cursor, e)
        if day_end > cursor and (day_end - cursor) >= min_slot:
            free.append((cursor, day_end))
        return free

    for day_offset in range(days_until_exam):
        if topic_idx >= total_topics:
            break

        study_date = today + _dt.timedelta(days=day_offset)
        date_str = study_date.isoformat()

        # Build busy intervals for this day from real event times
        day_busy = []
        for event in calendar_events:
            if event.get("date") != date_str:
                continue
            if event.get("all_day", False):
                day_busy.append((0.0, 24.0))
            else:
                s = _to_float(event.get("start_time", "00:00"))
                e = _to_float(event.get("end_time", "00:00"))
                day_busy.append((s, e))

        free_slots = _find_free_slots(
            date_str, day_busy,
            float(preferred_start_hour), float(preferred_end_hour),
            min_session_hours,
        )

        for slot_start, slot_end in free_slots:
            if topic_idx >= total_topics:
                break
            current_hour = slot_start
            while current_hour + min_session_hours <= slot_end and topic_idx < total_topics:
                topic = topics[topic_idx]
                session_hours = min(max_session_hours, slot_end - current_hour)
                session_hours = max(min_session_hours, round(session_hours * 2) / 2)

                start_str = f"{int(current_hour):02d}:{int((current_hour % 1) * 60):02d}"
                end_hour = current_hour + session_hours
                end_str = f"{int(end_hour):02d}:{int((end_hour % 1) * 60):02d}"

                study_blocks.append({
                    "date": date_str,
                    "start_time": start_str,
                    "end_time": end_str,
                    "topic": f"Study: {topic['doc']} — {topic['hint']}",
                    "duration_hours": session_hours,
                    "selected": True,
                })

                current_hour = end_hour + 0.5
                topic_idx += 1

    return study_blocks


def create_selected_study_sessions(study_blocks: list[dict]) -> list[dict]:
    """
    Create Google Calendar events for selected study blocks.

    Args:
        study_blocks: List of study block dicts with "selected" flag.
                      Only blocks where selected=True are created.

    Returns:
        List of result dicts: {"date": str, "status": "created"|"failed", "detail": str}
    """
    try:
        from google_auth.auth import get_calendar_service
        service = get_calendar_service()
    except Exception as e:
        return [{"date": "N/A", "status": "failed", "detail": f"Calendar not available: {e}"}]

    results = []
    for block in study_blocks:
        if not block.get("selected", True):
            continue
        try:
            event = {
                "summary": block.get("label") or block["topic"],
                "start": {
                    "dateTime": f"{block['date']}T{block['start_time']}:00",
                    "timeZone": "America/Los_Angeles",
                },
                "end": {
                    "dateTime": f"{block['date']}T{block['end_time']}:00",
                    "timeZone": "America/Los_Angeles",
                },
                "colorId": "2",
            }
            created = service.events().insert(calendarId="primary", body=event).execute()
            results.append({
                "date": block["date"],
                "status": "created",
                "detail": f"Created: {created.get('summary')} on {block['date']} {block['start_time']}-{block['end_time']}"
            })
        except Exception as e:
            results.append({"date": block["date"], "status": "failed", "detail": str(e)})

    return results


def run_supervisor_turn(
    messages: list[dict],
    prompt: str,
    conn: sqlite3.Connection,
    config: "Config",
    calendar_events: list[dict] | None = None,
) -> tuple[str, list[dict]]:
    """
    Run one turn of the supervisor agent using the standard boto3 converse()
    agentic loop.

    This function is the primary entry point for each user turn in the chat
    loop.  It mirrors the ``run_supervisor()`` pattern from
    ``multi_agent_research.py`` and the ``run_agent_turn()`` pattern from
    ``study_coach_agent.py``.

    Algorithm
    ---------
    1. Create a Bedrock Runtime client using ``config.region``.
    2. Append the user message to the messages list:
       ``messages = messages + [{"role": "user", "content": [{"text": prompt}]}]``
    3. Enter the agentic loop:
       a. If ``config.debug`` is True, print the full request payload.
       b. Call ``bedrock_client.converse()`` with ``config.model_id`` and
          ``SUPERVISOR_TOOL_CONFIG``.
       c. Append the assistant's output message to ``messages``.
       d. If ``stop_reason == "end_turn"``: extract text from the response,
          return ``(text, messages)``.
       e. If ``stop_reason == "tool_use"``: extract all tool use blocks, call
          ``dispatch_tool()`` for each, build a tool results message, append
          it to ``messages``, and continue the loop.
       f. For any other ``stop_reason``: return
          ``("I encountered an unexpected error.", messages)``.
    4. Return ``(final_text, updated_messages)``.

    Args:
        messages: The current conversation history in Bedrock converse() format.
                  This list is NOT mutated; a new list is built and returned.
        prompt:   The user's message text for this turn.
        conn:     Open SQLite connection to the vector store (passed through to
                  ``dispatch_tool()`` → specialist agents).
        config:   Runtime configuration supplying ``region``, ``model_id``, and
                  ``debug``.

    Returns:
        A tuple ``(response_text, updated_messages)`` where:
          - ``response_text`` is the supervisor's final text response.
          - ``updated_messages`` is the full updated message history including
            this turn's user message, any tool use/result pairs, and the final
            assistant message.

    Requirements: 5.1, 5.3, 5.5
    """
    # Step 1: Create the Bedrock client.
    bedrock_client = boto3.client("bedrock-runtime", region_name=config.region)

    prompt_lower = prompt.lower()
    current_date = datetime.date.today()
    current_date_str = current_date.isoformat()
    current_weekday = current_date.strftime("%A")

    show_calendar_phrases = [
        "show me my google calendar",
        "show me my calendar",
        "show me my calendar events",
        "show me my events",
        "can you show me my google calendar",
        "can you show me my calendar",
        "can you show me my calendar events",
        "what's on my calendar",
        "what is on my calendar",
        "what are my calendar events",
    ]
    study_gap_phrases = [
        "help me fill any gaps i can fill with studying",
        "help me find study gaps",
        "where can i study",
        "when can i study",
        "what gaps can i use for studying",
        "find gaps for studying",
    ]
    calendar_write_phrases = [
        "schedule those",
        "schedule them",
        "please schedule",
        "add those",
        "add them",
        "put them on my calendar",
        "add it to my google calendar",
        "add them to my google calendar",
    ]
    calendar_followup_issue_phrases = [
        "you said",
        "i dont see them",
        "i don't see them",
        "not on my calendar",
        "you added",
        "you've added",
        "you have added",
    ]

    wants_calendar_overview = any(phrase in prompt_lower for phrase in show_calendar_phrases)
    wants_study_gaps = any(phrase in prompt_lower for phrase in study_gap_phrases)
    wants_calendar_write = any(phrase in prompt_lower for phrase in calendar_write_phrases)
    is_calendar_followup_issue = any(phrase in prompt_lower for phrase in calendar_followup_issue_phrases)

    if wants_calendar_write or is_calendar_followup_issue:
        response_text = build_calendar_write_limitation_response()
        updated_messages = messages + [
            {"role": "user", "content": [{"text": prompt}]},
            {"role": "assistant", "content": [{"text": response_text}]},
        ]
        return response_text, updated_messages

    if calendar_events and (wants_calendar_overview or wants_study_gaps):
        response_text = build_calendar_summary_response(calendar_events, days_ahead=7)
        updated_messages = messages + [
            {"role": "user", "content": [{"text": prompt}]},
            {"role": "assistant", "content": [{"text": response_text}]},
        ]
        return response_text, updated_messages

    # Step 2: Automatically retrieve relevant chunks from the vector store.
    # Retrieve more chunks to cover broader document content, and list
    # all loaded documents so the model knows what's available.
    loaded_docs = list_documents(conn)
    loaded_docs_str = ", ".join(loaded_docs) if loaded_docs else "none"

    try:
        chunks = search_chunks(prompt, conn, config, top_n=15)
        if chunks:
            context_blocks = []
            for chunk_text_val, source_doc_val, _score in chunks[:10]:
                context_blocks.append(f"[Source: {source_doc_val}]\n{chunk_text_val}")
            context_text = "\n\n---\n\n".join(context_blocks)

            # Add calendar events if available
            cal_section = ""
            if calendar_events:
                cal_lines = [format_calendar_event_line(e) for e in calendar_events[:20]]
                cal_section = (
                    f"\n\nGoogle Calendar events (next 14 days):\n"
                    + "\n".join(cal_lines)
                    + "\n(Use these to avoid scheduling study sessions on busy days.)"
                )

            augmented_prompt = (
                f"Today's date is {current_weekday}, {current_date_str}.\n\n"
                f"Documents currently loaded in the knowledge base: {loaded_docs_str}\n\n"
                f"Relevant excerpts retrieved from the loaded documents:\n\n"
                f"{context_text}"
                f"{cal_section}\n\n"
                f"---\n\n"
                f"Student's request: {prompt}"
            )
        else:
            cal_section = ""
            if calendar_events:
                cal_lines = [format_calendar_event_line(e) for e in calendar_events[:20]]
                cal_section = (
                    f"\n\nGoogle Calendar events (next 14 days):\n"
                    + "\n".join(cal_lines)
                )
            augmented_prompt = (
                f"Today's date is {current_weekday}, {current_date_str}.\n\n"
                f"Documents currently loaded in the knowledge base: {loaded_docs_str}\n\n"
                f"No relevant excerpts were found for this query."
                f"{cal_section}\n\n"
                f"Student's request: {prompt}"
            )
    except Exception:
        cal_fallback = ""
        if calendar_events:
            cal_lines = [format_calendar_event_line(e) for e in calendar_events[:20]]
            cal_fallback = "\n\nGoogle Calendar events:\n" + "\n".join(cal_lines)
        augmented_prompt = (
            f"Today's date is {current_weekday}, {current_date_str}.\n\n"
            f"Documents currently loaded in the knowledge base: {loaded_docs_str}\n\n"
            f"{cal_fallback}\n\n"
            f"Student's request: {prompt}"
        )

    # Step 3: Append the (augmented) user message.
    messages = messages + [{"role": "user", "content": [{"text": augmented_prompt}]}]
    messages = _sanitize_conversation_messages(messages)

    # System prompt with strict accuracy requirements.
    system_prompt = [{"text": (
        "You are an expert AI study assistant. You help students learn and manage their time.\n\n"
        "You have access to two types of information:\n"
        "1. Course material excerpts from uploaded documents\n"
        "2. Google Calendar events (if connected)\n\n"
        "Rules for course material:\n"
        "- Quote exact wording from excerpts when explaining concepts\n"
        "- Cite which document a concept comes from\n"
        "- If a concept is not in the excerpts, say so clearly\n"
        "- For quizzes: base every question on a specific excerpt\n"
        "- For study plans: cover ALL topics found across ALL excerpts\n\n"
        "Rules for calendar:\n"
        f"- Today's date is {current_weekday}, {current_date_str}; use that exact date for all planning and date questions\n"
        "- The Google Calendar events provided ARE the user's personal calendar — treat them as such\n"
        "- If calendar events are provided in the message, you CAN and SHOULD display and discuss them\n"
        "- If the user asks to see their calendar, list ALL the events from the provided calendar data with their times in AM/PM format\n"
        "- If you suggest study sessions from calendar gaps, propose realistic focused sessions that are usually 1-2 hours long rather than using the entire free window\n"
        "- When turning gaps into a study plan, use the course material topics as the session content and keep each session specific and manageable\n"
        "- For study-plan requests, use calendar events only to avoid conflicts; do NOT treat existing calendar events as the answer unless the user explicitly asks to review what is already scheduled\n"
        "- For study-plan requests, output each session on its own line in this exact format: 'Month Day: Topic (X hours)' — for example: 'June 10: Fixed Size Chunking (1.5 hours)'. Always include the hours in parentheses. Do not use clock times unless the user explicitly asks for them.\n"
        "- Never propose study sessions outside the user's preferred daytime scheduling window when a planner import will handle slotting later\n"
        "- NEVER claim that you already added, created, or scheduled Google Calendar events unless a tool result explicitly confirms it\n"
        "- If the user asks to CREATE, ADD, or SCHEDULE calendar events, explain that chat suggests sessions but the actual calendar creation is done with the app's Add to Google Calendar controls\n"
        "- NEVER say you don't have access to the calendar if calendar events are provided in the message\n"
        "- If no calendar events are in the message, explain the calendar is not connected\n\n"
        "Be encouraging, accurate, and helpful."
    )}]

    # Step 4: Agentic loop.
    while True:
        # Build the request payload.
        request_payload = {
            "modelId": config.model_id,
            "messages": messages,
            "system": system_prompt,
        }

        # Add tool config only if the model supports it (Claude 3 Haiku does not
        # support tool use via converse(); newer models do).
        _models_with_tools = {
            "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "us.anthropic.claude-3-5-haiku-20241022-v1:0",
            "anthropic.claude-3-5-haiku-20241022-v1:0",
        }
        if config.model_id in _models_with_tools:
            request_payload["toolConfig"] = SUPERVISOR_TOOL_CONFIG

        # Debug: print the full request payload before each API call
        # (Requirement 8.4).
        if config.debug:
            print(json.dumps(request_payload, indent=2, default=str))

        # Call Bedrock converse().
        response = bedrock_client.converse(**request_payload)

        stop_reason: str = response.get("stopReason", "")
        output_message: dict = response["output"]["message"]

        # Append the assistant's output message to the history.
        output_message = _sanitize_conversation_messages([output_message])
        if not output_message:
            return "I hit a formatting error while generating that response. Please try again.", messages
        output_message = output_message[0]
        messages = messages + [output_message]

        if stop_reason == "end_turn":
            # Extract all text blocks from the response and join them.
            text_parts = [
                block["text"]
                for block in output_message.get("content", [])
                if "text" in block
            ]
            final_text = "\n".join(text_parts).strip()
            return final_text, messages

        elif stop_reason == "tool_use":
            # Extract all tool use blocks from the assistant's response.
            tool_uses = [
                block["toolUse"]
                for block in output_message.get("content", [])
                if "toolUse" in block
            ]

            # Dispatch each tool call and collect results.
            tool_results: list[dict] = []
            for tool_use in tool_uses:
                tool_name: str = tool_use["name"]
                tool_input: dict = tool_use.get("input", {})

                result_data = dispatch_tool(tool_name, tool_input, conn, config)

                tool_results.append({
                    "toolResult": {
                        "toolUseId": tool_use["toolUseId"],
                        "content": [{"json": result_data}],
                        "status": "success" if "error" not in result_data else "error",
                    }
                })

            # Append the tool results as a user message and continue the loop.
            messages = messages + [{"role": "user", "content": tool_results}]
            continue

        else:
            # Unexpected stop reason — return a safe error message and the
            # current history so the session can continue (Requirement 5.4).
            return "I encountered an unexpected error.", messages


# =============================================================================
# CLI — Chat Loop
# =============================================================================

def chat_loop(initial_prompt: str, conn: sqlite3.Connection, config: Config) -> None:
    """
    Run the multi-turn conversation loop until the user types "exit" or "quit".

    Algorithm
    ---------
    1. Create a Bedrock Runtime client for history compression.
    2. Initialize ``SessionMemory``.
    3. If ``initial_prompt`` is non-empty and non-whitespace, process it as the
       first turn without prompting the user for input.
    4. Main loop:
       a. If no initial prompt was processed this iteration: read input from
          stdin with ``input("You: ")``.
       b. Strip the input.  If empty/whitespace, print ``"Please enter a message."``
          and continue.
       c. If ``user_input.lower()`` is ``"exit"`` or ``"quit"``, print the
          session summary and break.
       d. Call ``run_supervisor_turn()`` to get the response and updated history.
       e. If the response text mentions "plan" (case-insensitive), store the
          response in ``memory.last_study_plan`` as a plain dict so it can be
          referenced on follow-up turns (Requirement 6.3).
       f. Call ``compress_history()`` to keep the history manageable (Req 6.4).
       g. Print the formatted turn via ``format_turn()``.
    5. Session summary: ``f"Session ended. {len(memory.messages)} messages exchanged."``
    6. The entire loop is wrapped in a try/except; on any unhandled exception
       the exception is re-raised so the session ends without printing a summary
       (Requirement 1.3).

    Args:
        initial_prompt: An optional first message to send without prompting the
                        user.  Pass an empty string (or whitespace) to skip the
                        initial turn and go straight to the interactive loop.
        conn:           Open SQLite connection to the vector store.
        config:         Runtime configuration (region, model_id, debug flag, etc.).

    Returns:
        None.

    Requirements: 1.1, 1.2, 1.3, 1.4, 6.1, 6.2, 6.3, 6.4
    """
    # Step 1: Create a Bedrock client for history compression.
    bedrock_client_for_compress = boto3.client(
        "bedrock-runtime", region_name=config.region
    )

    # Step 2: Initialize session memory.
    memory = SessionMemory()

    # Track whether the initial_prompt has already been consumed so we know
    # whether to call input() on the first iteration.
    initial_prompt_consumed = False

    try:
        while True:
            # ------------------------------------------------------------------
            # Step 3 / 4a: Determine the user input for this turn.
            # ------------------------------------------------------------------
            if not initial_prompt_consumed and initial_prompt.strip():
                # Use the CLI-supplied initial prompt as the first user message.
                user_input = initial_prompt.strip()
                initial_prompt_consumed = True
                # Echo the prompt so the user can see what was sent.
                print(f"You: {user_input}\n")
            else:
                # Interactive turn: read from stdin.
                initial_prompt_consumed = True  # ensure we always read after first turn
                try:
                    raw = input("You: ")
                except EOFError:
                    # Non-interactive environment (e.g. piped input exhausted).
                    break
                user_input = raw.strip()

            # ------------------------------------------------------------------
            # Step 4b: Reject empty / whitespace-only input.
            # ------------------------------------------------------------------
            if not user_input:
                print("Please enter a message.")
                continue

            # ------------------------------------------------------------------
            # Step 4c: Handle exit / quit.
            # ------------------------------------------------------------------
            if user_input.lower() in {"exit", "quit"}:
                print(f"Session ended. {len(memory.messages)} messages exchanged.")
                break

            # ------------------------------------------------------------------
            # Step 4d: Run the supervisor turn.
            # ------------------------------------------------------------------
            response_text, memory.messages = run_supervisor_turn(
                memory.messages, user_input, conn, config
            )

            # ------------------------------------------------------------------
            # Step 4e: Store the study plan in session memory if the response
            #          mentions a plan (Requirement 6.3).
            # ------------------------------------------------------------------
            if "plan" in response_text.lower():
                memory.last_study_plan = {"response": response_text}

            # ------------------------------------------------------------------
            # Step 4f: Compress history if it has grown beyond 20 messages.
            # ------------------------------------------------------------------
            memory = compress_history(memory, bedrock_client_for_compress)

            # ------------------------------------------------------------------
            # Step 4g: Print the formatted turn.
            # ------------------------------------------------------------------
            print(format_turn(user_input, response_text))

    except (KeyboardInterrupt, EOFError):
        # User pressed Ctrl-C or input stream closed — exit silently without
        # printing the session summary (Requirement 1.3).
        raise

    except Exception:
        # Any other unhandled exception: re-raise so the session ends without
        # printing a summary (Requirement 1.3).
        raise


# =============================================================================
# CLI — Entry Point
# =============================================================================

def main() -> None:
    """
    Parse command-line arguments and start the AI Study Assistant.

    Command-line interface
    ----------------------
    usage: study_assistant.py [prompt] [--ingest FILE_PATH] [--debug]

    positional arguments:
      prompt            Optional initial prompt to send as the first message.
                        Defaults to a standard study planning prompt when omitted.

    optional arguments:
      --ingest FILE_PATH
                        Ingest a course material file (.pdf or .txt) into the
                        vector store before starting the chat loop.
      --debug           Print the full Bedrock request payload before each API
                        call (useful for debugging tool use and model inputs).

    Environment variables read by Config()
    --------------------------------------
      AWS_REGION                  → config.region          (default: "us-east-1")
      STUDY_ASSISTANT_MODEL_ID    → config.model_id        (default: "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
      VECTOR_STORE_PATH           → config.vector_store_path (default: "study_assistant.db")

    Algorithm
    ---------
    1. Parse args with argparse.
    2. Build Config() — reads from environment variables automatically.
    3. If --debug: set config.debug = True.
    4. Initialize the vector store via init_vector_store(config.vector_store_path).
    5. If --ingest: call ingest_document() and print the result.
    6. Determine the initial prompt (CLI arg or default).
    7. Start chat_loop(initial_prompt, conn, config).

    Requirements: 1.5, 4.1, 8.1, 8.2, 8.3, 8.4, 8.5
    """
    # ------------------------------------------------------------------
    # 1. Parse command-line arguments
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="AI Study Assistant — terminal-based study tool"
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="",
        help="Optional initial prompt to send as the first message",
    )
    parser.add_argument(
        "--ingest",
        metavar="FILE_PATH",
        help="Ingest a course material file (.pdf or .txt) before starting",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print full Bedrock request payload before each API call",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 2. Build Config — reads AWS_REGION, STUDY_ASSISTANT_MODEL_ID,
    #    and VECTOR_STORE_PATH from environment variables automatically.
    # ------------------------------------------------------------------
    config = Config()

    # ------------------------------------------------------------------
    # 3. Apply --debug flag
    # ------------------------------------------------------------------
    if args.debug:
        config.debug = True

    # ------------------------------------------------------------------
    # 4. Initialize the SQLite vector store
    # ------------------------------------------------------------------
    conn = init_vector_store(config.vector_store_path)

    # ------------------------------------------------------------------
    # 5. Optionally ingest a document before starting the chat loop
    # ------------------------------------------------------------------
    if args.ingest:
        result = ingest_document(args.ingest, conn, config)
        print(result)

    # ------------------------------------------------------------------
    # 6. Determine the initial prompt
    # ------------------------------------------------------------------
    default_prompt = "I need help planning my studies. What can you help me with?"
    initial_prompt = args.prompt.strip() if args.prompt.strip() else default_prompt

    # ------------------------------------------------------------------
    # 7. Start the interactive chat loop
    # ------------------------------------------------------------------
    chat_loop(initial_prompt, conn, config)

if __name__ == "__main__":
    main()
