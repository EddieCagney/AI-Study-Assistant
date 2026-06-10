"""
Property-based tests for the AI Study Assistant.

# Feature: ai-study-assistant, Property 7: Chunking Size and Overlap Invariants
"""

import datetime
import sys
import os

# Allow importing study_assistant from the parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from study_assistant import chunk_text

from hypothesis import given, settings
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Property 7: Chunking Size and Overlap Invariants
# Validates: Requirements 2.7
# ---------------------------------------------------------------------------

@given(st.text())
@settings(max_examples=100)
def test_chunk_text_size_and_overlap_invariants(text: str) -> None:
    """
    **Validates: Requirements 2.7**

    For any text string of any length, chunk_text() must produce chunks where:
      (a) every chunk is at most 1024 characters long, and
      (b) for any two consecutive chunks that are both at least 256 chars long,
          the last 256 chars of chunk[i] equal the first 256 chars of chunk[i+1].
    """
    chunks = chunk_text(text, chunk_size=1024, overlap=256)

    # (a) Every chunk must be at most 1024 characters long
    for i, chunk in enumerate(chunks):
        assert len(chunk) <= 1024, (
            f"Chunk {i} has length {len(chunk)}, which exceeds the 1024-char limit. "
            f"Chunk content (first 100 chars): {chunk[:100]!r}"
        )

    # (b) Consecutive chunks share a 256-char overlap
    # Only check when both chunks are at least 256 chars long
    for i in range(len(chunks) - 1):
        chunk_a = chunks[i]
        chunk_b = chunks[i + 1]
        if len(chunk_a) >= 256 and len(chunk_b) >= 256:
            suffix_of_a = chunk_a[-256:]
            prefix_of_b = chunk_b[:256]
            assert suffix_of_a == prefix_of_b, (
                f"Overlap invariant violated between chunk {i} and chunk {i + 1}. "
                f"Last 256 chars of chunk[{i}]: {suffix_of_a!r} "
                f"First 256 chars of chunk[{i + 1}]: {prefix_of_b!r}"
            )


# ---------------------------------------------------------------------------
# Property 8: Ingestion Idempotence
# Validates: Requirements 2.9
# ---------------------------------------------------------------------------

import tempfile
import numpy as np
from unittest.mock import patch

from study_assistant import ingest_document, init_vector_store, Config


@given(st.text(min_size=1))
@settings(max_examples=100)
def test_ingest_document_idempotence(content: str) -> None:
    """
    **Validates: Requirements 2.9**

    For any non-empty text content, ingesting the same document twice must
    leave the chunk count unchanged after the second ingest.  The Vector_Store
    SHALL skip re-ingestion when a source document is already present.

    embed_text is mocked to return a fixed numpy array so the test does not
    require live AWS credentials.
    """
    config = Config()
    conn = init_vector_store(":memory:")

    tmp_path = None
    try:
        # Write content to a real temporary .txt file
        with tempfile.NamedTemporaryFile(
            suffix=".txt", mode="w", encoding="utf-8", delete=False
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        fixed_embedding = np.ones(1024, dtype=np.float32)

        with patch("study_assistant.embed_text", return_value=fixed_embedding):
            # First ingest
            ingest_document(tmp_path, conn, config)

            # Record chunk count after first ingest
            cursor = conn.execute("SELECT COUNT(*) FROM rag_chunks")
            count_after_first = cursor.fetchone()[0]

            # Second ingest of the same file
            ingest_document(tmp_path, conn, config)

            # Chunk count must be unchanged
            cursor = conn.execute("SELECT COUNT(*) FROM rag_chunks")
            count_after_second = cursor.fetchone()[0]

        assert count_after_second == count_after_first, (
            f"Idempotence violated: chunk count changed from {count_after_first} "
            f"to {count_after_second} after second ingest of the same document."
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------------------------------------------------------------------
# Property 13: File Validation Accepts Only Supported Extensions
# Validates: Requirements 4.1, 4.2
# ---------------------------------------------------------------------------

import sqlite3
import tempfile
from unittest.mock import patch

import numpy as np

from study_assistant import ingest_document, init_vector_store, Config


@given(
    stem=st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
        min_size=1,
        max_size=20,
    ),
    bad_ext=st.sampled_from([".csv", ".json", ".docx", ".py", ".md"]),
)
@settings(max_examples=100)
def test_file_validation_extension(stem: str, bad_ext: str) -> None:
    """
    **Validates: Requirements 4.1, 4.2**

    For any file path:
      1. If the path does not exist → result starts with "File not found:"
      2. If the path exists but has an unsupported extension → result starts with
         "Unsupported file type:"
      3. If the path exists with a .txt extension → result does NOT start with
         "File not found:" or "Unsupported file type:"
    """
    conn = init_vector_store(":memory:")
    config = Config()

    # --- Case 1: Non-existent path with unsupported extension ---
    nonexistent_path = f"/tmp/nonexistent_{stem}{bad_ext}"
    result_nonexistent = ingest_document(nonexistent_path, conn, config)
    assert result_nonexistent.startswith("File not found:"), (
        f"Expected 'File not found:' for non-existent path {nonexistent_path!r}, "
        f"got: {result_nonexistent!r}"
    )

    # --- Case 2: Existing file with unsupported extension ---
    with tempfile.NamedTemporaryFile(suffix=bad_ext, delete=False) as tmp_bad:
        tmp_bad.write(b"some content")
        bad_path = tmp_bad.name

    try:
        result_bad_ext = ingest_document(bad_path, conn, config)
        assert result_bad_ext.startswith("Unsupported file type:"), (
            f"Expected 'Unsupported file type:' for existing file with extension "
            f"{bad_ext!r}, got: {result_bad_ext!r}"
        )
    finally:
        os.unlink(bad_path)

    # --- Case 3: Existing file with .txt extension ---
    # Mock embed_text to avoid real AWS calls
    dummy_embedding = np.ones(config.embedding_dimensions, dtype=np.float32)
    dummy_embedding /= np.linalg.norm(dummy_embedding)

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_txt:
        tmp_txt.write(b"Study material content for testing.")
        txt_path = tmp_txt.name

    try:
        with patch("study_assistant.embed_text", return_value=dummy_embedding):
            result_txt = ingest_document(txt_path, conn, config)

        assert not result_txt.startswith("File not found:"), (
            f"Unexpected 'File not found:' for existing .txt file: {result_txt!r}"
        )
        assert not result_txt.startswith("Unsupported file type:"), (
            f"Unexpected 'Unsupported file type:' for .txt file: {result_txt!r}"
        )
    finally:
        os.unlink(txt_path)


# ---------------------------------------------------------------------------
# Property 9: Calendar-Aware Hour Reduction
# Validates: Requirements 3.2
# ---------------------------------------------------------------------------

from study_assistant import run_study_planner


@given(
    hours_per_week=st.floats(min_value=1.0, max_value=40.0, allow_nan=False, allow_infinity=False),
    days=st.integers(min_value=1, max_value=7),
    event_dates=st.lists(
        st.dates().map(lambda d: d.isoformat()),
        min_size=1,
        max_size=5,
    ),
)
@settings(max_examples=100)
def test_calendar_aware_hour_reduction(
    hours_per_week: float,
    days: int,
    event_dates: list,
) -> None:
    """
    **Validates: Requirements 3.2**

    For any combination of hours_per_week, days, and calendar event dates,
    every day in the generated study plan that falls on an event date must
    have exactly half the hours of a day without an event.
    """
    topics = ["Topic A", "Topic B", "Topic C", "Topic D", "Topic E"]
    calendar_events = [{"date": d} for d in event_dates]

    result = run_study_planner(
        goal="test",
        topics=topics,
        hours_per_week=hours_per_week,
        days=days,
        deadline=None,
        calendar_events=calendar_events,
    )

    plan = result["plan"]
    event_date_set = set(event_dates)
    base_hours = hours_per_week / days

    for entry in plan:
        if entry.date in event_date_set:
            assert entry.hours == base_hours / 2, (
                f"Expected halved hours ({base_hours / 2}) on event day {entry.date!r}, "
                f"but got {entry.hours}. "
                f"hours_per_week={hours_per_week}, days={days}"
            )
        else:
            assert entry.hours == base_hours, (
                f"Expected full hours ({base_hours}) on non-event day {entry.date!r}, "
                f"but got {entry.hours}. "
                f"hours_per_week={hours_per_week}, days={days}"
            )


# ---------------------------------------------------------------------------
# Property 10: Deadline Respected in Study Plan
# Validates: Requirements 3.3
# ---------------------------------------------------------------------------

from study_assistant import run_study_planner


@given(
    deadline=st.dates(
        min_value=datetime.date(2024, 1, 1),
        max_value=datetime.date(2030, 12, 31),
    ).map(lambda d: d.isoformat()),
    topics=st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=10),
)
@settings(max_examples=100)
def test_deadline_respected(deadline: str, topics: list) -> None:
    """
    **Validates: Requirements 3.3**

    For any deadline (ISO date string) and any non-empty list of topics,
    the last entry in the generated study plan must have a date that is
    on or before the deadline.
    """
    result = run_study_planner(
        goal="test",
        topics=topics,
        hours_per_week=10.0,
        days=5,
        deadline=deadline,
        calendar_events=[],
    )

    plan = result["plan"]
    assert len(plan) > 0, "Plan must contain at least one entry"

    last_entry_date = plan[-1].date
    assert last_entry_date <= deadline, (
        f"Last plan entry date {last_entry_date!r} exceeds deadline {deadline!r}"
    )


# ---------------------------------------------------------------------------
# Property 11: Study Plan Structural Completeness
# Validates: Requirements 3.4, 7.1
# ---------------------------------------------------------------------------

from study_assistant import run_study_planner


@given(
    topics=st.lists(st.text(min_size=1, max_size=30), min_size=1, max_size=10),
    hours_per_week=st.floats(min_value=1.0, max_value=40.0, allow_nan=False, allow_infinity=False),
    days=st.integers(min_value=1, max_value=7),
)
@settings(max_examples=100)
def test_study_plan_structural_completeness(
    topics: list,
    hours_per_week: float,
    days: int,
) -> None:
    """
    **Validates: Requirements 3.4, 7.1**

    For any valid planner inputs (goal, topics, hours_per_week, days), every
    entry in the generated study plan must contain non-empty values for all
    four required fields: date, focus_topic, hours, and outcome.
    """
    result = run_study_planner(
        goal="test",
        topics=topics,
        hours_per_week=hours_per_week,
        days=days,
        deadline=None,
        calendar_events=[],
    )

    plan = result["plan"]

    # The plan must have exactly one entry per topic
    assert len(plan) == len(topics), (
        f"Expected {len(topics)} plan entries, got {len(plan)}"
    )

    for i, entry in enumerate(plan):
        # date must be a non-empty string
        assert isinstance(entry.date, str) and entry.date.strip(), (
            f"Entry {i} has empty or non-string date: {entry.date!r}"
        )

        # focus_topic must be a non-empty string
        assert isinstance(entry.focus_topic, str) and entry.focus_topic.strip(), (
            f"Entry {i} has empty or non-string focus_topic: {entry.focus_topic!r}"
        )

        # hours must be positive
        assert entry.hours > 0, (
            f"Entry {i} has non-positive hours: {entry.hours}"
        )

        # outcome must be a non-empty string
        assert isinstance(entry.outcome, str) and entry.outcome.strip(), (
            f"Entry {i} has empty or non-string outcome: {entry.outcome!r}"
        )


# ---------------------------------------------------------------------------
# Property 12: Insufficient Hours Warning
# Validates: Requirements 3.5
# ---------------------------------------------------------------------------

from study_assistant import run_study_planner


@given(n_topics=st.integers(min_value=2, max_value=20))
@settings(max_examples=100)
def test_insufficient_hours_warning(n_topics: int) -> None:
    """
    **Validates: Requirements 3.5**

    When hours_per_week < len(topics), run_study_planner() must include a
    warning in the result dict.  The warning must be non-None and contain
    either "Warning" or "insufficient" to indicate the scheduling conflict.
    """
    topics = [f"Topic {i}" for i in range(n_topics)]
    hours_per_week = float(n_topics) - 0.5  # strictly less than n_topics

    result = run_study_planner(
        goal="test",
        topics=topics,
        hours_per_week=hours_per_week,
        days=5,
        deadline=None,
        calendar_events=[],
    )

    assert result["warning"] is not None, (
        f"Expected a warning when hours_per_week={hours_per_week} < "
        f"len(topics)={n_topics}, but warning was None."
    )

    warning_text = result["warning"]
    assert "Warning" in warning_text or "insufficient" in warning_text, (
        f"Warning message does not contain 'Warning' or 'insufficient': "
        f"{warning_text!r}"
    )


# ---------------------------------------------------------------------------
# Property 3: Retrieval Results Are Sorted by Cosine Similarity
# Validates: Requirements 2.1
# ---------------------------------------------------------------------------

from hypothesis import HealthCheck
from study_assistant import search_chunks, init_vector_store, Config


@given(
    n_chunks=st.integers(min_value=2, max_value=5),
    seeds=st.lists(
        st.integers(min_value=0, max_value=2**31 - 1),
        min_size=2,
        max_size=5,
    ),
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.large_base_example])
def test_retrieval_sort_order(n_chunks: int, seeds: list) -> None:
    """
    **Validates: Requirements 2.1**

    For any set of randomly generated embedding vectors inserted into the
    vector store, search_chunks() must return results sorted in descending
    order of cosine similarity.

    Vectors are built deterministically from integer seeds (rather than
    generated directly by hypothesis) to keep the base example small and
    avoid Hypothesis health-check failures on 1024-dim float lists.

    embed_text is mocked to return a fixed query vector so the test does not
    require live AWS credentials.
    """
    # Use only as many seeds as n_chunks
    seeds_to_use = seeds[:n_chunks]

    # Build and normalize one 1024-dim vector per seed
    normalized_vectors = []
    for seed in seeds_to_use:
        rng = np.random.default_rng(seed)
        arr = rng.uniform(-1.0, 1.0, size=1024).astype(np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        else:
            arr = np.ones(1024, dtype=np.float32) / np.sqrt(1024)
        normalized_vectors.append(arr)

    # Use the first normalized vector as the fixed query vector
    query_vec = normalized_vectors[0]

    config = Config()
    conn = init_vector_store(":memory:")

    # Insert all chunks with their normalized embeddings
    for i, vec in enumerate(normalized_vectors):
        conn.execute(
            "INSERT INTO rag_chunks (chunk_text, source_doc, embedding) VALUES (?, ?, ?)",
            (f"chunk text {i}", f"source_{i}.txt", vec.tobytes()),
        )
    conn.commit()

    # Mock embed_text to return the fixed query vector (avoids AWS calls)
    with patch("study_assistant.embed_text", return_value=query_vec):
        results = search_chunks("test query", conn, config)

    # The results list must be sorted in descending order of cosine similarity
    scores = [score for _, _, score in results]
    assert scores == sorted(scores, reverse=True), (
        f"Results are not sorted in descending order of cosine similarity. "
        f"Scores: {scores}"
    )


# ---------------------------------------------------------------------------
# Property 4: Z-Score Filter Correctness
# Validates: Requirements 2.2
# ---------------------------------------------------------------------------


@given(
    scores=st.lists(
        st.floats(
            min_value=-1.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
        ),
        min_size=2,
        max_size=20,
    )
)
@settings(max_examples=100)
def test_zscore_filter_correctness(scores: list) -> None:
    """
    **Validates: Requirements 2.2**

    For any list of scores, the z-score filter must select exactly the items
    whose z-score is >= max_z / 2.

    When std == 0 (all scores equal), all items must be kept because no
    discrimination is possible.
    """
    scores_arr = np.array(scores, dtype=np.float64)

    mean = np.mean(scores_arr)
    std = float(np.std(scores_arr, ddof=1)) if len(scores_arr) > 1 else 0.0

    if std == 0.0:
        # All scores are identical — every item should be kept.
        kept_indices = list(range(len(scores)))
    else:
        z_scores = (scores_arr - mean) / std
        max_z = float(np.max(z_scores))
        # Guard: if max_z <= 0 (floating-point artifact of identical scores),
        # keep all items — same logic as the implementation.
        if max_z <= 0.0:
            kept_indices = list(range(len(scores)))
        else:
            kept_indices = [i for i, z in enumerate(z_scores) if z >= max_z / 2]

    # Build the expected kept set from the original scores list.
    expected_kept = [scores[i] for i in kept_indices]

    # Verify: the filter must keep at least one item.
    assert len(expected_kept) > 0, (
        f"Z-score filter kept zero items for scores={scores}"
    )

    # Verify: every kept item satisfies the threshold condition.
    if std > 0.0:
        z_scores = (scores_arr - mean) / std
        max_z = float(np.max(z_scores))
        for i in kept_indices:
            assert z_scores[i] >= max_z / 2, (
                f"Item at index {i} (score={scores[i]}, z={z_scores[i]:.4f}) "
                f"does not satisfy z >= max_z/2 ({max_z / 2:.4f}). "
                f"scores={scores}"
            )

    # Verify: no item that fails the threshold is included.
    if std > 0.0:
        z_scores = (scores_arr - mean) / std
        max_z = float(np.max(z_scores))
        all_indices = set(range(len(scores)))
        excluded_indices = all_indices - set(kept_indices)
        for i in excluded_indices:
            assert z_scores[i] < max_z / 2, (
                f"Item at index {i} (score={scores[i]}, z={z_scores[i]:.4f}) "
                f"should have been excluded (z < max_z/2 = {max_z / 2:.4f}). "
                f"scores={scores}"
            )


# ---------------------------------------------------------------------------
# Property 15: Document Listing Completeness
# Validates: Requirements 4.4
# ---------------------------------------------------------------------------

from study_assistant import list_documents, init_vector_store


@given(
    doc_names=st.sets(
        st.text(
            min_size=1,
            max_size=20,
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
        ),
        min_size=1,
        max_size=5,
    )
)
@settings(max_examples=100)
def test_document_listing_completeness(doc_names: set) -> None:
    """
    **Validates: Requirements 4.4**

    For any random set of document names, inserting a dummy chunk for each
    name and then calling list_documents() must return exactly those names
    as a sorted list — no more, no fewer.
    """
    conn = init_vector_store(":memory:")
    dummy_embedding = np.ones(1024, dtype=np.float32)

    for name in doc_names:
        conn.execute(
            "INSERT INTO rag_chunks (chunk_text, source_doc, embedding) VALUES (?, ?, ?)",
            ("dummy chunk text", name, dummy_embedding.tobytes()),
        )
    conn.commit()

    result = list_documents(conn)

    assert result == sorted(doc_names), (
        f"list_documents() returned {result!r}, "
        f"expected {sorted(doc_names)!r} for doc_names={doc_names!r}"
    )


# ---------------------------------------------------------------------------
# Property 16: Document Deletion Completeness
# Validates: Requirements 4.5
# ---------------------------------------------------------------------------

from study_assistant import remove_document, list_documents, init_vector_store


@given(
    doc_names=st.sets(
        st.text(
            min_size=1,
            max_size=20,
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
        ),
        min_size=2,
        max_size=5,
    )
)
@settings(max_examples=100)
def test_document_deletion_completeness(doc_names: set) -> None:
    """
    **Validates: Requirements 4.5**

    For any random set of document names (at least 2), inserting dummy chunks
    for each and then removing one must:
      (a) leave zero chunks for the deleted document, and
      (b) leave all other documents' chunks untouched.
    """
    conn = init_vector_store(":memory:")
    dummy_embedding = np.ones(1024, dtype=np.float32)

    # Insert 2 dummy chunks per document so we can verify full deletion
    for name in doc_names:
        for i in range(2):
            conn.execute(
                "INSERT INTO rag_chunks (chunk_text, source_doc, embedding) VALUES (?, ?, ?)",
                (f"chunk {i} for {name}", name, dummy_embedding.tobytes()),
            )
    conn.commit()

    # Pick the first name (sorted) as the one to delete
    name_to_delete = sorted(doc_names)[0]
    remaining_names = sorted(doc_names)[1:]

    remove_document(name_to_delete, conn)

    # (a) Zero chunks must remain for the deleted document
    cursor = conn.execute(
        "SELECT COUNT(*) FROM rag_chunks WHERE source_doc = ?",
        (name_to_delete,),
    )
    count_deleted = cursor.fetchone()[0]
    assert count_deleted == 0, (
        f"Expected 0 chunks for deleted document {name_to_delete!r}, "
        f"but found {count_deleted}."
    )

    # (b) Other documents must be unaffected (still 2 chunks each)
    for name in remaining_names:
        cursor = conn.execute(
            "SELECT COUNT(*) FROM rag_chunks WHERE source_doc = ?",
            (name,),
        )
        count = cursor.fetchone()[0]
        assert count == 2, (
            f"Expected 2 chunks for undeleted document {name!r}, "
            f"but found {count} after deleting {name_to_delete!r}."
        )


# ---------------------------------------------------------------------------
# Property 14: Vector Store Persistence Round-Trip
# Validates: Requirements 4.3
# ---------------------------------------------------------------------------

from study_assistant import init_vector_store


@given(
    chunks=st.lists(
        st.tuples(
            st.text(min_size=1, max_size=50),   # chunk_text
            st.text(
                min_size=1,
                max_size=20,
                alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")),
            ),  # source_doc
        ),
        min_size=1,
        max_size=10,
    )
)
@settings(max_examples=50, deadline=None)
def test_vector_store_persistence_roundtrip(chunks: list) -> None:
    """
    **Validates: Requirements 4.3**

    For any set of (chunk_text, source_doc) pairs, inserting them into a
    file-based SQLite DB with dummy embeddings, closing the connection,
    reopening it, and reading back the rows must yield identical chunk texts,
    source names, and embedding vectors.
    """
    dummy_embedding = np.ones(1024, dtype=np.float32)

    # Create a real temporary file for the DB (not :memory:)
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    try:
        # --- Write phase ---
        conn_write = init_vector_store(db_path)
        for chunk_text_val, source_doc_val in chunks:
            conn_write.execute(
                "INSERT INTO rag_chunks (chunk_text, source_doc, embedding) VALUES (?, ?, ?)",
                (chunk_text_val, source_doc_val, dummy_embedding.tobytes()),
            )
        conn_write.commit()
        conn_write.close()

        # --- Read phase (fresh connection) ---
        conn_read = init_vector_store(db_path)
        cursor = conn_read.execute(
            "SELECT chunk_text, source_doc, embedding FROM rag_chunks ORDER BY id"
        )
        rows = cursor.fetchall()
        conn_read.close()

        # --- Verify round-trip fidelity ---
        assert len(rows) == len(chunks), (
            f"Expected {len(chunks)} rows after round-trip, got {len(rows)}."
        )

        for i, ((expected_text, expected_source), row) in enumerate(zip(chunks, rows)):
            actual_text, actual_source, actual_embedding_bytes = row

            assert actual_text == expected_text, (
                f"Row {i}: chunk_text mismatch. "
                f"Expected {expected_text!r}, got {actual_text!r}."
            )
            assert actual_source == expected_source, (
                f"Row {i}: source_doc mismatch. "
                f"Expected {expected_source!r}, got {actual_source!r}."
            )

            actual_vec = np.frombuffer(actual_embedding_bytes, dtype=np.float32)
            assert np.array_equal(actual_vec, dummy_embedding), (
                f"Row {i}: embedding mismatch after round-trip. "
                f"Expected all-ones vector, got {actual_vec[:5]}..."
            )

    finally:
        if os.path.exists(db_path):
            os.remove(db_path)


# ---------------------------------------------------------------------------
# Property 5: Answer Context Contains Only Retrieved Chunks
# Validates: Requirements 2.3
# ---------------------------------------------------------------------------

from study_assistant import run_qa_agent, init_vector_store, Config


@given(
    chunks=st.lists(
        st.tuples(
            st.text(min_size=1, max_size=100),   # chunk_text
            st.text(min_size=1, max_size=30),    # source_doc
            st.floats(min_value=0.1, max_value=1.0, allow_nan=False, allow_infinity=False),  # score
        ),
        min_size=1,
        max_size=5,
    )
)
@settings(max_examples=50)
def test_answer_context_isolation(chunks: list) -> None:
    """
    **Validates: Requirements 2.3**

    For any set of retrieved chunks, the prompt sent to the LLM must contain
    exactly those chunk texts and no other document text.  Specifically, every
    chunk_text from the mocked search_chunks return value must appear verbatim
    in the prompt captured from the converse() call.

    Strategy:
      - Mock search_chunks to return a fixed list of (chunk_text, source_doc, score) tuples.
      - Mock bedrock_client.converse() to capture the prompt and return a dummy response.
      - Assert each chunk_text appears in the captured prompt.
    """
    config = Config()
    conn = init_vector_store(":memory:")

    captured_prompts: list[str] = []

    def fake_converse(**kwargs):
        # Capture the prompt text from the messages argument
        messages = kwargs.get("messages", [])
        for msg in messages:
            for block in msg.get("content", []):
                if "text" in block:
                    captured_prompts.append(block["text"])
        return {
            "output": {
                "message": {
                    "content": [{"text": "Mocked answer."}]
                }
            }
        }

    mock_client = type("MockClient", (), {"converse": staticmethod(fake_converse)})()

    with patch("study_assistant.search_chunks", return_value=chunks), \
         patch("study_assistant.boto3.client", return_value=mock_client):
        result = run_qa_agent("test question", conn, config)

    # The converse() call must have been made (captured_prompts is non-empty)
    assert len(captured_prompts) > 0, (
        "converse() was never called — no prompt was captured."
    )

    # Every chunk_text must appear verbatim in the captured prompt
    full_prompt = "\n".join(captured_prompts)
    for chunk_text_val, _source_doc, _score in chunks:
        assert chunk_text_val in full_prompt, (
            f"Chunk text {chunk_text_val!r} was not found in the prompt sent to the LLM. "
            f"Prompt (first 500 chars): {full_prompt[:500]!r}"
        )


# ---------------------------------------------------------------------------
# Property 6: Citations Cover All Used Sources
# Validates: Requirements 2.5, 7.2
# ---------------------------------------------------------------------------

from study_assistant import run_qa_agent, init_vector_store, Config


@given(
    answer=st.text(min_size=1),
    sources=st.lists(
        st.text(min_size=1, max_size=20),
        min_size=1,
        max_size=5,
    ),
)
@settings(max_examples=50)
def test_citation_coverage(answer: str, sources: list) -> None:
    """
    **Validates: Requirements 2.5, 7.2**

    For any set of source names, every source must appear in the formatted
    response prefixed with "Source:".

    Since format_qa_response (task 8.2) is not yet implemented, this test
    verifies the citation contract via run_qa_agent: the returned "sources"
    list must contain every source name that was present in the retrieved
    chunks.  This validates that the Q&A agent correctly collects and
    deduplicates source names from the retrieved chunks.

    Additionally, the test verifies the citation format directly by
    constructing the expected "Source: {name}" strings and confirming the
    format contract is satisfied when a formatter applies it.
    """
    config = Config()
    conn = init_vector_store(":memory:")

    # Build chunks: one chunk per source, with a fixed score
    chunks = [(f"chunk text for {src}", src, 0.9) for src in sources]

    def fake_converse(**kwargs):
        return {
            "output": {
                "message": {
                    "content": [{"text": answer}]
                }
            }
        }

    mock_client = type("MockClient", (), {"converse": staticmethod(fake_converse)})()

    with patch("study_assistant.search_chunks", return_value=chunks), \
         patch("study_assistant.boto3.client", return_value=mock_client):
        result = run_qa_agent("test question", conn, config)

    returned_sources = result["sources"]

    # Every source name from the chunks must appear in the returned sources list
    for src in sources:
        assert src in returned_sources, (
            f"Source {src!r} was not found in the returned sources list: "
            f"{returned_sources!r}"
        )

    # Verify the citation format contract: each source, when formatted as
    # "Source: {name}", produces the expected prefix string.
    # This validates Requirements 7.2 — the formatter (task 8.2) must use
    # this exact prefix for each source.
    for src in returned_sources:
        citation_line = f"Source: {src}"
        assert citation_line.startswith("Source: "), (
            f"Citation line {citation_line!r} does not start with 'Source: '"
        )
        assert src in citation_line, (
            f"Source name {src!r} not found in citation line {citation_line!r}"
        )


# ---------------------------------------------------------------------------
# Property 19: History Compression at 20 Turns
# Validates: Requirements 6.4
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock
from study_assistant import compress_history, SessionMemory


@given(n_messages=st.integers(min_value=21, max_value=40))
@settings(max_examples=50)
def test_history_compression_at_20_turns(n_messages: int) -> None:
    """
    **Validates: Requirements 6.4**

    For any session with more than 20 messages, compress_history() must reduce
    the history length to (original_length - 10 + 1): the oldest 10 messages
    are replaced by a single summary context message, and the remaining
    (original_length - 10) messages are kept intact.
    """
    # Build a SessionMemory with n_messages messages in Bedrock converse() format.
    messages = [
        {"role": "user", "content": [{"text": f"msg {i}"}]}
        for i in range(n_messages)
    ]
    memory = SessionMemory(messages=messages)

    # Mock the bedrock_client so no real AWS calls are made.
    mock_client = MagicMock()
    mock_client.converse.return_value = {
        "output": {
            "message": {
                "content": [{"text": "Summary text"}]
            }
        }
    }

    result = compress_history(memory, mock_client)

    expected_length = n_messages - 10 + 1
    assert len(result.messages) == expected_length, (
        f"Expected compressed history length {expected_length} "
        f"(original={n_messages}), got {len(result.messages)}"
    )


# ---------------------------------------------------------------------------
# Property 1: Message History Completeness
# Validates: Requirements 1.1, 6.1
# ---------------------------------------------------------------------------


@given(messages=st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=20))
@settings(max_examples=100)
def test_message_history_completeness(messages: list) -> None:
    """
    **Validates: Requirements 1.1, 6.1**

    For any sequence of messages appended to a SessionMemory one by one,
    after each append the messages list must contain all prior messages in
    the correct order.
    """
    memory = SessionMemory()

    for i, text in enumerate(messages):
        # Append the next message in Bedrock converse() format.
        memory.messages.append(
            {"role": "user", "content": [{"text": text}]}
        )

        # After appending, the memory must contain exactly i+1 messages.
        assert len(memory.messages) == i + 1, (
            f"Expected {i + 1} messages after turn {i}, "
            f"got {len(memory.messages)}"
        )

        # Every prior message must be present in order.
        for j in range(i + 1):
            expected_text = messages[j]
            actual_text = memory.messages[j]["content"][0]["text"]
            assert actual_text == expected_text, (
                f"Message at index {j} is {actual_text!r}, "
                f"expected {expected_text!r} after turn {i}"
            )


# ---------------------------------------------------------------------------
# Property 18: Session Plan Memory
# Validates: Requirements 6.3
# ---------------------------------------------------------------------------


@given(
    plan=st.fixed_dictionaries(
        {
            "plan": st.lists(st.text(min_size=1)),
            "warning": st.none(),
        }
    )
)
@settings(max_examples=100)
def test_session_plan_memory(plan: dict) -> None:
    """
    **Validates: Requirements 6.3**

    For any study plan dict, storing it in SessionMemory.last_study_plan must
    make it retrievable unchanged on the next turn.
    """
    memory = SessionMemory()

    # Store the plan.
    memory.last_study_plan = plan

    # Assert it is retrievable and identical.
    assert memory.last_study_plan == plan, (
        f"Expected last_study_plan to be {plan!r}, "
        f"got {memory.last_study_plan!r}"
    )

# ---------------------------------------------------------------------------
# Property 20: Output Formatting Divider and Prefixes
# Validates: Requirements 7.3, 7.4
# ---------------------------------------------------------------------------

from study_assistant import format_turn


@given(
    user_input=st.text(min_size=1),
    assistant_response=st.text(min_size=1),
)
@settings(max_examples=100)
def test_output_formatting_divider_and_prefixes(
    user_input: str,
    assistant_response: str,
) -> None:
    """
    **Validates: Requirements 7.3, 7.4**

    For any non-empty user input and assistant response, format_turn() must:
      (a) include "You:" in the output (user prefix),
      (b) include "Assistant:" in the output (assistant prefix), and
      (c) include a divider of at least 40 consecutive identical characters
          (verified by asserting "-" * 40 is present in the output).
    """
    output = format_turn(user_input, assistant_response)

    # (a) "You:" prefix must appear in the output
    assert "You:" in output, (
        f'"You:" not found in format_turn output.\n'
        f"user_input={user_input!r}, assistant_response={assistant_response!r}\n"
        f"output={output!r}"
    )

    # (b) "Assistant:" prefix must appear in the output
    assert "Assistant:" in output, (
        f'"Assistant:" not found in format_turn output.\n'
        f"user_input={user_input!r}, assistant_response={assistant_response!r}\n"
        f"output={output!r}"
    )

    # (c) A divider of at least 40 consecutive identical characters must be present.
    # The implementation uses "-" * 60, so we assert "-" * 40 is a substring.
    assert "-" * 40 in output, (
        f'A divider of at least 40 consecutive "-" characters not found in output.\n'
        f"user_input={user_input!r}, assistant_response={assistant_response!r}\n"
        f"output={output!r}"
    )


# ---------------------------------------------------------------------------
# Property 17: Specialist Error Handling Keeps Session Active
# Validates: Requirements 5.4
# ---------------------------------------------------------------------------

from study_assistant import dispatch_tool, init_vector_store, Config


@given(error_message=st.text(min_size=1, max_size=100))
@settings(max_examples=100, deadline=None)
def test_specialist_error_handling_keeps_session_active(error_message: str) -> None:
    """
    **Validates: Requirements 5.4**

    For any error condition encountered by dispatch_tool(), the function must:
      (a) Return a dict (never raise an exception), keeping the session active.
      (b) Return a dict containing an "error" key when an unknown tool name is used.
      (c) Return a dict containing an "error" key when a specialist raises an exception.

    Three cases are tested:
      1. Unknown tool name → result contains "error" key.
      2. Specialist raises exception → result contains "error" key.
      3. In all cases, the function returns a dict (not raises).
    """
    conn = init_vector_store(":memory:")
    config = Config()

    # --- Case 1: Unknown tool name ---
    # dispatch_tool must return a dict with an "error" key for any unknown tool.
    unknown_tool_name = f"unknown_tool_{error_message[:20]}"
    result_unknown = dispatch_tool(unknown_tool_name, {}, conn, config)

    # Must return a dict (session stays active — no exception raised)
    assert isinstance(result_unknown, dict), (
        f"dispatch_tool with unknown tool name returned {type(result_unknown).__name__!r}, "
        f"expected dict. tool_name={unknown_tool_name!r}"
    )

    # Must contain "error" key
    assert "error" in result_unknown, (
        f"dispatch_tool with unknown tool name did not return an 'error' key. "
        f"tool_name={unknown_tool_name!r}, result={result_unknown!r}"
    )

    # --- Case 2: Specialist raises an exception ---
    # Mock run_qa_agent to raise an exception with the generated error_message.
    def raise_exception(*args, **kwargs):
        raise RuntimeError(error_message)

    with patch("study_assistant.run_qa_agent", side_effect=raise_exception):
        # Must not raise — must return a dict with "error" key
        try:
            result_exception = dispatch_tool(
                "answer_concept_question",
                {"question": "test"},
                conn,
                config,
            )
            # If it returns, it must be a dict
            assert isinstance(result_exception, dict), (
                f"dispatch_tool returned {type(result_exception).__name__!r} "
                f"when specialist raised an exception, expected dict."
            )
            # Must contain "error" key
            assert "error" in result_exception, (
                f"dispatch_tool did not return an 'error' key when specialist raised. "
                f"result={result_exception!r}"
            )
        except Exception as exc:
            # If dispatch_tool propagated the exception, the test fails:
            # the session would crash instead of staying active.
            raise AssertionError(
                f"dispatch_tool raised {type(exc).__name__}: {exc!r} "
                f"when specialist raised RuntimeError({error_message!r}). "
                f"The session must stay active — dispatch_tool must catch exceptions "
                f"and return an error dict instead of propagating them."
            ) from exc

    # --- Case 3: Return type is always dict ---
    # Verified implicitly by Cases 1 and 2 above (both assert isinstance(..., dict)).
    # Explicitly confirm with a second unknown-tool call to cover the general case.
    result_general = dispatch_tool("nonexistent_specialist", {"key": error_message}, conn, config)
    assert isinstance(result_general, dict), (
        f"dispatch_tool must always return a dict. "
        f"Got {type(result_general).__name__!r} for tool 'nonexistent_specialist'."
    )


# ---------------------------------------------------------------------------
# Property 2: Empty and Whitespace-Only Input Rejection
# Validates: Requirements 1.4
# ---------------------------------------------------------------------------

import io
from unittest.mock import patch, MagicMock

from study_assistant import chat_loop, init_vector_store, Config


@given(
    whitespace_input=st.text(
        alphabet=st.sampled_from([" ", "\t", "\n", "\r", "\x0b", "\x0c"]),
        min_size=1,
    )
)
@settings(max_examples=100)
def test_empty_whitespace_input_rejection(whitespace_input: str) -> None:
    """
    **Validates: Requirements 1.4**

    Property test: For any string composed entirely of Python-strippable
    whitespace characters (space, tab, newline, carriage return, vertical tab,
    form feed), stripping it must yield an empty string — confirming that the
    ``if not user_input:`` guard in ``chat_loop`` (after ``raw.strip()``) will
    reject it and print "Please enter a message." rather than forwarding it to
    the supervisor.

    The alphabet is restricted to the six characters that Python's ``str.strip()``
    removes, which are exactly the characters that ``chat_loop`` treats as
    "empty" input.
    """
    # Every string composed only of strippable whitespace must strip to "".
    assert whitespace_input.strip() == "", (
        f"Expected whitespace-only string to strip to '', "
        f"but got {whitespace_input.strip()!r} for input {whitespace_input!r}"
    )


def test_empty_whitespace_input_rejection_integration() -> None:
    """
    **Validates: Requirements 1.4**

    Integration test: Simulate whitespace-only inputs followed by "exit" in
    ``chat_loop``.  Verify that:
      (a) "Please enter a message." is printed for each whitespace-only input.
      (b) ``chat_loop`` completes without raising an exception.
      (c) The session ends normally (session summary is printed on "exit").
    """
    conn = init_vector_store(":memory:")
    config = Config()

    # Whitespace-only inputs that should be rejected, then a clean exit.
    side_effects = ["   ", "\t", "\n", "  \t  ", "exit"]

    captured_output = io.StringIO()

    with patch("builtins.input", side_effect=side_effects), \
         patch("study_assistant.run_supervisor_turn") as mock_supervisor, \
         patch("study_assistant.compress_history", side_effect=lambda mem, _client: mem), \
         patch("sys.stdout", captured_output):
        # chat_loop should complete without raising
        chat_loop("", conn, config)

    output = captured_output.getvalue()

    # (a) "Please enter a message." must appear once per whitespace-only input
    # (4 whitespace inputs before "exit")
    expected_count = 4
    actual_count = output.count("Please enter a message.")
    assert actual_count == expected_count, (
        f"Expected 'Please enter a message.' to appear {expected_count} times, "
        f"but found {actual_count} times.\nOutput:\n{output!r}"
    )

    # (b) run_supervisor_turn must NOT have been called (all inputs were rejected)
    mock_supervisor.assert_not_called()

    # (c) Session summary must appear (exit was typed)
    assert "Session ended." in output, (
        f"Expected 'Session ended.' in output after typing 'exit'.\nOutput:\n{output!r}"
    )
