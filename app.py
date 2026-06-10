"""
AI Study Assistant — Streamlit UI
==================================
Run with:
    streamlit run Agents/study_assistant/app.py
"""

import os
import sys
import tempfile
import datetime
import re

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from study_assistant import (
    Config,
    SessionMemory,
    DayEntry,
    init_vector_store,
    ingest_document,
    list_documents,
    remove_document,
    run_supervisor_turn,
    run_calendar_agent,
    create_study_sessions,
    create_selected_study_sessions,
    compress_history,
    parse_study_plan_entries_from_text,
    parse_study_blocks_from_text,
    build_study_blocks_from_plan_entries,
    extract_study_blocks_with_llm,
    parse_natural_date_to_iso,
    parse_month_day_to_iso,
)
import boto3

# =============================================================================
# Page config
# =============================================================================
st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="🎓",
    layout="wide",
)

# =============================================================================
# Session state
# =============================================================================
if "config" not in st.session_state:
    st.session_state.config = Config()
# If a stale session cached a different model ID than what the code currently
# defaults to, replace it so the app always uses the configured default.
elif st.session_state.config.model_id != Config().model_id:
    st.session_state.config = Config()
if "conn" not in st.session_state:
    st.session_state.conn = init_vector_store(st.session_state.config.vector_store_path)
if "memory" not in st.session_state:
    st.session_state.memory = SessionMemory()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "bedrock_client" not in st.session_state:
    st.session_state.bedrock_client = boto3.client(
        "bedrock-runtime", region_name=st.session_state.config.region
    )
if "calendar_connected" not in st.session_state:
    st.session_state.calendar_connected = False
if "calendar_events" not in st.session_state:
    st.session_state.calendar_events = []
if "last_study_plan" not in st.session_state:
    st.session_state.last_study_plan = None
if "smart_plan_blocks" not in st.session_state:
    st.session_state.smart_plan_blocks = []
if "smart_plan_stats" not in st.session_state:
    st.session_state.smart_plan_stats = {}
if "smart_plan_version" not in st.session_state:
    st.session_state.smart_plan_version = 0
if "planner_prefill_source" not in st.session_state:
    st.session_state.planner_prefill_source = None
if "chat_import_notice" not in st.session_state:
    st.session_state.chat_import_notice = ""
if "last_study_plan_response_text" not in st.session_state:
    st.session_state.last_study_plan_response_text = ""
if "planner_preferred_start" not in st.session_state:
    st.session_state.planner_preferred_start = 9
if "planner_preferred_end" not in st.session_state:
    st.session_state.planner_preferred_end = 21
if "planner_max_session" not in st.session_state:
    st.session_state.planner_max_session = 2.0
if "planner_min_session" not in st.session_state:
    st.session_state.planner_min_session = 0.5


def _looks_like_placeholder_midnight_schedule(blocks: list[dict]) -> bool:
    """
    Detect parser-produced placeholder times for date-only study plans.
    """
    if not blocks:
        return False

    placeholder_end_times = {"00:30", "01:00", "01:30", "02:00"}
    return all(
        block.get("start_time") == "00:00"
        and block.get("end_time") in placeholder_end_times
        for block in blocks
    )


def _time_str_to_float(time_str: str) -> float:
    hour_str, minute_str = time_str.split(":")
    return int(hour_str) + int(minute_str) / 60.0


def _blocks_respect_preferences_and_calendar(
    blocks: list[dict],
    calendar_events: list[dict],
    preferred_start_hour: int,
    preferred_end_hour: int,
) -> bool:
    """
    Validate imported explicit-time blocks against preferred hours and busy calendar time.
    """
    for block in blocks:
        try:
            block_start = _time_str_to_float(block["start_time"])
            block_end = _time_str_to_float(block["end_time"])
        except Exception:
            return False

        if block_start < float(preferred_start_hour) or block_end > float(preferred_end_hour):
            return False

        block_date = block.get("date", "")
        for event in calendar_events:
            if event.get("date") != block_date:
                continue
            if event.get("all_day", False):
                return False
            try:
                event_start = _time_str_to_float(event.get("start_time", "00:00"))
                event_end = _time_str_to_float(event.get("end_time", "23:59"))
            except Exception:
                continue
            if block_start < event_end and block_end > event_start:
                return False

    return True


def _extract_day_entries_from_response(response_text: str) -> list[DayEntry]:
    """
    Parse study-plan entries from chat responses, including inline bullet formats.
    """
    entries = parse_study_plan_entries_from_text(response_text)
    if entries:
        return entries

    normalized_text = response_text.replace(" • ", "\n• ").replace("• ", "\n• ")
    inline_entries: list[DayEntry] = []

    natural_inline_matches = re.finditer(
        r"(?:^|[•\n])\s*(?:[A-Za-z]+,\s*)?([A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?)"
        r"(?:\s*\([^)]*\))?\s*:\s*([^()\n]+?)\s*\(([0-9.\-\s]+)\s*hours?\)",
        normalized_text,
        flags=re.IGNORECASE,
    )
    for match in natural_inline_matches:
        date_text, topic, hours_text = match.groups()
        date_str = parse_natural_date_to_iso(date_text, default_year=datetime.date.today().year)
        if not date_str:
            continue
        try:
            hours = float(hours_text.strip())
        except ValueError:
            continue
        topic = topic.strip().rstrip(".:")
        inline_entries.append(
            DayEntry(
                date=date_str,
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
        date_str = parse_month_day_to_iso(date_text, default_year=datetime.date.today().year)
        if not date_str:
            continue
        try:
            hours = float(hours_text.strip())
        except ValueError:
            continue
        topic = topic.strip().rstrip(".:")
        inline_entries.append(
            DayEntry(
                date=date_str,
                focus_topic=topic,
                hours=hours,
                outcome=f"Complete study of {topic}",
            )
        )

    return inline_entries


def _store_smart_plan_blocks(blocks: list[dict], stats: dict) -> None:
    """
    Save planner blocks and bump the widget version so stale labels/checks do not persist.
    """
    st.session_state.smart_plan_blocks = blocks
    st.session_state.smart_plan_stats = stats
    st.session_state.smart_plan_version += 1


def _import_chat_plan_into_planner(response_text: str) -> bool:
    """
    Import a chat study plan into the planner using real daytime slots when possible.
    """
    direct_blocks = extract_study_blocks_with_llm(
        response_text=response_text,
        bedrock_client=st.session_state.bedrock_client,
        model_id=st.session_state.config.model_id,
        reference_year=datetime.date.today().year,
    )
    if not direct_blocks:
        direct_blocks = parse_study_blocks_from_text(response_text)

    day_entries = _extract_day_entries_from_response(response_text)
    if direct_blocks and not _blocks_respect_preferences_and_calendar(
        direct_blocks,
        st.session_state.calendar_events if st.session_state.calendar_connected else [],
        st.session_state.planner_preferred_start,
        st.session_state.planner_preferred_end,
    ):
        direct_blocks = []

    if day_entries:
        blocks, stats = build_study_blocks_from_plan_entries(
            day_entries,
            st.session_state.calendar_events if st.session_state.calendar_connected else [],
            preferred_start_hour=st.session_state.planner_preferred_start,
            preferred_end_hour=st.session_state.planner_preferred_end,
            max_session_hours=st.session_state.planner_max_session,
            min_session_hours=st.session_state.planner_min_session,
        )
        if not blocks:
            return False
        _store_smart_plan_blocks(blocks, stats)
    elif direct_blocks:
        _store_smart_plan_blocks(
            direct_blocks,
            {
                "source": "chat_plan",
                "total_topics": len(direct_blocks),
                "total_sessions": len(direct_blocks),
                "total_study_hours": sum(b["duration_hours"] for b in direct_blocks),
                "unscheduled_topics": 0,
            },
        )
    else:
        return False

    st.session_state.planner_prefill_source = "chat"
    st.session_state.chat_import_notice = (
        f"Imported {len(st.session_state.smart_plan_blocks)} chat-planned study session(s) into the study planner tab."
    )
    return True


def _remap_last_study_plan_with_preferences() -> bool:
    """
    Rebuild planner sessions from the last chat study plan using current preferences.
    """
    if not st.session_state.last_study_plan:
        return False

    blocks, stats = build_study_blocks_from_plan_entries(
        st.session_state.last_study_plan,
        st.session_state.calendar_events if st.session_state.calendar_connected else [],
        preferred_start_hour=st.session_state.planner_preferred_start,
        preferred_end_hour=st.session_state.planner_preferred_end,
        max_session_hours=st.session_state.planner_max_session,
        min_session_hours=st.session_state.planner_min_session,
    )
    if not blocks:
        return False

    _store_smart_plan_blocks(blocks, stats)
    st.session_state.planner_prefill_source = "chat"
    return True

# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.title("🎓 AI Study Assistant")
    st.markdown("---")

    # --- File upload ---
    st.subheader("📂 Load Course Material")
    uploaded_file = st.file_uploader(
        "Upload notes, slides, or textbook (PDF or TXT)",
        type=["pdf", "txt"],
    )
    if uploaded_file is not None:
        if st.button("📥 Ingest Document", use_container_width=True):
            with st.spinner(f"Loading {uploaded_file.name}..."):
                suffix = os.path.splitext(uploaded_file.name)[1]
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                try:
                    result = ingest_document(tmp_path, st.session_state.conn, st.session_state.config)
                    orig_name = uploaded_file.name
                    tmp_name = os.path.basename(tmp_path)
                    st.session_state.conn.execute(
                        "UPDATE rag_chunks SET source_doc = ? WHERE source_doc = ?",
                        (orig_name, tmp_name),
                    )
                    st.session_state.conn.commit()
                    if result.startswith("Ingested"):
                        st.success(f"✅ {orig_name} loaded!")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"📄 I've loaded **{orig_name}** into my knowledge base. Ask me questions, request a quiz, or use the Smart Study Planner tab to schedule sessions.",
                        })
                    else:
                        st.warning(result)
                finally:
                    os.unlink(tmp_path)

    st.markdown("---")

    # --- Loaded documents ---
    st.subheader("📋 Loaded Documents")
    docs = list_documents(st.session_state.conn)
    if docs:
        for doc in docs:
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"📄 {doc}")
            if col2.button("🗑️", key=f"del_{doc}", help=f"Remove {doc}"):
                remove_document(doc, st.session_state.conn)
                st.rerun()
    else:
        st.info("No documents loaded yet.")

    st.markdown("---")

    # --- Google Calendar ---
    st.subheader("📅 Google Calendar")
    if not st.session_state.calendar_connected:
        st.info("Connect to schedule study sessions automatically.")
        if st.button("🔗 Connect Google Calendar", use_container_width=True):
            with st.spinner("Connecting..."):
                try:
                    from google_auth.auth import get_calendar_service
                    get_calendar_service()  # verify auth works
                    events = run_calendar_agent(days_ahead=30)
                    st.session_state.calendar_events = events
                    st.session_state.calendar_connected = True
                    st.success(f"Connected! {len(events)} events loaded.")
                    st.rerun()
                except Exception as e:
                    st.error(
                        f"Could not connect: {e}\n\n"
                        "Make sure `google_auth/credentials.json` is present."
                    )
    else:
        st.success(f"Calendar connected ({len(st.session_state.calendar_events)} events)")
        col1, col2 = st.columns(2)
        if col1.button("🔄 Refresh", use_container_width=True):
            with st.spinner("Refreshing..."):
                try:
                    st.session_state.calendar_events = run_calendar_agent(days_ahead=30)
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
        if col2.button("🔌 Disconnect", use_container_width=True):
            st.session_state.calendar_connected = False
            st.session_state.calendar_events = []
            st.rerun()

    st.markdown("---")

    # --- Settings ---
    st.subheader("⚙️ Settings")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.memory = SessionMemory()
        st.rerun()

    st.caption(f"Model: `{st.session_state.config.model_id}`")
    st.caption(f"Region: `{st.session_state.config.region}`")

# =============================================================================
# Main area — two tabs
# =============================================================================
tab_chat, tab_planner = st.tabs(["💬 Study Assistant", "📅 Smart Study Planner"])

# ---------------------------------------------------------------------------
# TAB 1: Interactive Study Assistant (chat)
# ---------------------------------------------------------------------------
with tab_chat:
    st.markdown(
        "Upload your course materials in the sidebar, then chat with your study assistant. "
        "Ask for quizzes, concept explanations, or a personalised study plan."
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question, request a quiz, or say 'load my notes'..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    prompt_lower = prompt.lower()
                    planner_handoff_request = (
                        ("smart study planner" in prompt_lower or "study planner" in prompt_lower)
                        and any(term in prompt_lower for term in ["send", "use", "move", "add", "put"])
                    )

                    if planner_handoff_request:
                        if st.session_state.smart_plan_blocks:
                            response_text = (
                                "Yes. Your chat study plan is already available in the Smart Study Planner tab. "
                                "Switch to that tab to review the imported sessions, edit labels, and add them to Google Calendar."
                            )
                            st.session_state.memory.messages = st.session_state.memory.messages + [
                                {"role": "user", "content": [{"text": prompt}]},
                                {"role": "assistant", "content": [{"text": response_text}]},
                            ]
                        elif st.session_state.last_study_plan_response_text and _import_chat_plan_into_planner(
                            st.session_state.last_study_plan_response_text
                        ):
                            response_text = (
                                "Yes. I sent the chat study plan to the Smart Study Planner tab. "
                                "You can switch there now to review the sessions and add them to Google Calendar."
                            )
                            st.session_state.memory.messages = st.session_state.memory.messages + [
                                {"role": "user", "content": [{"text": prompt}]},
                                {"role": "assistant", "content": [{"text": response_text}]},
                            ]
                        elif st.session_state.last_study_plan:
                            blocks, stats = build_study_blocks_from_plan_entries(
                                st.session_state.last_study_plan,
                                st.session_state.calendar_events,
                            )
                            _store_smart_plan_blocks(blocks, stats)
                            st.session_state.planner_prefill_source = "chat"
                            st.session_state.chat_import_notice = (
                                "Chat study plan imported into the Smart Study Planner tab."
                            )
                            response_text = (
                                "Yes. I sent the chat study plan to the Smart Study Planner tab. "
                                "You can switch there now to review the sessions and add them to Google Calendar."
                            )
                            st.session_state.memory.messages = st.session_state.memory.messages + [
                                {"role": "user", "content": [{"text": prompt}]},
                                {"role": "assistant", "content": [{"text": response_text}]},
                            ]
                        else:
                            response_text, st.session_state.memory.messages = run_supervisor_turn(
                                st.session_state.memory.messages,
                                prompt,
                                st.session_state.conn,
                                st.session_state.config,
                                calendar_events=st.session_state.calendar_events if st.session_state.calendar_connected else None,
                            )
                    else:
                        response_text, st.session_state.memory.messages = run_supervisor_turn(
                            st.session_state.memory.messages,
                            prompt,
                            st.session_state.conn,
                            st.session_state.config,
                            calendar_events=st.session_state.calendar_events if st.session_state.calendar_connected else None,
                        )

                    # Extract structured study sessions for the planner tab.
                    # Try the LLM extractor first (handles date+topic+duration format).
                    llm_blocks = extract_study_blocks_with_llm(
                        response_text=response_text,
                        bedrock_client=st.session_state.bedrock_client,
                        model_id=st.session_state.config.model_id,
                        reference_year=datetime.date.today().year,
                    )
                    # Only fall back to the regex timed-block parser if the LLM extractor
                    # found nothing at all. The regex parser requires explicit section headers
                    # and clock times, so it only adds value for that narrow format.
                    regex_blocks = [] if llm_blocks else parse_study_blocks_from_text(response_text)
                    direct_blocks = llm_blocks or regex_blocks

                    day_entries = _extract_day_entries_from_response(response_text)

                    # Only reject explicit-time regex blocks that conflict with preferences/
                    # calendar. LLM-derived blocks use sensible defaults (09:00) and will be
                    # re-slotted by build_study_blocks_from_plan_entries, so skip this check
                    # for them.
                    if regex_blocks and direct_blocks and not _blocks_respect_preferences_and_calendar(
                        direct_blocks,
                        st.session_state.calendar_events if st.session_state.calendar_connected else [],
                        st.session_state.planner_preferred_start,
                        st.session_state.planner_preferred_end,
                    ):
                        direct_blocks = []

                    # day_entries path: use build_study_blocks_from_plan_entries to slot
                    # topics into real calendar free time. This is the preferred path because
                    # it respects preferences and avoids calendar conflicts.
                    if day_entries:
                        st.session_state.last_study_plan = day_entries
                        st.session_state.memory.last_study_plan = {"response": response_text}
                        st.session_state.last_study_plan_response_text = response_text
                        _import_chat_plan_into_planner(response_text)
                    elif direct_blocks and not _looks_like_placeholder_midnight_schedule(direct_blocks):
                        # Fallback: use LLM/regex blocks directly when no DayEntry objects
                        # were parsed (e.g. the model didn't include hours in each line).
                        _store_smart_plan_blocks(
                            direct_blocks,
                            {
                                "source": "chat_plan",
                                "total_topics": len(direct_blocks),
                                "total_sessions": len(direct_blocks),
                                "total_study_hours": sum(b["duration_hours"] for b in direct_blocks),
                                "unscheduled_topics": 0,
                            },
                        )
                        st.session_state.planner_prefill_source = "chat"
                        st.session_state.chat_import_notice = (
                            f"Imported {len(direct_blocks)} chat-planned study session(s) into the Smart Study Planner tab."
                        )
                        st.session_state.last_study_plan_response_text = response_text

                    st.session_state.memory = compress_history(
                        st.session_state.memory,
                        st.session_state.bedrock_client,
                    )

                    st.markdown(response_text)
                    if st.session_state.chat_import_notice:
                        st.success(st.session_state.chat_import_notice)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})

                except Exception as e:
                    error_msg = f"⚠️ Error: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # Quick-add to calendar from chat-generated plan
    if st.session_state.last_study_plan and st.session_state.calendar_connected:
        with st.expander("📅 Add chat study plan to Google Calendar"):
            if st.button("Use This Plan in Smart Study Planner", key="chat_to_planner", use_container_width=True):
                blocks, stats = build_study_blocks_from_plan_entries(
                    st.session_state.last_study_plan,
                    st.session_state.calendar_events,
                    preferred_start_hour=st.session_state.planner_preferred_start,
                    preferred_end_hour=st.session_state.planner_preferred_end,
                    max_session_hours=st.session_state.planner_max_session,
                    min_session_hours=st.session_state.planner_min_session,
                )
                _store_smart_plan_blocks(blocks, stats)
                st.session_state.planner_prefill_source = "chat"
                if blocks:
                    st.success("Chat study plan sent to the Smart Study Planner tab.")
                else:
                    st.warning("I found a study plan in chat, but couldn't map it into free calendar slots.")

            col1, col2 = st.columns(2)
            start_hour = col1.slider("Start time (hour)", 6, 22, 9, key="chat_start")
            duration = col2.slider("Session length (hours)", 0.5, 4.0, 1.5, step=0.5, key="chat_dur")
            if st.button("Add to Google Calendar", key="chat_add_cal"):
                with st.spinner("Creating events..."):
                    results = create_study_sessions(
                        st.session_state.last_study_plan,
                        start_hour=start_hour,
                        session_duration_hours=duration,
                    )
                created = [r for r in results if r["status"] == "created"]
                failed = [r for r in results if r["status"] == "failed"]
                if created:
                    st.success(f"✅ Created {len(created)} study sessions!")
                if failed:
                    st.warning(f"⚠️ {len(failed)} sessions failed.")

# ---------------------------------------------------------------------------
# TAB 2: Smart Study Planner
# ---------------------------------------------------------------------------
with tab_planner:
    st.header("📅 Smart Study Planner")
    st.markdown(
        "Use the scheduling controls below to shape how your chat-generated study plan "
        "gets mapped into free time on your Google Calendar."
    )

    if not docs:
        st.warning("Upload course materials in the sidebar first.")
    elif not st.session_state.calendar_connected:
        st.warning("Connect your Google Calendar in the sidebar to use the smart planner.")
    else:
        if st.session_state.planner_prefill_source == "chat" and st.session_state.smart_plan_blocks:
            stats = st.session_state.get("smart_plan_stats", {})
            imported = stats.get("total_sessions", len(st.session_state.smart_plan_blocks))
            unscheduled = stats.get("unscheduled_topics", 0)
            st.info(
                f"Using a study plan imported from chat. {imported} session(s) were mapped into planner slots."
                + (f" {unscheduled} topic(s) could not be auto-scheduled." if unscheduled else "")
            )
        st.subheader("Scheduling Preferences")
        st.caption(
            "These settings control how study topics are mapped into open calendar time."
        )

        col1, col2 = st.columns(2)
        preferred_start = col1.slider(
            "Earliest study hour", 6, 14, 9,
            help="Earliest time to start a study session",
            key="planner_preferred_start",
        )
        preferred_end = col2.slider(
            "Latest study hour", 15, 23, 21,
            help="Latest time to end a study session",
            key="planner_preferred_end",
        )

        col3, col4 = st.columns(2)
        max_session = col3.slider("Max session length (hours)", 0.5, 4.0, 2.0, step=0.5, key="planner_max_session")
        min_session = col4.slider("Min session length (hours)", 0.25, 2.0, 0.5, step=0.25, key="planner_min_session")

        st.caption(
            "Adjust the rules, then refresh the proposed sessions before sending them to Google Calendar."
        )

        if st.session_state.last_study_plan:
            if st.button("Refresh Proposed Sessions", use_container_width=True):
                if _remap_last_study_plan_with_preferences():
                    st.success("Updated the proposed sessions using your current scheduling preferences.")
                else:
                    st.warning("I couldn't rebuild the study sessions from the last chat plan.")

        if st.session_state.smart_plan_blocks:
            st.markdown("---")

            # Show plan stats
            stats = st.session_state.get("smart_plan_stats", {})
            if stats:
                if stats.get("source") == "chat_plan":
                    st.info(
                        f"📝 **Source:** Chat-generated study plan  \n"
                        f"📅 **Mapped sessions:** {stats.get('total_sessions', 0)}  \n"
                        f"🕐 **Total scheduled time:** {stats.get('total_study_hours', 0):.1f} hours  \n"
                        f"📚 **Topics found in chat plan:** {stats.get('total_topics', 0)}  \n"
                        f"⚠️ **Unscheduled topics:** {stats.get('unscheduled_topics', 0)}"
                    )
                else:
                    complexity_colors = {"light": "🟢", "moderate": "🟡", "heavy": "🔴"}
                    icon = complexity_colors.get(stats.get("complexity", ""), "📚")
                    st.info(
                        f"{icon} **Material complexity:** {stats.get('complexity', 'unknown').capitalize()}  \n"
                        f"📖 **Research-recommended:** {stats.get('recommended_daily_hours', 0):.1f} hours/day  \n"
                        f"⏱️ **Your target:** {stats.get('actual_daily_target', 0):.1f} hours/day  \n"
                        f"📅 **Total sessions:** {stats.get('total_sessions', 0)}  \n"
                        f"🕐 **Total study time:** {stats.get('total_study_hours', 0):.1f} hours"
                    )

            st.subheader("Proposed Study Sessions")
            st.markdown(
                "Check/uncheck sessions, edit labels, then click "
                "**Add Selected to Google Calendar**."
            )

            # Column headers
            h1, h2, h3, h4, h5, h6 = st.columns([0.5, 1.5, 1.5, 3, 2, 0.8])
            h1.markdown("**✓**")
            h2.markdown("**Date**")
            h3.markdown("**Time**")
            h4.markdown("**Auto-generated topic**")
            h5.markdown("**Calendar label (editable)**")
            h6.markdown("**Hrs**")
            st.markdown("---")

            total_selected = 0
            updated_blocks = []
            block_version = st.session_state.smart_plan_version
            for i, block in enumerate(st.session_state.smart_plan_blocks):
                col_check, col_date, col_time, col_topic, col_label, col_dur = st.columns([0.5, 1.5, 1.5, 3, 2, 0.8])

                selected = col_check.checkbox("", value=block["selected"], key=f"block_{block_version}_{i}")
                col_date.markdown(f"**{block['date']}**")
                col_time.markdown(f"{block['start_time']} – {block['end_time']}")
                col_topic.markdown(f"<small>{block['topic']}</small>", unsafe_allow_html=True)

                # Editable label — default to the auto-generated topic, user can change it
                default_label = block.get("label", block["topic"])
                label = col_label.text_input(
                    "",
                    value=default_label,
                    key=f"label_{block_version}_{i}",
                    label_visibility="collapsed",
                    placeholder="Event title...",
                )
                col_dur.markdown(f"{block['duration_hours']:.1f}h")

                updated_blocks.append({**block, "selected": selected, "label": label})
                if selected:
                    total_selected += 1

            st.session_state.smart_plan_blocks = updated_blocks

            st.markdown(f"**{total_selected}** sessions selected "
                        f"({sum(b['duration_hours'] for b in updated_blocks if b['selected']):.1f} hours total)")

            if st.button(
                f"📅 Add {total_selected} Sessions to Google Calendar",
                use_container_width=True,
                type="primary",
                disabled=total_selected == 0,
            ):
                with st.spinner("Creating calendar events..."):
                    results = create_selected_study_sessions(
                        [b for b in updated_blocks if b["selected"]]
                    )
                created = [r for r in results if r["status"] == "created"]
                failed = [r for r in results if r["status"] == "failed"]
                if created:
                    st.success(f"✅ Created {len(created)} study sessions in your Google Calendar!")
                if failed:
                    st.warning(f"⚠️ {len(failed)} sessions could not be created.")
                    for f in failed:
                        st.caption(f"{f['date']}: {f['detail']}")
        else:
            st.info(
                "No chat study plan has been imported yet. Ask the chat assistant to create a study plan "
                "from your notes and schedule, and the suggested sessions will appear here."
            )
