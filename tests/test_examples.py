"""
Example-based unit tests for the AI Study Assistant.

Tests cover:
  1. Exit/quit handling — session summary printed on "exit"/"quit", NOT on crash
  2. Default prompt — default prompt used when no CLI argument is provided
  3. Compound request routing — both Q&A and Planner dispatched for compound requests
  4. Debug flag — request payload printed when --debug is set
  5. Environment variable configuration — AWS_REGION, STUDY_ASSISTANT_MODEL_ID,
     VECTOR_STORE_PATH read correctly
  6. Model ID constant — supervisor uses anthropic.claude-3-5-haiku-20241022-v1:0
  7. Fallback message — exact fallback when no positive z-score chunks exist

Requirements: 1.3, 1.5, 2.4, 5.2, 5.5, 8.1, 8.2, 8.3, 8.4
"""

import io
import os
import sys
import sqlite3
import unittest
from unittest.mock import patch, MagicMock, call

# ---------------------------------------------------------------------------
# Make the study_assistant module importable from this test file regardless of
# how pytest is invoked (from the repo root or from within the package dir).
# ---------------------------------------------------------------------------
_AGENT_DIR = os.path.join(os.path.dirname(__file__), "..")
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)

import study_assistant  # noqa: E402  (import after sys.path manipulation)
from study_assistant import (
    Config,
    chat_loop,
    main,
    dispatch_tool,
    run_qa_agent,
    SUPERVISOR_MODEL_ID,
    init_vector_store,
)


# ===========================================================================
# 1. Exit / quit handling
# ===========================================================================

class TestExitQuitHandling(unittest.TestCase):
    """Verify session summary is printed on 'exit'/'quit' and NOT on crash."""

    def _make_conn(self):
        """Return an in-memory SQLite connection with the schema initialized."""
        return init_vector_store(":memory:")

    def test_exit_prints_session_ended(self):
        """'exit' input should print 'Session ended.' to stdout."""
        conn = self._make_conn()
        config = Config()

        captured = io.StringIO()
        with (
            patch("builtins.input", side_effect=["exit"]),
            patch("study_assistant.run_supervisor_turn") as mock_supervisor,
            patch("study_assistant.compress_history", side_effect=lambda m, _: m),
            patch("sys.stdout", captured),
            # Prevent boto3 client creation from hitting AWS
            patch("boto3.client", return_value=MagicMock()),
        ):
            mock_supervisor.return_value = ("Hello!", [])
            chat_loop("", conn, config)

        output = captured.getvalue()
        self.assertIn("Session ended.", output)

    def test_quit_prints_session_ended(self):
        """'quit' input should also print 'Session ended.' to stdout."""
        conn = self._make_conn()
        config = Config()

        captured = io.StringIO()
        with (
            patch("builtins.input", side_effect=["quit"]),
            patch("study_assistant.run_supervisor_turn") as mock_supervisor,
            patch("study_assistant.compress_history", side_effect=lambda m, _: m),
            patch("sys.stdout", captured),
            patch("boto3.client", return_value=MagicMock()),
        ):
            mock_supervisor.return_value = ("Hello!", [])
            chat_loop("", conn, config)

        output = captured.getvalue()
        self.assertIn("Session ended.", output)

    def test_crash_does_not_print_session_ended(self):
        """An unhandled exception should NOT print 'Session ended.'."""
        conn = self._make_conn()
        config = Config()

        captured = io.StringIO()
        with (
            patch("builtins.input", side_effect=["hello"]),
            patch(
                "study_assistant.run_supervisor_turn",
                side_effect=RuntimeError("boom"),
            ),
            patch("sys.stdout", captured),
            patch("boto3.client", return_value=MagicMock()),
        ):
            with self.assertRaises(RuntimeError):
                chat_loop("", conn, config)

        output = captured.getvalue()
        self.assertNotIn("Session ended.", output)


# ===========================================================================
# 2. Default prompt
# ===========================================================================

class TestDefaultPrompt(unittest.TestCase):
    """Verify the default prompt is used when no CLI argument is provided."""

    EXPECTED_DEFAULT = "I need help planning my studies. What can you help me with?"

    def test_default_prompt_used_when_no_arg(self):
        """main() with no positional arg should pass the default prompt to chat_loop."""
        captured_prompt = {}

        def fake_chat_loop(initial_prompt, conn, config):
            captured_prompt["value"] = initial_prompt

        with (
            patch("sys.argv", ["study_assistant.py"]),
            patch("study_assistant.chat_loop", side_effect=fake_chat_loop),
            patch("study_assistant.init_vector_store", return_value=MagicMock()),
        ):
            main()

        self.assertEqual(captured_prompt["value"], self.EXPECTED_DEFAULT)

    def test_explicit_prompt_overrides_default(self):
        """main() with a positional arg should pass that arg, not the default."""
        captured_prompt = {}

        def fake_chat_loop(initial_prompt, conn, config):
            captured_prompt["value"] = initial_prompt

        with (
            patch("sys.argv", ["study_assistant.py", "My custom prompt"]),
            patch("study_assistant.chat_loop", side_effect=fake_chat_loop),
            patch("study_assistant.init_vector_store", return_value=MagicMock()),
        ):
            main()

        self.assertEqual(captured_prompt["value"], "My custom prompt")
        self.assertNotEqual(captured_prompt["value"], self.EXPECTED_DEFAULT)


# ===========================================================================
# 3. Compound request routing
# ===========================================================================

class TestCompoundRequestRouting(unittest.TestCase):
    """Verify both Q&A and Planner are dispatched for compound requests."""

    def test_both_tools_dispatched(self):
        """dispatch_tool routes both answer_concept_question and generate_study_plan without error."""
        conn = init_vector_store(":memory:")
        config = Config()

        with (
            patch("study_assistant.run_qa_agent", return_value={"answer": "ok", "sources": []}),
            patch("study_assistant.run_calendar_agent", return_value=[]),
            patch("study_assistant.run_study_planner", return_value={"plan": [], "warning": None}),
        ):
            result_qa = study_assistant.dispatch_tool(
                "answer_concept_question",
                {"question": "What is AI?"},
                conn,
                config,
            )
            result_plan = study_assistant.dispatch_tool(
                "generate_study_plan",
                {"goal": "Pass exam", "topics": ["AI basics"], "hours_per_week": 10, "days": 5},
                conn,
                config,
            )

        # Both calls should succeed (no error key) confirming both tools are routed
        self.assertNotIn("error", result_qa)
        self.assertNotIn("error", result_plan)

    def test_dispatch_answer_concept_question_calls_qa_agent(self):
        """dispatch_tool('answer_concept_question', ...) should call run_qa_agent."""
        conn = init_vector_store(":memory:")
        config = Config()

        with patch("study_assistant.run_qa_agent") as mock_qa:
            mock_qa.return_value = {"answer": "test answer", "sources": []}
            result = dispatch_tool(
                "answer_concept_question",
                {"question": "What is machine learning?"},
                conn,
                config,
            )

        mock_qa.assert_called_once()
        self.assertEqual(result["answer"], "test answer")

    def test_dispatch_generate_study_plan_calls_planner(self):
        """dispatch_tool('generate_study_plan', ...) should call run_study_planner."""
        conn = init_vector_store(":memory:")
        config = Config()

        with (
            patch("study_assistant.run_calendar_agent", return_value=[]),
            patch("study_assistant.run_study_planner") as mock_planner,
        ):
            mock_planner.return_value = {"plan": [], "warning": None}
            result = dispatch_tool(
                "generate_study_plan",
                {
                    "goal": "Study for finals",
                    "topics": ["Topic A", "Topic B"],
                    "hours_per_week": 10,
                    "days": 5,
                },
                conn,
                config,
            )

        mock_planner.assert_called_once()
        self.assertIn("plan", result)


# ===========================================================================
# 4. Debug flag
# ===========================================================================

class TestDebugFlag(unittest.TestCase):
    """Verify config.debug == True when --debug is passed to main()."""

    def test_debug_flag_sets_config_debug_true(self):
        """main() with --debug should set config.debug = True before calling chat_loop."""
        captured_config = {}

        def fake_chat_loop(initial_prompt, conn, config):
            captured_config["debug"] = config.debug

        with (
            patch("sys.argv", ["study_assistant.py", "--debug"]),
            patch("study_assistant.chat_loop", side_effect=fake_chat_loop),
            patch("study_assistant.init_vector_store", return_value=MagicMock()),
        ):
            main()

        self.assertTrue(captured_config["debug"])

    def test_no_debug_flag_leaves_config_debug_false(self):
        """main() without --debug should leave config.debug = False."""
        captured_config = {}

        def fake_chat_loop(initial_prompt, conn, config):
            captured_config["debug"] = config.debug

        with (
            patch("sys.argv", ["study_assistant.py"]),
            patch("study_assistant.chat_loop", side_effect=fake_chat_loop),
            patch("study_assistant.init_vector_store", return_value=MagicMock()),
        ):
            main()

        self.assertFalse(captured_config["debug"])


# ===========================================================================
# 5. Environment variable configuration
# ===========================================================================

class TestEnvironmentVariableConfiguration(unittest.TestCase):
    """Verify AWS_REGION, STUDY_ASSISTANT_MODEL_ID, VECTOR_STORE_PATH are read correctly."""

    def test_env_vars_are_read_by_config(self):
        """Config() should pick up custom values from environment variables."""
        env_overrides = {
            "AWS_REGION": "eu-west-1",
            "STUDY_ASSISTANT_MODEL_ID": "anthropic.claude-3-haiku-20240307-v1:0",
            "VECTOR_STORE_PATH": "/tmp/custom_store.db",
        }

        with patch.dict(os.environ, env_overrides, clear=False):
            config = Config()

        self.assertEqual(config.region, "eu-west-1")
        self.assertEqual(config.model_id, "anthropic.claude-3-haiku-20240307-v1:0")
        self.assertEqual(config.vector_store_path, "/tmp/custom_store.db")

    def test_config_defaults_when_env_vars_absent(self):
        """Config() should use documented defaults when env vars are not set."""
        # Remove the env vars if they happen to be set in the test environment
        env_to_remove = {
            "AWS_REGION": None,
            "STUDY_ASSISTANT_MODEL_ID": None,
            "VECTOR_STORE_PATH": None,
        }

        # patch.dict with a value of None doesn't remove keys; use a custom approach
        original = {}
        for key in env_to_remove:
            original[key] = os.environ.pop(key, None)

        try:
            config = Config()
            self.assertEqual(config.region, "us-west-2")
            self.assertEqual(
                config.model_id, "anthropic.claude-3-5-haiku-20241022-v1:0"
            )
            self.assertEqual(config.vector_store_path, "study_assistant.db")
        finally:
            # Restore original env vars
            for key, val in original.items():
                if val is not None:
                    os.environ[key] = val


# ===========================================================================
# 6. Model ID constant
# ===========================================================================

class TestModelIDConstant(unittest.TestCase):
    """Verify supervisor uses anthropic.claude-3-5-haiku-20241022-v1:0."""

    EXPECTED_MODEL_ID = "anthropic.claude-3-5-haiku-20241022-v1:0"

    def test_supervisor_model_id_constant(self):
        """SUPERVISOR_MODEL_ID should equal the expected cross-region inference profile."""
        self.assertEqual(SUPERVISOR_MODEL_ID, self.EXPECTED_MODEL_ID)

    def test_config_default_model_id(self):
        """Config().model_id should equal the expected model ID when env var is not set."""
        original = os.environ.pop("STUDY_ASSISTANT_MODEL_ID", None)
        try:
            config = Config()
            self.assertEqual(config.model_id, self.EXPECTED_MODEL_ID)
        finally:
            if original is not None:
                os.environ["STUDY_ASSISTANT_MODEL_ID"] = original


# ===========================================================================
# 7. Fallback message
# ===========================================================================

class TestFallbackMessage(unittest.TestCase):
    """Verify exact fallback message when no positive z-score chunks exist."""

    EXPECTED_FALLBACK = "I don't have enough course material to answer that question."

    def test_fallback_when_search_returns_empty(self):
        """run_qa_agent should return the exact fallback when search_chunks returns []."""
        conn = init_vector_store(":memory:")
        config = Config()

        with patch("study_assistant.search_chunks", return_value=[]):
            result = run_qa_agent("What is deep learning?", conn, config)

        self.assertEqual(result["answer"], self.EXPECTED_FALLBACK)
        self.assertEqual(result["sources"], [])

    def test_fallback_sources_are_empty_list(self):
        """Fallback response should always have an empty sources list."""
        conn = init_vector_store(":memory:")
        config = Config()

        with patch("study_assistant.search_chunks", return_value=[]):
            result = run_qa_agent("Explain transformers", conn, config)

        self.assertIsInstance(result["sources"], list)
        self.assertEqual(len(result["sources"]), 0)


if __name__ == "__main__":
    unittest.main()
