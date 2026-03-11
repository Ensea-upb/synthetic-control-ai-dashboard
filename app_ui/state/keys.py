# app_ui/state/keys.py

"""
Single source of truth for Streamlit session_state keys.

All pages, controllers, and services must use these constants
instead of raw string literals.
"""

from __future__ import annotations


# -------------------------
# Core data state
# -------------------------
DF_RAW = "df_raw"
DATA_CONFIG = "data_config"
SC_FORMAT = "sc_format"

# Optional lightweight summaries
DATA_SUMMARY = "data_summary"
EXPLORATION_SUMMARY = "exploration_summary"

# -------------------------
# Estimation state
# -------------------------
ESTIMATION_CONFIG = "estimation_config"
ESTIMATION_RESULT = "estimation_result"
FIT_SUMMARY_DATA = "fit_summary_data"

# -------------------------
# Robustness state
# -------------------------
ROBUSTNESS_CONFIG = "robustness_config"
ROBUSTNESS_RESULTS = "robustness_results"

# -------------------------
# Plot caching / figure data
# -------------------------
FIGURE_CACHE = "figure_cache"

# -------------------------
# Workflow state
# -------------------------
WORKFLOW = "workflow"
CURRENT_PAGE = "current_page"
LAST_VALID_STEP = "last_valid_step"

# -------------------------
# Global app controls
# -------------------------
APP_INITIALIZED = "app_initialized"
APP_MESSAGES = "app_messages"

# -------------------------
# AI integration state
# -------------------------
AI_ENABLED = "ai_enabled"
AI_BACKEND = "ai_backend"              # e.g. "local", "api", "none"
AI_MODEL_STATUS = "ai_model_status"    # e.g. "ready", "loading", "error", "disabled"
AI_MESSAGES = "ai_messages"            # conversational history for UI
AI_CONTEXT_CACHE = "ai_context_cache"  # per-page context cache
AI_LAST_RESPONSE = "ai_last_response"
AI_LAST_ERROR = "ai_last_error"

# -------------------------
# Export state
# -------------------------
EXPORT_PAYLOAD = "export_payload"

DFS_BY_NAME = "dfs_by_name"
PREPARED_DFS_BY_NAME = "prepared_dfs_by_name"

EXPLORATION_CONFIGS = "exploration_configs"
EXPLORATION_RENDERED = "exploration_rendered"

ESTIMATION_INPUT_SNAPSHOT = "estimation_input_snapshot"

# -------------------------
# Stale-estimation flag
# Set to True when SCFormat changes but estimation has not been re-run.
# -------------------------
ESTIMATION_STALE = "estimation_stale"