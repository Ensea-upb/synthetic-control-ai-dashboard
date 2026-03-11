from __future__ import annotations

"""
Legacy compatibility layer.

Do not use this module as an authority for invalidation.
The canonical invalidation API lives in app_ui.state.invalidation.
"""

from app_ui.state.invalidation import (
    invalidate_from_data_change,
    invalidate_from_estimation_change,
    maybe_invalidate_on_data_config_change,
    maybe_invalidate_on_estimation_config_change,
)

from app_ui.state.initialization import initialize_app_state

def clear_estimation_outputs() -> None:
    invalidate_from_estimation_change()


def clear_sc_format_and_downstream() -> None:
    invalidate_from_data_change()


def clear_all_app_state() -> None:
    initialize_app_state(force=True)
