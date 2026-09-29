from app.prompts.chat_history_summary.registry import (
    ACTIVE_CHAT_HISTORY_SUMMARY_VERSION,
    get_chat_history_summary_prompt,
)

CHAT_HISTORY_SUMMARY_PROMPT = get_chat_history_summary_prompt()


__all__ = [
    "ACTIVE_CHAT_HISTORY_SUMMARY_VERSION",
    "CHAT_HISTORY_SUMMARY_PROMPT",
    "get_chat_history_summary_prompt",
]
