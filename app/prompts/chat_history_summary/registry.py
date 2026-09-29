from app.prompts.chat_history_summary.v1_0_0 import (
    CHAT_HISTORY_SUMMARY_PROMPT_V1_0_0,
)
from app.prompts.template import (
    PromptTemplate,
    PromptTemplateError,
)

ACTIVE_CHAT_HISTORY_SUMMARY_VERSION = "1.0.0"


_CHAT_HISTORY_SUMMARY_PROMPTS: dict[
    str,
    PromptTemplate,
] = {
    CHAT_HISTORY_SUMMARY_PROMPT_V1_0_0.version: (CHAT_HISTORY_SUMMARY_PROMPT_V1_0_0),
}


def get_chat_history_summary_prompt(
    version: str | None = None,
) -> PromptTemplate:
    selected_version = (
        ACTIVE_CHAT_HISTORY_SUMMARY_VERSION if version is None else version.strip()
    )

    try:
        return _CHAT_HISTORY_SUMMARY_PROMPTS[selected_version]
    except KeyError as error:
        available = ", ".join(_CHAT_HISTORY_SUMMARY_PROMPTS)
        raise PromptTemplateError(
            "Неизвестная версия chat_history_summary: "
            f"{selected_version}. Доступны: {available}."
        ) from error
