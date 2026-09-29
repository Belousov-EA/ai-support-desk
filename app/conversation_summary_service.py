from app.llm.client import LLMClient, Message
from app.llm.errors import LLMClientError
from app.prompts.chat_history_summary import (
    CHAT_HISTORY_SUMMARY_PROMPT,
)

SUMMARY_INSTRUCTION = CHAT_HISTORY_SUMMARY_PROMPT.render()


class ConversationSummaryError(RuntimeError):
    pass


class ConversationSummaryService:
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    def summarize(
        self,
        previous_summary: str | None,
        messages: list[Message],
    ) -> str:
        if not messages:
            raise ConversationSummaryError("Нет сообщений для суммаризации.")

        request_messages: list[Message] = [
            {
                "role": "developer",
                "content": SUMMARY_INSTRUCTION,
            },
            {
                "role": "user",
                "content": self._build_source(
                    previous_summary=previous_summary,
                    messages=messages,
                ),
            },
        ]

        try:
            result = self._llm_client.generate(request_messages)
        except LLMClientError as error:
            raise ConversationSummaryError(
                f"Не удалось создать summary: {error}"
            ) from error

        if result.finish_reason == "length":
            raise ConversationSummaryError(
                "Summary остановлено из-за ограничения длины."
            )
        if result.finish_reason != "stop":
            raise ConversationSummaryError(
                "Модель не вернула готовое summary. "
                f"Причина завершения: {result.finish_reason}."
            )

        if result.text is None or not result.text.strip():
            raise ConversationSummaryError("Модель вернула пустое summary.")

        return result.text.strip()

    @classmethod
    def _build_source(
        cls,
        previous_summary: str | None,
        messages: list[Message],
    ) -> str:
        old_summary = previous_summary or "Отсутствует."
        transcript = cls._format_messages(messages)

        return (
            "<previous_summary>\n"
            f"{old_summary}\n"
            "</previous_summary>\n\n"
            "<conversation_fragment>\n"
            f"{transcript}\n"
            "</conversation_fragment>"
        )

    @staticmethod
    def _format_messages(messages: list[Message]) -> str:
        role_names = {
            "user": "Клиент",
            "assistant": "Поддержка",
        }
        lines: list[str] = []

        for message in messages:
            role = role_names.get(message["role"], message["role"])
            lines.append(f"{role}: {message['content']}")

        return "\n".join(lines)
