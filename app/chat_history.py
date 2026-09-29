from dataclasses import dataclass

import tiktoken

from app.llm.client import Message

TOKENS_PER_MESSAGE = 3
REPLY_PRIMER_TOKENS = 3


class ChatHistoryError(RuntimeError):
    pass


@dataclass(frozen=True)
class PreparedHistory:
    messages: list[Message]
    estimated_tokens: int
    context_history_messages: int
    removed_history_messages: int


class TokenCounter:
    def __init__(self, encoding_name: str = "o200k_base") -> None:
        self._encoding = tiktoken.get_encoding(encoding_name)

    def count_text(self, text: str) -> int:
        return len(self._encoding.encode(text))

    def count_messages(self, messages: list[Message]) -> int:
        total = REPLY_PRIMER_TOKENS

        for message in messages:
            total += TOKENS_PER_MESSAGE
            total += self.count_text(message["role"])
            total += self.count_text(message["content"])

        return total


class SlidingWindowHistory:
    def __init__(self, token_counter: TokenCounter, input_token_budget: int) -> None:
        if input_token_budget <= 0:
            raise ValueError("Бюджет входных токенов должен быть больше нуля.")

        self._token_counter = token_counter
        self._input_token_budget = input_token_budget

    def prepare(
        self,
        developer_message: Message,
        history: list[Message],
        user_message: Message,
    ) -> PreparedHistory:
        turns = self._split_into_turns(history)
        required_messages = [developer_message, user_message]
        required_tokens = self._token_counter.count_messages(required_messages)

        if required_tokens > self._input_token_budget:
            raise ChatHistoryError(
                "Developer-инструкция и новое сообщение "
                "не помещаются во входной бюджет."
            )

        selected_history: list[Message] = []

        for turn in reversed(turns):
            candidate_history = [*turn, *selected_history]
            candidate_messages = [developer_message, *candidate_history, user_message]
            candidate_tokens = self._token_counter.count_messages(candidate_messages)

            if candidate_tokens > self._input_token_budget:
                break

            selected_history = candidate_history

        messages = [developer_message, *selected_history, user_message]
        estimated_tokens = self._token_counter.count_messages(messages)

        return PreparedHistory(
            messages=messages,
            estimated_tokens=estimated_tokens,
            context_history_messages=len(selected_history),
            removed_history_messages=(len(history) - len(selected_history)),
        )

    @staticmethod
    def _split_into_turns(history: list[Message]) -> list[list[Message]]:
        if len(history) % 2 != 0:
            raise ChatHistoryError("История должна состоять из законченных ходов.")

        turns: list[list[Message]] = []

        for index in range(0, len(history), 2):
            turn = history[index : index + 2]
            roles = [message["role"] for message in turn]

            if roles != ["user", "assistant"]:
                raise ChatHistoryError("Ожидалась пара сообщений user/assistant.")

            turns.append(turn)

        return turns
