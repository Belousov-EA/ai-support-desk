from dataclasses import dataclass, field
from threading import Lock
from time import perf_counter

from app.llm.client import LLMClient, Message
from app.llm.errors import LLMClientError
from app.prompts.support_chat import SUPPORT_CHAT_PROMPT

CHAT_INSTRUCTION = SUPPORT_CHAT_PROMPT.render()


class ChatServiceError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatReply:
    text: str
    session_id: str
    history_messages: int
    model: str
    prompt_id: str
    prompt_version: str
    elapsed_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    response_id: str


@dataclass
class ChatSession:
    messages: list[Message] = field(default_factory=list)
    lock: Lock = field(default_factory=Lock)


class ChatService:
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client
        self._sessions: dict[str, ChatSession] = {}
        self._sessions_lock = Lock()

    def reply(
        self,
        session_id: str,
        user_text: str,
    ) -> ChatReply:
        session_key = self._normalize_session_id(session_id)

        text = user_text.strip()
        if not text:
            raise ChatServiceError("Сообщение не должно быть пустым.")

        session = self._get_or_create_session(session_key)

        with session.lock:
            user_message = self._build_user_message(text)
            messages = self._build_messages(
                history=session.messages,
                user_message=user_message,
            )

            started_at = perf_counter()
            try:
                llm_result = self._llm_client.generate(messages)
            except LLMClientError as error:
                raise ChatServiceError(
                    f"Не удалось получить ответ модели: {error}"
                ) from error
            elapsed_seconds = perf_counter() - started_at

            if llm_result.finish_reason == "length":
                raise ChatServiceError("Ответ остановлен из-за ограничения длины.")
            if llm_result.finish_reason != "stop":
                raise ChatServiceError(
                    "Модель не вернула готовый ответ. "
                    f"Причина завершения: {llm_result.finish_reason}."
                )

            reply_text = self._validate_reply(llm_result.text)

            assistant_message: Message = {
                "role": "assistant",
                "content": reply_text,
            }
            session.messages.extend([user_message, assistant_message])
            history_messages = len(session.messages)

        return ChatReply(
            text=reply_text,
            session_id=session_key,
            history_messages=history_messages,
            model=llm_result.model,
            prompt_id=SUPPORT_CHAT_PROMPT.prompt_id,
            prompt_version=SUPPORT_CHAT_PROMPT.version,
            elapsed_seconds=elapsed_seconds,
            prompt_tokens=llm_result.prompt_tokens,
            completion_tokens=llm_result.completion_tokens,
            total_tokens=llm_result.total_tokens,
            response_id=llm_result.response_id,
        )

    def reset_session(self, session_id: str) -> None:
        session_key = self._normalize_session_id(session_id)

        with self._sessions_lock:
            session = self._sessions.get(session_key)

        if session is None:
            return

        with session.lock:
            session.messages.clear()

    def _get_or_create_session(
        self,
        session_id: str,
    ) -> ChatSession:
        with self._sessions_lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = ChatSession()
                self._sessions[session_id] = session
            return session

    @staticmethod
    def _normalize_session_id(session_id: str) -> str:
        session_key = session_id.strip()
        if not session_key:
            raise ChatServiceError("Session ID не должен быть пустым.")
        return session_key

    @staticmethod
    def _build_user_message(user_text: str) -> Message:
        return {
            "role": "user",
            "content": (f"<customer_message>\n{user_text}\n</customer_message>"),
        }

    @staticmethod
    def _build_messages(
        history: list[Message],
        user_message: Message,
    ) -> list[Message]:
        return [
            {
                "role": "developer",
                "content": CHAT_INSTRUCTION,
            },
            *history,
            user_message,
        ]

    @staticmethod
    def _validate_reply(reply: str | None) -> str:
        if reply is None:
            raise ChatServiceError("Модель не вернула текст ответа.")

        cleaned_reply = reply.strip()
        if not cleaned_reply:
            raise ChatServiceError("Модель вернула пустой ответ.")

        return cleaned_reply
