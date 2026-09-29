from uuid import uuid4

from app.chat_service import (
    ChatReply,
    ChatService,
    ChatServiceError,
)


def print_usage(result: ChatReply) -> None:
    if result.total_tokens is None:
        print("Провайдер не вернул статистику токенов.")
        return

    print(f"Входные токены: {result.prompt_tokens}")
    print(f"Выходные токены: {result.completion_tokens}")
    print(f"Всего токенов: {result.total_tokens}")


def run_console(
    service: ChatService,
    app_env: str,
    configured_model: str,
) -> None:
    print(f"Окружение: {app_env}")
    print(f"Настроенная модель: {configured_model}")

    session_id = uuid4().hex
    print(f"Session ID: {session_id}")
    print("Команды: /new — новый диалог, /exit — выход.")

    while True:
        user_text = input("\nВы: ")
        command = user_text.strip()

        if command == "/exit":
            return

        if command == "/new":
            service.reset_session(session_id)
            session_id = uuid4().hex
            print(f"Новый Session ID: {session_id}")
            continue

        try:
            result = service.reply(session_id, user_text)
        except ChatServiceError as error:
            print(f"Ошибка чата: {error}")
            continue

        print(f"Assistant: {result.text}")
        print(f"Session ID ответа: {result.session_id}")
        print(f"Сообщений в истории: {result.history_messages}")
        print(f"Промпт: {result.prompt_id}@{result.prompt_version}")
        print(f"Время ответа: {result.elapsed_seconds:.2f} с")
        print_usage(result)
        print(f"ID ответа: {result.response_id}")
