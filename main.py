from time import perf_counter

from pydantic import ValidationError

from app.config import Settings
from app.llm.client import LLMClient, LLMResult
from app.llm.errors import LLMClientError

SYSTEM_INSTRUCTION = (
    "Ты помогаешь оператору службы поддержки. "
    "Кратко пересказывай обращение одним предложением. "
    "Используй только факты из обращения. "
    "Не придумывай суммы, даты, причины и действия. "
    "Если данных недостаточно, прямо сообщай об этом."
)


def build_messages(user_text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": SYSTEM_INSTRUCTION,
        },
        {
            "role": "user",
            "content": f"Обращение:\n{user_text}",
        },
    ]


def print_usage(result: LLMResult) -> None:
    if result.total_tokens is None:
        print("Провайдер не вернул статистику токенов.")
        return

    print(f"Входные токены: {result.prompt_tokens}")
    print(f"Выходные токены: {result.completion_tokens}")
    print(f"Всего токенов: {result.total_tokens}")


def summarize_request(client: LLMClient, user_text: str, settings: Settings) -> None:
    """Кратко пересказывает обращение и печатает метрики запроса."""
    text = user_text.strip()
    if not text:
        print("Ошибка: обращение не должно быть пустым.")
        return

    started_at = perf_counter()

    try:
        result = client.generate(build_messages(text))
    except LLMClientError as error:
        print(f"Не удалось получить ответ модели: {error}")
        return

    elapsed_seconds = perf_counter() - started_at

    answer = result.text

    print("\nРезультат:")
    print(answer or "Модель не вернула текстовый ответ")

    print("\nМетрики:")
    print(f"Модель: {result.model}")
    print(f"Завершение: {result.finish_reason}")
    print(f"Время: {elapsed_seconds:.2f} с")

    print_usage(result)

    print(f"ID ответа: {result.response_id}")


def main() -> None:
    try:
        settings = Settings()
    except ValidationError as error:
        print("Ошибка конфигурации:")
        for issue in error.errors():
            field = ".".join(str(part) for part in issue["loc"])
            print(f"- {field}: {issue['msg']}")
        return

    client = LLMClient(settings)

    print(f"Окружение: {settings.app_env}")
    print(f"Модель: {settings.model}")

    user_text = input("Введите текст обращения: ")
    summarize_request(client, user_text, settings)


if __name__ == "__main__":
    main()
