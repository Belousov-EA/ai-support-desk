from pydantic import ValidationError

from app.chat_history import (
    SlidingWindowHistory,
    TokenCounter,
)
from app.chat_service import ChatService
from app.config import Settings
from app.console import run_console
from app.llm.client import LLMClient


def main() -> None:
    try:
        settings = Settings()
    except ValidationError as error:
        print("Ошибка конфигурации:")
        for issue in error.errors():
            field = ".".join(str(part) for part in issue["loc"])
            print(f"- {field}: {issue['msg']}")
        return

    llm_client = LLMClient(settings)
    token_counter = TokenCounter()
    history_policy = SlidingWindowHistory(
        token_counter=token_counter,
        input_token_budget=settings.chat_input_token_budget,
    )
    chat_service = ChatService(
        llm_client=llm_client,
        history_policy=history_policy,
    )

    try:
        run_console(
            chat_service,
            app_env=settings.app_env,
            configured_model=settings.model,
        )
    finally:
        llm_client.close()


if __name__ == "__main__":
    main()
