import json
from pathlib import Path

import requests


def _load_telegram_credentials(config) -> tuple[str, str]:
    """Read Telegram bot credentials stored outside source control."""
    credentials_path = Path(config.paths.telegram_credentials)

    if not credentials_path.exists():
        raise FileNotFoundError(
            f"Telegram credentials not found: {credentials_path}. "
            "Copy telegram_credits.example.json to telegram_credits.json and fill it in."
        )

    with credentials_path.open("r", encoding="utf-8") as file:
        credentials = json.load(file)

    token = str(credentials["bot_token"]).strip()
    chat_id = str(credentials["chat_id"]).strip()

    if not token or token == "PUT_BOT_TOKEN_HERE":
        raise ValueError("Telegram bot_token is not configured")
    if not chat_id or chat_id == "PUT_CHAT_ID_HERE":
        raise ValueError("Telegram chat_id is not configured")

    return token, chat_id


def send_telegram(
    message: str,
    config,
    *,
    force: bool = False,
    raise_on_error: bool = False,
) -> bool:
    """Send a Telegram message without breaking training on notification failure."""
    if not force and not bool(config.logging.telegram):
        return False

    try:
        token, chat_id = _load_telegram_credentials(config)
        url = f"https://api.telegram.org/bot{token}/sendMessage"

        response = requests.post(
            url,
            data={"chat_id": chat_id, "text": message},
            timeout=10,
        )
        response.raise_for_status()
        return True
    except Exception as error:
        if raise_on_error:
            raise
        print(f"Telegram notification failed: {error}")
        return False
