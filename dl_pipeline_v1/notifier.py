import json
from pathlib import Path

import requests


class TelegramNotifier:
    def __init__(self, config):
        self.enabled = bool(config.logging.telegram)
        self.credentials_path = Path(config.paths.telegram_credentials)
        self.token = None
        self.user = None

        if self.enabled:
            if not self.credentials_path.exists():
                raise FileNotFoundError(
                    f"Telegram logging is enabled but credentials were not found: {self.credentials_path}"
                )
            credentials = json.loads(self.credentials_path.read_text(encoding="utf-8"))
            self.token = credentials["bot"]
            self.user = credentials["user"]

    def send(self, message):
        if not self.enabled:
            return
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        requests.post(url, data={"chat_id": self.user, "text": str(message)}, timeout=10).raise_for_status()
