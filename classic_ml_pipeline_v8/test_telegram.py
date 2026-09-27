from config import config
from notifier import send_telegram


if __name__ == "__main__":
    send_telegram(
        "Classic ML pipeline: Telegram connection works.",
        config,
        force=True,
        raise_on_error=True,
    )
    print("Telegram test message sent successfully.")
