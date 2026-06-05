from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path

import requests


def _post_discord(webhook_url: str, summary_text: str, markdown_path: str) -> None:
    with open(markdown_path, "rb") as handle:
        response = requests.post(
            webhook_url,
            data={"content": summary_text},
            files={"file": (Path(markdown_path).name, handle, "text/markdown")},
            timeout=15,
        )
    response.raise_for_status()


def _send_email(email_cfg: dict, html_path: str, subject: str = "Daily Push Report") -> None:
    host = os.environ[email_cfg["smtp_host_env"]]
    port = int(os.environ[email_cfg["smtp_port_env"]])
    username = os.environ[email_cfg["username_env"]]
    password = os.environ[email_cfg["password_env"]]
    sender = os.environ[email_cfg["from_env"]]
    recipient = os.environ[email_cfg["to_env"]]

    html_body = Path(html_path).read_text(encoding="utf-8")
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = recipient
    message.set_content("HTML report attached in body view.")
    message.add_alternative(html_body, subtype="html")

    context = ssl.create_default_context()
    with smtplib.SMTP(host, port) as server:
        server.starttls(context=context)
        server.login(username, password)
        server.send_message(message)


def publish_reports(config: dict, markdown_path: str, html_path: str, summary_text: str) -> dict:
    result = {"discord": {"ok": False}, "email": {"ok": False}}
    notifications = config.get("notifications", {})
    discord_cfg = notifications.get("discord", {})
    email_cfg = notifications.get("email", {})

    if discord_cfg.get("enabled"):
        try:
            _post_discord(os.environ[discord_cfg["webhook_env"]], summary_text, markdown_path)
            result["discord"]["ok"] = True
        except Exception as exc:
            result["discord"]["error"] = str(exc)

    if email_cfg.get("enabled"):
        try:
            _send_email(email_cfg, html_path)
            result["email"]["ok"] = True
        except Exception as exc:
            result["email"]["error"] = str(exc)

    return result
