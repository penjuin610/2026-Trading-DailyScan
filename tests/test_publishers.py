import os
import tempfile
import unittest
from unittest.mock import patch

from src.reporting.publishers import publish_reports


class PublisherTests(unittest.TestCase):
    def test_publish_reports_keeps_email_running_when_discord_fails(self) -> None:
        os.environ["DISCORD_WEBHOOK_URL"] = "https://discord.test/webhook"
        os.environ["SMTP_HOST"] = "smtp.test"
        os.environ["SMTP_PORT"] = "587"
        os.environ["SMTP_USERNAME"] = "user"
        os.environ["SMTP_PASSWORD"] = "pass"
        os.environ["SMTP_FROM"] = "from@test"
        os.environ["SMTP_TO"] = "to@test"

        with tempfile.TemporaryDirectory() as tmpdir:
            markdown_path = f"{tmpdir}/report.md"
            html_path = f"{tmpdir}/report.html"
            with open(markdown_path, "w", encoding="utf-8") as handle:
                handle.write("# report")
            with open(html_path, "w", encoding="utf-8") as handle:
                handle.write("<html></html>")

            with patch("src.reporting.publishers._post_discord", side_effect=RuntimeError("discord down")), patch("src.reporting.publishers._send_email") as send_email:
                result = publish_reports(
                    {
                        "notifications": {
                            "discord": {"enabled": True, "webhook_env": "DISCORD_WEBHOOK_URL"},
                            "email": {
                                "enabled": True,
                                "smtp_host_env": "SMTP_HOST",
                                "smtp_port_env": "SMTP_PORT",
                                "username_env": "SMTP_USERNAME",
                                "password_env": "SMTP_PASSWORD",
                                "from_env": "SMTP_FROM",
                                "to_env": "SMTP_TO",
                            },
                        }
                    },
                    markdown_path,
                    html_path,
                    "summary",
                )

        self.assertFalse(result["discord"]["ok"])
        self.assertTrue(result["email"]["ok"])
        self.assertTrue(send_email.called)
