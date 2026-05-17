from __future__ import annotations

from primary_reason.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "primary-reason" in result.stdout.lower()


def test_cli_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "primary-reason" in result.stdout
