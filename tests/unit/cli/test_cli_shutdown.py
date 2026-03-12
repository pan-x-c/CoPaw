# -*- coding: utf-8 -*-
from __future__ import annotations

from click.testing import CliRunner

from copaw.cli.main import cli


def test_stop_command_stops_backend_and_frontend(monkeypatch) -> None:
    monkeypatch.setattr(
        "copaw.cli.shutdown_cmd._listening_pids_for_port",
        lambda _port: {1001},
    )
    monkeypatch.setattr(
        "copaw.cli.shutdown_cmd._find_frontend_dev_pids",
        lambda: {2002},
    )
    monkeypatch.setattr(
        "copaw.cli.shutdown_cmd._find_desktop_wrapper_pids",
        set,
    )
    monkeypatch.setattr(
        "copaw.cli.shutdown_cmd._terminate_pid",
        lambda _pid: True,
    )

    result = CliRunner().invoke(cli, ["shutdown"])

    assert result.exit_code == 0
    assert "1001" in result.output
    assert "2002" in result.output


def test_stop_command_reports_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        "copaw.cli.shutdown_cmd._listening_pids_for_port",
        lambda _port: {1001},
    )
    monkeypatch.setattr(
        "copaw.cli.shutdown_cmd._find_frontend_dev_pids",
        set,
    )
    monkeypatch.setattr(
        "copaw.cli.shutdown_cmd._find_desktop_wrapper_pids",
        set,
    )
    monkeypatch.setattr(
        "copaw.cli.shutdown_cmd._terminate_pid",
        lambda _pid: False,
    )

    result = CliRunner().invoke(cli, ["shutdown"])

    assert result.exit_code != 0
    assert "Failed to shutdown process" in result.output


def test_stop_command_reports_nothing_found(monkeypatch) -> None:
    monkeypatch.setattr(
        "copaw.cli.shutdown_cmd._listening_pids_for_port",
        lambda _port: set(),
    )
    monkeypatch.setattr(
        "copaw.cli.shutdown_cmd._find_frontend_dev_pids",
        set,
    )
    monkeypatch.setattr(
        "copaw.cli.shutdown_cmd._find_desktop_wrapper_pids",
        set,
    )

    result = CliRunner().invoke(cli, ["shutdown"])

    assert result.exit_code != 0
    assert "No running CoPaw" in result.output
