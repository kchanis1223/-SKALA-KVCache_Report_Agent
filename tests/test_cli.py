import sys

import pytest

from skala_agent import cli, runtime
from skala_agent.providers import DemoProvider


def run_cli(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["skala-agent", *argv])
    cli.main()


def test_default_mode_runs_without_api_keys(monkeypatch, tmp_path, capsys):
    output = tmp_path / "report.md"
    run_cli(monkeypatch, ["--output", str(output)])
    assert output.read_text(encoding="utf-8")
    assert "[demo]" in capsys.readouterr().out


def test_demo_mode_uses_demo_provider():
    assert isinstance(runtime.load_provider("demo"), DemoProvider)


def test_real_mode_fails_loudly_instead_of_falling_back_to_demo(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "ADAPTER_MODULE", "skala_agent._missing_adapter")
    output = tmp_path / "report.md"
    with pytest.raises(SystemExit) as caught:
        run_cli(monkeypatch, ["--mode", "real", "--output", str(output)])
    assert "real 모드 provider가 없습니다" in str(caught.value)
    assert not output.exists()


def test_real_mode_rejects_adapter_without_factory(monkeypatch):
    monkeypatch.setattr(runtime, "ADAPTER_MODULE", "skala_agent.providers")
    monkeypatch.setattr(runtime, "ADAPTER_FACTORY", "build_provider")
    with pytest.raises(runtime.ProviderUnavailableError, match="build_provider"):
        runtime.load_provider("real")


def test_real_mode_uses_adapter_factory(monkeypatch):
    sentinel = DemoProvider()
    monkeypatch.setattr(runtime, "ADAPTER_MODULE", "skala_agent.providers")
    monkeypatch.setattr(runtime, "ADAPTER_FACTORY", "DemoProvider")
    monkeypatch.setattr("skala_agent.providers.DemoProvider", lambda: sentinel)
    assert runtime.load_provider("real") is sentinel
