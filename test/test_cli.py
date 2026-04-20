import sys

from nextrec import cli


def test_main_dispatches_studio_subcommand(monkeypatch):
    captured = {}

    def fake_run_studio_app(host=None, port=None):
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr(sys, "argv", ["nextrec", "studio", "--host", "0.0.0.0", "--port", "15173"])
    monkeypatch.setattr(cli, "run_studio_app", fake_run_studio_app)

    cli.main()

    assert captured == {"host": "0.0.0.0", "port": 15173}


def test_main_dispatches_docs_subcommand(monkeypatch):
    captured = {}

    def fake_run_docs_app(host=None, port=None):
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr(sys, "argv", ["nextrec", "docs", "--port", "4173"])
    monkeypatch.setattr(cli, "run_docs_app", fake_run_docs_app)

    cli.main()

    assert captured == {"host": None, "port": 4173}


def test_main_dispatches_studio_default_port(monkeypatch):
    captured = {}

    def fake_run_studio_app(host=None, port=None):
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr(sys, "argv", ["nextrec", "studio"])
    monkeypatch.setattr(cli, "run_studio_app", fake_run_studio_app)

    cli.main()

    assert captured == {"host": None, "port": None}


def test_run_studio_app_uses_default_port(monkeypatch):
    captured = {}

    def fake_run_frontend_app(app_name, script_name, host=None, port=None):
        captured["app_name"] = app_name
        captured["script_name"] = script_name
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr(cli, "run_frontend_app", fake_run_frontend_app)

    cli.run_studio_app()

    assert captured == {"app_name": "studio", "script_name": "dev", "host": None, "port": 15123}


def test_run_docs_app_uses_default_port(monkeypatch):
    captured = {}

    def fake_run_frontend_app(app_name, script_name, host=None, port=None):
        captured["app_name"] = app_name
        captured["script_name"] = script_name
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr(cli, "run_frontend_app", fake_run_frontend_app)

    cli.run_docs_app()

    assert captured == {"app_name": "docs", "script_name": "docs:dev", "host": None, "port": 15124}


def test_run_frontend_app_installs_dependencies_when_missing(monkeypatch, tmp_path):
    app_dir = tmp_path / "nextrec_studio"
    app_dir.mkdir()
    (app_dir / "package.json").write_text("{}", encoding="utf-8")

    commands = []

    def fake_run(command, cwd=None, check=None):
        commands.append({"command": command, "cwd": cwd, "check": check})

    monkeypatch.setattr(cli, "get_workspace_app_dir", lambda app_name: app_dir)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/npm")
    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    cli.run_frontend_app("studio", "dev", host="127.0.0.1", port=3000)

    assert commands == [
        {"command": ["/usr/bin/npm", "install"], "cwd": app_dir, "check": True},
        {
            "command": ["/usr/bin/npm", "run", "dev", "--", "--host", "127.0.0.1", "--port", "3000"],
            "cwd": app_dir,
            "check": True,
        },
    ]


def test_run_frontend_app_skips_install_when_dependencies_exist(monkeypatch, tmp_path):
    app_dir = tmp_path / "docs"
    app_dir.mkdir()
    (app_dir / "package.json").write_text("{}", encoding="utf-8")
    (app_dir / "node_modules").mkdir()

    commands = []

    def fake_run(command, cwd=None, check=None):
        commands.append({"command": command, "cwd": cwd, "check": check})

    monkeypatch.setattr(cli, "get_workspace_app_dir", lambda app_name: app_dir)
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/npm")
    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    cli.run_frontend_app("docs", "docs:dev")

    assert commands == [
        {"command": ["/usr/bin/npm", "run", "docs:dev"], "cwd": app_dir, "check": True},
    ]