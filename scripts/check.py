#!/usr/bin/env python3
"""Verificação de qualidade para código Python: ruff autofix, ruff lint e mypy.

Dois modos de execução: sem argumentos verifica todo o repositório (CI e pre-commit);
com ``--hook`` lê o payload JSON via stdin e verifica apenas o arquivo editado.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"

LINT_TARGETS = ("src", "tests", "scripts")
TYPE_CHECK_TARGET = "src"
HOOK_FLAG = "--hook"

# Código de saída 2 sinaliza bloqueio do gate para o hook de validação.
BLOCKING_EXIT_CODE = 2


class CheckFailure(Exception):
    """Lançada quando a verificação de qualidade reprova o estado atual."""


def running_inside_project_venv() -> bool:
    """Informa se o interpretador atual é o do ambiente virtual do projeto.

    A detecção usa sys.prefix, nunca o caminho do executável: num venv criado com symlinks,
    `.venv/bin/python` resolve para o mesmo binário do sistema, então comparar executáveis
    conclui "já estou no venv" justamente quando não está. Só o interpretador do venv aponta
    o prefixo para o diretório do venv.
    """
    return Path(sys.prefix).resolve() == (REPO_ROOT / ".venv").resolve()


def reexec_in_project_venv() -> None:
    """Reinicia a execução dentro do venv do projeto caso tenha sido chamado externamente."""
    if running_inside_project_venv() or not VENV_PYTHON.exists():
        return
    # Interpretador e script vêm de caminhos derivados de __file__, não de entrada externa.
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), str(SCRIPT_PATH), *sys.argv[1:]])  # noqa: S606


def read_edited_path() -> str | None:
    """Extrai o caminho do arquivo editado a partir do payload JSON recebido via stdin."""
    raw = sys.stdin.read().strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None

    tool_input = payload.get("tool_input") if isinstance(payload, dict) else None
    file_path = tool_input.get("file_path") if isinstance(tool_input, dict) else None
    return file_path if isinstance(file_path, str) else None


def resolve_edited_file() -> Path | None:
    """Retorna o caminho do arquivo caso seja um código Python monitorado pelo gate."""
    raw_path = read_edited_path()
    if raw_path is None:
        return None

    path = Path(raw_path).resolve()
    if path.suffix != ".py":
        return None
    return path if any(path.is_relative_to(REPO_ROOT / t) for t in LINT_TARGETS) else None


def run(command: list[str], failure_message: str) -> None:
    # O comando é montado a partir de constantes deste módulo; nada vem do payload do hook.
    result = subprocess.run(  # noqa: S603
        command, cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        report = "\n".join(part for part in (result.stdout, result.stderr) if part.strip())
        raise CheckFailure(f"{failure_message}\n\n{report}")


def check(paths: list[str]) -> None:
    run([sys.executable, "-m", "ruff", "check", "--fix", *paths], "ruff não conseguiu corrigir tudo")
    run([sys.executable, "-m", "ruff", "check", *paths], "ruff reprovou")
    run([sys.executable, "-m", "mypy", TYPE_CHECK_TARGET], "mypy reprovou")


def main() -> int:
    reexec_in_project_venv()

    if HOOK_FLAG in sys.argv[1:]:
        edited = resolve_edited_file()
        if edited is None:
            return 0
        targets = [str(edited.relative_to(REPO_ROOT))]
    else:
        targets = list(LINT_TARGETS)

    try:
        check(targets)
    except CheckFailure as failure:
        sys.stderr.write(f"{failure}\n")
        return BLOCKING_EXIT_CODE
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
