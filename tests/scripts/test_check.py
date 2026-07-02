import sys

from scripts import check


class FakeExistingPath:
    """Um caminho que se declara existente, para desacoplar o teste do ambiente."""

    def __init__(self, path):
        self._path = path

    def exists(self) -> bool:
        return True

    def __str__(self) -> str:
        return str(self._path)


# ---------------------------------------------------------------------------
# 1. Detecção de "estou rodando no venv do projeto"
# ---------------------------------------------------------------------------


class TestRunningInsideProjectVenv:
    def test_true_when_prefix_is_the_project_venv(self, monkeypatch):
        monkeypatch.setattr(sys, "prefix", str(check.REPO_ROOT / ".venv"))
        assert check.running_inside_project_venv() is True
