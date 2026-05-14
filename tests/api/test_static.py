import pytest
from fastapi.testclient import TestClient

from src.api.app import NoCacheStaticFiles, app, build_static_files
from src.core.config import WEB_DIR


@pytest.fixture
def client():
    """Cliente sem lifespan: servir arquivo estático não depende de modelo carregado."""
    return TestClient(app)

# ---------------------------------------------------------------------------
# Servir estáticos sem cache em desenvolvimento
# ---------------------------------------------------------------------------


class TestBuildStaticFiles:
    def test_development_serves_without_cache(self):
        """Em desenvolvimento o navegador guardava a versão antiga do JS mesmo depois de
        reiniciar o servidor, e o sintoma - código certo no disco, comportamento antigo na
        tela - custa caro justamente numa demonstração."""
        assert isinstance(build_static_files(WEB_DIR, "development"), NoCacheStaticFiles)
