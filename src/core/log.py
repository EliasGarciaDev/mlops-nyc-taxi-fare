import json
import logging
import os
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any, TextIO

DEFAULT_LOG_LEVEL: str = "INFO"
CORRELATION_FIELD: str = "correlation_id"

# Atributos que o próprio logging coloca em cada registro. Tudo que não estiver nesta lista
# veio de `extra=` e é conteúdo do evento, não da infraestrutura.
_RESERVED_RECORD_FIELDS: set[str] = {
    "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName",
    "levelname", "levelno", "lineno", "message", "module", "msecs", "msg", "name",
    "pathname", "process", "processName", "relativeCreated", "stack_info",
    "taskName", "thread", "threadName",
}

_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def set_correlation_id(value: str | None) -> None:
    """Bind an identifier that every log record in the current context will carry."""
    _correlation_id.set(value)


def get_correlation_id() -> str | None:
    return _correlation_id.get()


class JsonFormatter(logging.Formatter):
    """Render each record as one JSON object per line.

    Formato de linha única é o que ferramentas de agregação esperam, e é também o formato
    que a Fase 4 vai consumir para popular a tabela de predições - o mesmo evento serve
    para leitura humana e para persistência.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }

        correlation_id = get_correlation_id()
        if correlation_id is not None:
            payload[CORRELATION_FIELD] = correlation_id

        payload.update(
            {
                key: value
                for key, value in record.__dict__.items()
                if key not in _RESERVED_RECORD_FIELDS and not key.startswith("_")
            }
        )

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str | None = None, stream: TextIO | None = None) -> None:
    """Install the JSON handler on the root logger, once per process.

    Chamada na subida da aplicação e nos pontos de entrada de linha de comando. Se o handler
    JSON já estiver instalado, apenas o nível é ajustado: reinstalar duplicaria cada registro,
    e trocar o destino apagaria o de quem configurou primeiro - um teste, ou o processo que
    hospeda a aplicação.
    """
    root = logging.getLogger()
    root.setLevel(level or os.getenv("LOG_LEVEL") or DEFAULT_LOG_LEVEL)

    if any(isinstance(handler.formatter, JsonFormatter) for handler in root.handlers):
        return

    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(JsonFormatter())
    root.handlers = [handler]


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
