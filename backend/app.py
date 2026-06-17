import json
import logging
import sys
from datetime import datetime, timezone
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import os
import sys

# Ensure backend directory is in python path for module resolution
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from routes.alerts import router as alerts_router
from routes.detections import router as detections_router


class JsonFormatter(logging.Formatter):
    """
    Custom formatter that outputs logs as single-line JSON strings,
    enabling structured logging.
    """
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        # Extract extra fields passed to the logging call
        standard_attrs = {
            "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
            "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
            "created", "msecs", "relativeCreated", "thread", "threadName",
            "processName", "process", "message"
        }
        for key, value in record.__dict__.items():
            if key not in standard_attrs:
                log_record[key] = value

        return json.dumps(log_record)


def setup_logging() -> None:
    """
    Configures the root logger and FastAPI/Uvicorn loggers to use the structured JSON formatter.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    # Create console stream handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JsonFormatter())
    root_logger.addHandler(console_handler)

    # Route default Uvicorn loggers through our JSON formatter and prevent double-logging
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uv_logger = logging.getLogger(logger_name)
        uv_logger.handlers.clear()
        uv_logger.addHandler(console_handler)
        uv_logger.propagate = False


# Initialize structured logging
setup_logging()
logger = logging.getLogger("backend.app")

# Initialize FastAPI App
app = FastAPI(
    title="Edge AI Alert System Mock Backend",
    description="Temporary mock backend for integration testing Raspberry Pi abnormal behavior alerts.",
    version="1.0.0"
)

# Register endpoints
app.include_router(alerts_router)
app.include_router(detections_router)


@app.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Endpoint for verifying service status."
)
def health_check() -> dict:
    """
    GET /health
    """
    return {"status": "ok"}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """
    Handles Pydantic validation errors by structured logging the payload and returning 422.
    """
    errors = exc.errors()

    # Attempt to extract request body for logging
    try:
        body = await request.json()
    except Exception:
        try:
            body_bytes = await request.body()
            body = body_bytes.decode("utf-8") if body_bytes else None
        except Exception:
            body = "<unable to parse body>"

    logger.warning(
        "Request validation failed",
        extra={
            "errors": errors,
            "invalidBody": body
        }
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": errors,
            "message": "Validation failed for the requested payload"
        }
    )
