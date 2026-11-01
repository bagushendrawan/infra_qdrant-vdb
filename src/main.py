import logging
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.storage.config import Settings
from src.storage.exceptions import StorageError
from src.storage.router import router
from src.storage.worker import Worker

logger = logging.getLogger("uvicorn.error")


def create_app(settings=None):
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app):
        app.state.worker = Worker(settings)
        await app.state.worker.start()
        try:
            yield
        finally:
            await app.state.worker.close()

    app = FastAPI(
        title="Qdrant vector database",
        lifespan=lifespan,
        openapi_url="/openapi.json" if settings.environment in {"local", "staging"} else None,
    )
    from src.storage.middleware import BodyLimit

    app.add_middleware(BodyLimit, limit=settings.max_bytes)
    app.state.settings = settings
    app.include_router(router)

    @app.exception_handler(StorageError)
    async def speech_error(request, exc):
        return JSONResponse(
            {"detail": exc.detail},
            status_code=exc.status,
            headers={"Retry-After": "5"} if exc.status == 503 else None,
        )

    @app.middleware("http")
    async def timing(request: Request, call_next):
        request_id = str(uuid4())
        start = time.monotonic()
        logger.info(
            "request_id=%s started method=%s path=%s", request_id, request.method, request.url.path
        )
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request_id=%s status=%s elapsed_ms=%.2f",
            request_id,
            response.status_code,
            (time.monotonic() - start) * 1000,
        )
        return response

    @app.get("/health/live")
    async def live():
        return {"status": "ok"}

    @app.get("/health/ready")
    async def ready():
        worker = getattr(app.state, "worker", None)
        available = bool(worker and worker.ready)
        return JSONResponse({"ready": available}, status_code=200 if available else 503)

    return app


app = create_app()
