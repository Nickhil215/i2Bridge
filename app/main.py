from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import settings
from app.core.telemetry import setup_telemetry
from app.routers import analyzer_router


@asynccontextmanager
async def lifespan(context):
    # Setup
    setup_telemetry()
    yield
    # Cleanup
    pass

def create_app() -> FastAPI:

    app = FastAPI(
        title="i2-bridge",
        version=settings.VERSION,
        description="Integration bridge service",
        docs_url=f"{settings.API_V1_STR}/docs",
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        lifespan=lifespan,
        root_path=settings.API_V1_STR
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(
        analyzer_router.app,
        tags=["kg"]
    )

    return app

app = create_app() 