
from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.documents import router as documents_router
from app.api.routes.chat import router as chat_router

app = FastAPI(title="AI PDF Assistant")

app.include_router(
    health_router,
    prefix="/api",
    tags=["Health"],
)

app.include_router(
    documents_router,
    prefix="/api/documents",
    tags=["Documents"],
)

app.include_router(
    chat_router,
    prefix="/api",
    tags=["Chat"],
)


@app.get("/")
def root():
    return {"message": "AI PDF Assistant API"}
