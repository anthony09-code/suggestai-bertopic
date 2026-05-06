from fastapi import FastAPI

from app.api.routes import router

app = FastAPI(title="BERTopic Service")
app.include_router(router, prefix="/api")
