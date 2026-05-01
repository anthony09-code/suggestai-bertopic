from api.routes import router
from fastapi import FastAPI

app = FastAPI(title="BERTopic Service")

app.include_router(router)
