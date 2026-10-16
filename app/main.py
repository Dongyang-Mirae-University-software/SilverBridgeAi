from fastapi import FastAPI
from app.emotion.router import router as emotion_router
from app.companion.router import router as companion_router

app = FastAPI()
app.include_router(emotion_router)
app.include_router(companion_router)
