import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from Routers import build_api_router, preload_gallery
from contextlib import asynccontextmanager

# from Services.QwenVLService import get_pipe
logging.basicConfig(
    level=os.environ.get("RAIR_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("rair")

@asynccontextmanager
async def lifespan(app: FastAPI):
    preload_gallery()
    yield
    logger.info("App shutdown")

app = FastAPI(title="VLM Backend", 
              lifespan=lifespan
              )

# ---- CORS CONFIG ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",   # Angular dev server
        "http://127.0.0.1:4200",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(build_api_router(), prefix="/api/v1")

# app.mount('/media', StaticFiles(directory=r"F:\RAGInteractIR\datasets\VG\VG_100K"), name="images")

MEDIA_DIR = os.environ.get(
    "MEDIA_DIR",
    os.path.join(os.path.dirname(__file__), "datasets", "MSCOCO"),
)

if os.path.isdir(MEDIA_DIR):
    app.mount('/media', StaticFiles(directory=MEDIA_DIR), name="images")
else:
    logger.warning("MEDIA_DIR does not exist, skipping /media mount: %s", MEDIA_DIR)
