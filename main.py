from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from Routers import build_api_router
from contextlib import asynccontextmanager

from Services.QwenVLService import get_pipe

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=== Preloading Qwen pipeline ===")
    get_pipe()
    print("=== Qwen pipeline ready ===")
    yield
    print("=== App shutdown ===")

app = FastAPI(title="VLM Backend", lifespan=lifespan)

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

app.mount('/media', StaticFiles(directory="/home/map4/ThisPC/RAGInteractIR/datasets/MSCOCO"), name="images")