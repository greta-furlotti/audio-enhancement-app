from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

from src.inference.denoise import denoise_audio

api = FastAPI()

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@api.post("/denoise")
async def denoise(file: UploadFile = File(...)):
    input_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    output_path = denoise_audio(
        input_path,
        ckpt_path="models/unet_best.pt",
    )

    output_filename = os.path.basename(output_path)
    return {
        "output_file":     output_path,
        "output_filename": output_filename,
    }


@api.get("/audio/{filename}")
async def get_audio(filename: str):
    """Serve a processed audio file from the temp directory."""
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    return FileResponse(file_path, media_type="audio/wav", filename=filename)