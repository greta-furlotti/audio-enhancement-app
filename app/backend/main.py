from fastapi import FastAPI, UploadFile, File
import shutil
import sys
import os

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.inference.denoise import denoise_audio

api = FastAPI()

UPLOAD_DIR = "temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@api.post("/denoise")
async def denoise(file: UploadFile = File(...)):
    input_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    output_path = denoise_audio(input_path)

    return {"output_file": output_path}