from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import numpy as np
from PIL import Image
import io
import sys
import os
import base64
import cv2
import torch


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline.image_pipeline import ImagePipeline
from pipeline.video_pipeline import VideoPipeline

app = FastAPI(title="DeepShield API", version="1.0")

# CORS - HTML file connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pipelines
image_pipeline = ImagePipeline()
video_pipeline = VideoPipeline()

# Load trained model weights
model_weights_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 
    'models', 'best_model.pth'
)

if os.path.exists(model_weights_path):
    image_pipeline.model.load_state_dict(
        torch.load(model_weights_path, map_location='cpu')
    )
    image_pipeline.model.eval()
    video_pipeline.model.load_state_dict(
        torch.load(model_weights_path, map_location='cpu')
    )
    video_pipeline.model.eval()
    print("Trained model loaded!")
else:
    print("Warning: No trained model found, using random weights!")

@app.get("/")
def root():
    return FileResponse("app/deepshield_app.html")

@app.get("/health")
def health():
    return {"status": "operational", "model": "EfficientNet-B4"}



@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    import base64
    
    contents = await file.read()
    
    # Image format handle karo
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        return {"label": "ERROR", "confidence": 0, 
                "faces_detected": 0, "artifact_score": "N/A",
                "frames_analyzed": None, "heatmap": None,
                "error": "Invalid image format. Use JPG or PNG."}
    
    image_np = np.array(image)
    result = image_pipeline.run(image_np)
    
    if 'error' in result:
        return {"label": "NO_FACE", "confidence": 0,
                "faces_detected": 0, "artifact_score": "N/A",
                "frames_analyzed": None, "heatmap": None}
    
    heatmap_b64 = None
    if result.get("heatmap") is not None:
        _, buffer = cv2.imencode('.jpg', result["heatmap"])
        heatmap_b64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "label": result.get("label", "ERROR"),
        "confidence": result.get("confidence", 0),
        "faces_detected": 1,
        "artifact_score": "High" if result.get("label") == "FAKE" else "Low",
        "frames_analyzed": None,
        "heatmap": heatmap_b64
    }

@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    import tempfile
    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
    result = video_pipeline.run(tmp_path)
    os.unlink(tmp_path)
    return {
        "label": result.get("label", "ERROR"),
        "confidence": result.get("confidence", 0),
        "faces_detected": 1,
        "artifact_score": "High" if result.get("label") == "FAKE" else "Low",
        "frames_analyzed": result.get("frames_analyzed", 0)
    }