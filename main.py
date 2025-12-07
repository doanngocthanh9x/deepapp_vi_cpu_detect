from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import os
import uuid
from PIL import Image
import io
import numpy as np
import cv2
import json
from PaddletOCRApi import PaddleOCRProcessor
from typing import Optional
import sqlite3
from datetime import datetime

# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

app = FastAPI(title="YOLOv8 Detection API")

app.mount("/static", StaticFiles(directory="static"), name="static")

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
MODELS_DB = os.path.join(MODELS_DIR, "models.db")


def init_db():
    conn = sqlite3.connect(MODELS_DB)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS models (
            id TEXT PRIMARY KEY,
            name TEXT UNIQUE,
            filename TEXT,
            ext TEXT,
            uploaded_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()


def insert_model_record(model_id: str, name: str, filename: str, ext: str):
    conn = sqlite3.connect(MODELS_DB)
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO models (id, name, filename, ext, uploaded_at) VALUES (?, ?, ?, ?, ?)",
        (model_id, name, filename, ext, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_model_record_by_key(key: str):
    """Try to find model by id or by name. Returns row dict or None"""
    conn = sqlite3.connect(MODELS_DB)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM models WHERE id=?", (key,))
    row = cur.fetchone()
    if row:
        conn.close()
        return dict(row)
    cur.execute("SELECT * FROM models WHERE name=?", (key,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def delete_model_record_by_key(key: str):
    conn = sqlite3.connect(MODELS_DB)
    cur = conn.cursor()
    cur.execute("DELETE FROM models WHERE id=? OR name=?", (key, key))
    affected = cur.rowcount
    conn.commit()
    conn.close()
    return affected


# Initialize DB and sync simple existing files (if any)
init_db()
conn = sqlite3.connect(MODELS_DB)
cur = conn.cursor()
for filename in os.listdir(MODELS_DIR):
    model_id, ext = os.path.splitext(filename)
    if ext.lower() in ('.pt', '.onnx'):
        # if not present, insert with name equal to filename without ext
        cur.execute("SELECT 1 FROM models WHERE id=?", (model_id,))
        if cur.fetchone() is None:
            insert_model_record(model_id, model_id, filename, ext)
conn.close()

# Initialize PaddleOCR processor
ocr_processor = PaddleOCRProcessor()
@app.get("/download-models/{id}")
async def download_model(id: str):  
    rec = get_model_record_by_key(id)
    if rec:
        file_path = os.path.join(MODELS_DIR, rec['filename'])
        if os.path.exists(file_path):
            return FileResponse(file_path, media_type='application/octet-stream', filename=rec['filename'])
    raise HTTPException(status_code=404, detail="Model not found")
@app.get("/models")
async def list_models():
    conn = sqlite3.connect(MODELS_DB)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT id, name, filename, ext, uploaded_at FROM models ORDER BY uploaded_at DESC")
    rows = cur.fetchall()
    models = [dict(r) for r in rows]
    conn.close()
    return {"models": models}
@app.delete("/model/{model_key}")
async def delete_model(model_key: str):
    # Try to find record by id or name
    rec = get_model_record_by_key(model_key)
    if rec:
        file_path = os.path.join(MODELS_DIR, rec['filename'])
        if os.path.exists(file_path):
            os.remove(file_path)
        delete_model_record_by_key(model_key)
        return {"detail": "Model deleted"}

    # fallback: raw filename/id on disk
    for ext in ['.pt', '.onnx']:
        file_path = os.path.join(MODELS_DIR, f"{model_key}{ext}")
        if os.path.exists(file_path):
            os.remove(file_path)
            # also remove any DB rows referencing this filename
            delete_model_record_by_key(model_key)
            return {"detail": "Model deleted"}

    raise HTTPException(status_code=404, detail="Model not found")

@app.post("/upload-model")
async def upload_model(file: UploadFile = File(...), name: Optional[str] = Form(None)):
    # Accept an optional human-friendly name; otherwise use original filename (without ext)
    ext = os.path.splitext(file.filename)[1]
    if ext.lower() not in ('.pt', '.onnx'):
        raise HTTPException(status_code=400, detail="Model must be a .pt or .onnx file")

    model_id = str(uuid.uuid4())
    saved_filename = f"{model_id}{ext}"
    file_path = os.path.join(MODELS_DIR, saved_filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Determine display name
    display_name = (name.strip() if name and name.strip() else os.path.splitext(file.filename)[0])
    # Ensure unique name: if name exists, append short suffix
    conn = sqlite3.connect(MODELS_DB)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM models WHERE name=?", (display_name,))
    if cur.fetchone():
        display_name = f"{display_name}-{model_id[:8]}"
    conn.close()

    insert_model_record(model_id, display_name, saved_filename, ext)

    return {"model_id": model_id, "name": display_name, "filename": saved_filename}
@app.post("/detect/{model_key}")
async def detect(model_key: str, file: UploadFile = File(...)):
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image")
        
        # Resolve model by id or friendly name
        rec = get_model_record_by_key(model_key)
        model_path = None
        if rec:
            model_path = os.path.join(MODELS_DIR, rec['filename'])
        else:
            # fallback to raw id filename check
            for ext in ['.pt', '.onnx']:
                potential_path = os.path.join(MODELS_DIR, f"{model_key}{ext}")
                if os.path.exists(potential_path):
                    model_path = potential_path
                    break

        if not model_path:
            raise HTTPException(status_code=404, detail="Model not found")
        
        try:
            model = YOLO(model_path)
            names = model.names
            image = Image.open(io.BytesIO(await file.read()))

            # Get original size
            orig_width, orig_height = image.size

            # Resize image to model's input size
            imgsz = model.overrides.get('imgsz', 640)
            if isinstance(imgsz, list):
                imgsz = imgsz[0]
            image = image.resize((imgsz, imgsz), Image.LANCZOS)

            results = model(image)
            
            # Scale factors
            scale_x = orig_width / imgsz
            scale_y = orig_height / imgsz
            
            # Process results
            detections = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy.tolist()[0]
                    scaled_bbox = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
                    detections.append({
                        "class": int(box.cls),
                        "class_name": names[int(box.cls)],
                        "confidence": float(box.conf),
                        "bbox": [scaled_bbox]
                    })
            
            return {"detections": detections}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/multi-ocr")
async def multi_ocr_detect(files: list[UploadFile] = File(...)):
    results = []
    for file in files:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image")
    
        try:
            # Read image
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            
            # Convert PIL image to RGB if necessary (remove alpha channel, convert to RGB)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert PIL image to numpy array for PaddleOCR
            image_np = np.array(image)
            
            # Convert RGB to BGR for OpenCV (should be safe now)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            # Process with PaddleOCR
            result = ocr_processor.process_full_image(image_np)
            
            # Format results similar to YOLO detection
            detections = []
            for i, (text, confidence, bbox) in enumerate(zip(result['texts'], result['confidences'], result['bboxes'])):
                # Convert polygon bbox to xyxy format for consistency
                yolo_bbox = ocr_processor.convert_paddle_bbox_to_yolo(bbox)
                # Ensure yolo_bbox is a Python list with float values
                yolo_bbox = [float(x) for x in yolo_bbox]
                
                # Handle confidence - it might be a list or single value
                if isinstance(confidence, (list, tuple, np.ndarray)) and len(confidence) > 0:
                    # Take the average confidence if it's a list/array
                    avg_confidence = float(sum(float(c) for c in confidence) / len(confidence))
                elif isinstance(confidence, (list, tuple, np.ndarray)) and len(confidence) == 0:
                    avg_confidence = 0.0
                else:
                    avg_confidence = float(confidence) if confidence is not None else 0.0
                
                detections.append({
                    "class": int(0),  # Single class for text
                    "class_name": "text",
                    "confidence": avg_confidence,
                    "bbox": [yolo_bbox],
                    "text": str(text) if text else ""  # Ensure text is string
                })
        
        
            results.append({"filename": file.filename, "detections": detections})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return {"results": results}
@app.post("/ocr")
async def ocr_detect(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image")
    
    try:
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Convert PIL image to RGB if necessary (remove alpha channel, convert to RGB)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert PIL image to numpy array for PaddleOCR
        image_np = np.array(image)
        
        # Convert RGB to BGR for OpenCV (should be safe now)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Process with PaddleOCR
        result = ocr_processor.process_full_image(image_np)
        
        # Format results similar to YOLO detection
        detections = []
        for i, (text, confidence, bbox) in enumerate(zip(result['texts'], result['confidences'], result['bboxes'])):
            # Convert polygon bbox to xyxy format for consistency
            yolo_bbox = ocr_processor.convert_paddle_bbox_to_yolo(bbox)
            # Ensure yolo_bbox is a Python list with float values
            yolo_bbox = [float(x) for x in yolo_bbox]
            
            # Handle confidence - it might be a list or single value
            if isinstance(confidence, (list, tuple, np.ndarray)) and len(confidence) > 0:
                # Take the average confidence if it's a list/array
                avg_confidence = float(sum(float(c) for c in confidence) / len(confidence))
            elif isinstance(confidence, (list, tuple, np.ndarray)) and len(confidence) == 0:
                avg_confidence = 0.0
            else:
                avg_confidence = float(confidence) if confidence is not None else 0.0
            
            detections.append({
                "class": int(0),  # Single class for text
                "class_name": "text",
                "confidence": avg_confidence,
                "bbox": [yolo_bbox],
                "text": str(text) if text else ""  # Ensure text is string
            })
        
        return {"detections": detections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8111)
