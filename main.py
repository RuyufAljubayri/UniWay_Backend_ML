import io
import os
import sys
import json
import asyncio
import numpy as np
from PIL import Image
import nest_asyncio
import cv2
import torch

from fastapi import FastAPI, Header, HTTPException, status, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import firebase_admin
from firebase_admin import credentials, firestore

from ultralytics import YOLO
import easyocr

try:
    import ultralytics.nn.tasks
    import torch.nn.modules.container
    import torch.nn.modules.conv
    import torch.nn.modules.activation
    import torch.nn.modules.batchnorm
    import ultralytics.nn.modules.block
    import ultralytics.nn.modules.conv
    import ultralytics.nn.modules.head

    torch.serialization.add_safe_globals([
        ultralytics.nn.tasks.DetectionModel,
        torch.nn.modules.container.Sequential,
        torch.nn.modules.conv.Conv2d,
        torch.nn.modules.activation.SiLU,
        torch.nn.modules.batchnorm.BatchNorm2d,
        ultralytics.nn.modules.block.C3k2,
        ultralytics.nn.modules.block.C2f,
        ultralytics.nn.modules.block.C3k,
        ultralytics.nn.modules.block.Bottleneck,
        ultralytics.nn.modules.block.DFL,
        ultralytics.nn.modules.conv.Conv,
        ultralytics.nn.modules.conv.Concat,
        ultralytics.nn.modules.head.Detect,
        torch.Size
    ])
except Exception:
    pass

app = FastAPI(
    title="UniWay Centralized Backend API",
    description="Asynchronous cloud core serving computer vision workflows and secure Firestore transaction mappings.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = None
detection_model = None
reader = None

class BookmarkPayload(BaseModel):
    roomId: str

def normalize_ml_text(text: str) -> str:
    if not text:
        return ""
    text = str(text).strip().lower()
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ة", "ه").replace("ى", "ي")
    text = text.replace(" ", "").replace("\t", "").replace("\n", "")
    return text

def enforce_uqu_room_constraints(digits: str) -> str:
    if not digits:
        return ""
    if len(digits) == 4:
        if digits.endswith('1'):
            return digits[:3]
        elif digits.startswith('1'):
            return digits[1:]
        return digits[:3]
    return digits

def your_custom_preprocessing_pipeline(pil_img):
    try:
        open_cv_image = np.array(pil_img)
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
                
        lab = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
                
        final_img = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        return final_img
    except Exception:
        return None

@app.on_event("startup")
async def startup_event():
    global db, detection_model, reader
    print("[INIT] Igniting cloud resources initialization sequence...")
    import urllib.request
        
    try:
        if not firebase_admin._apps:
            # محاولة القراءة الآمنة من متغيرات البيئة في هاقينق فيس أولاً
            firebase_secret = os.environ.get("FIREBASE_KEY")
            
            if firebase_secret:
                print("[INIT] Detecting production secrets environment variable...")
                cred_dict = json.loads(firebase_secret)
                cred = credentials.Certificate(cred_dict)
                firebase_admin.initialize_app(cred)
                print("[INIT] Firebase Administrative SDK bound successfully via Production Secrets.")
            else:
                # النظام الاحتياطي للملف المحلي في حال التجربة المحلية
                cred_path = os.environ.get("FIREBASE_CREDENTIALS_PATH", "serviceAccountKey.json")
                if os.path.exists(cred_path):
                    cred = credentials.Certificate(cred_path)
                    firebase_admin.initialize_app(cred)
                    print(f"[INIT] Firebase Administrative SDK bound successfully via local file: {cred_path}")
                else:
                    print(f"[CRITICAL] Firebase key missing at environment variables and local path. Cloud repositories will be offline.")
        
        db = firestore.client()
    except Exception as fb_err:
        print(f"[CRITICAL] Firebase administrative handshake ruptured: {str(fb_err)}")

    try:
        model_filename = "best_uqu_v1.pt"
        drive_url = f"https://docs.google.com/uc?export=download&id=1EC86KnVqaDQT1ipgkWyYVwie1xU4TWVD"
        
        if not os.path.exists(model_filename) or os.path.getsize(model_filename) < 1000000:
            print(f"[INIT] Downloading high-resolution weights directly from secure stream...")
            urllib.request.urlretrieve(drive_url, model_filename)
            print(f"[INIT] Dynamic stream sync completed successfully.")
            
        if os.path.exists(model_filename):
            detection_model = YOLO(model_filename)
            print(f"[INIT] YOLO weight matrix loaded securely from: {model_filename}")
        else:
            detection_model = YOLO("yolov8n.pt")
            print("[WARNING] Custom weights transfer delayed. Loaded fallback yolov8n.pt successfully.")
    except Exception as yolo_err:
        print(f"[CRITICAL] Spatial framework configuration locked: {str(yolo_err)}")

    try:
        reader = easyocr.Reader(['ar', 'en'], gpu=False, model_storage_directory=".", user_network_directory=".")
        print("[INIT] Asynchronous bilingual EasyOCR pipelines generated with dynamic custom weights.")
    except Exception as ocr_err:
        print(f"[CRITICAL] Linguistic pipeline parsing arrays failed to compile: {str(ocr_err)}")

@app.post("/predict")
async def predict_signage(file: UploadFile = File(...)):
    global detection_model
    try:
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as img_err:
            return {
                "status": "error", "error_code": "INVALID_IMAGE_PAYLOAD",
                "message": f"Submitted binary file is corrupted: {str(img_err)}"
            }

        if detection_model is None:
            try:
                if os.path.exists("best_uqu_v1.pt"):
                    detection_model = YOLO("best_uqu_v1.pt")
                else:
                    detection_model = YOLO("yolov8n.pt")
            except Exception as e:
                return {"status": "error", "error_code": "MODEL_UNAVAILABLE", "message": f"YOLO weights allocation failed: {str(e)}"}

        try:
            detection_results = detection_model(image, conf=0.3)
        except Exception as run_err:
            return {"status": "error", "error_code": "MODEL_UNAVAILABLE", "message": f"YOLO execution runtime fault: {str(run_err)}"}

        if len(detection_results[0].boxes) == 0:
            return {"status": "error", "error_code": "NO_SIGNAGE_FOUND", "message": "Signage localization failed."}

        box = detection_results[0].boxes[0].xyxy[0].cpu().numpy().astype(int)
        if (box[2] <= box[0]) or (box[3] <= box[1]):
            return {"status": "error", "error_code": "INVALID_BOUNDING_BOX", "message": "Invalid bounding box localization dimensions."}
                    
        cropped_img = image.crop((box[0], box[1], box[2], box[3]))
        enhanced_numpy = your_custom_preprocessing_pipeline(cropped_img)
        if enhanced_numpy is None:
            enhanced_numpy = np.array(cropped_img)

        raw_text = ""
        if reader is not None:
            ocr_results = reader.readtext(enhanced_numpy, detail=1)
            raw_text = " ".join([res[1] for res in ocr_results]) if ocr_results else ""
            if not raw_text:
                ocr_results_raw = reader.readtext(np.array(cropped_img))
                raw_text = " ".join([res[1] for res in ocr_results_raw]) if ocr_results_raw else ""

        if not raw_text:
            return {"status": "error", "error_code": "OCR_FAILED", "message": "Text extraction failed."}

        hindi_to_eng = {'١':'1','٢':'2','٣':'3','٤':'4','٥':'5','٦':'6','٧':'7','٨':'8','٩':'9','٠':'0'}
        clean_str = str(raw_text).strip()
        for h, e in hindi_to_eng.items():
            clean_str = clean_str.replace(h, e)

        clean_str = clean_str.replace('إ', 'أ').replace('آ', 'أ').replace('A', 'أ').replace('a', 'أ')
        clean_str = clean_str.replace('D', 'د').replace('d', 'د')
        
        raw_digits = "".join(filter(str.isdigit, clean_str))
        letters_part = "".join(filter(lambda x: not x.isdigit(), clean_str)).replace(" ", "")
        digits_part = enforce_uqu_room_constraints(raw_digits)

        target_room_id = f"{digits_part}{letters_part}" if letters_part else digits_part

        if db is None:
            return {"status": "error", "error_code": "FIRESTORE_OFFLINE", "message": "Database link unavailable."}

        try:
            if letters_part:
                doc_ref = db.collection('ClassRoom').document(target_room_id)
                doc = doc_ref.get()
                if doc.exists:
                    data = doc.to_dict()
                    return {
                        "status": "success", "message": "Match Found directly via Document ID Lookup",
                        "data": {
                            "detected_text_raw": raw_text, "processed_room_id": doc.id, "className": doc.id,
                            "buildingId": data.get("buildingId", "Not Specified"), "floorNum": data.get("floorNum", "Not Specified"),
                            "description": data.get("description", "No description available.")
                        }
                    }

            normalized_ocr_query = normalize_ml_text(digits_part)
            all_classrooms = db.collection('ClassRoom').stream()
                        
            for room_doc in all_classrooms:
                room_id_str = str(room_doc.id)
                room_data = room_doc.to_dict()
                db_classname = room_data.get("classname", room_data.get("className", ""))
                
                normalized_db_id = normalize_ml_text(room_id_str)
                normalized_db_classname = normalize_ml_text(db_classname)
                                
                if (normalized_ocr_query in normalized_db_id) or (normalized_ocr_query in normalized_db_classname):
                    return {
                        "status": "success", "message": "Match Found via High-Tolerance Digits Lookup Sequence",
                        "data": {
                            "detected_text_raw": raw_text, "processed_room_id": room_doc.id, "className": db_classname if db_classname else room_doc.id,
                            "buildingId": room_data.get("buildingId", "Not Specified"), "floorNum": room_data.get("floorNum", "Not Specified"),
                            "description": room_data.get("description", "No description available.")
                        }
                    }

            return {
                "status": "success", "message": "Location could not be identified in database schemas.",
                "data": {
                    "detected_text_raw": raw_text, "processed_room_id": target_room_id, "className": target_room_id,
                    "buildingId": "Unknown", "floorNum": "Unknown", "description": "Location key not found in current database mapping."
                }
            }
        except Exception as db_err:
            return {"status": "error", "error_code": "DATABASE_ERROR", "message": str(db_err)}
    except Exception as e:
        return {"status": "error", "error_code": "SERVER_ERROR", "message": str(e)}

@app.get("/classrooms/search")
async def search_classrooms(query: str):
    if not query:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Search context query parameter string cannot be empty.")
            
    if db is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Cloud database engine offline.")

    try:
        normalized_user_query = normalize_ml_text(query)
        matching_results = []
        classrooms_stream = db.collection('ClassRoom').stream()

        for doc in classrooms_stream:
            doc_id = str(doc.id)
            room_data = doc.to_dict()
                        
            db_classname = room_data.get("classname", room_data.get("className", ""))
            normalized_id = normalize_ml_text(doc_id)
            normalized_classname = normalize_ml_text(db_classname)

            if (normalized_user_query in normalized_id) or (normalized_user_query in normalized_classname):
                matching_results.append({
                    "roomId": doc_id,
                    "className": db_classname if db_classname else doc_id,
                    "buildingId": room_data.get("buildingId", "Not Specified"),
                    "floorNum": room_data.get("floorNum", "Not Specified"),
                    "description": room_data.get("description", "No description available.")
                })

        return {
            "status": "success",
            "query_evaluated": query,
            "results_count": len(matching_results),
            "data": matching_results
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Search pipeline operational error: {str(e)}")

@app.get("/bookmarks/my")
async def fetch_my_bookmarks(x_device_id: str = Header(None, alias="x-device-id")):
    if not x_device_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Required hardware identification token 'x-device-id' header is missing.")
            
    if db is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Active database connections are currently unavailable.")

    try:
        bookmarks_stream = db.collection('Bookmark').where('userId', '==', x_device_id).stream()
        detailed_bookmarks = []

        for b_doc in bookmarks_stream:
            b_data = b_doc.to_dict()
            room_id = b_data.get('classId')
            
            if room_id:
                room_ref = db.collection('ClassRoom').document(str(room_id).strip())
                room_doc = room_ref.get()
                
                if room_doc.exists:
                    r_data = room_doc.to_dict()
                    detailed_bookmarks.append({
                        "bookmarkDocId": b_doc.id,
                        "roomId": room_doc.id,
                        "className": r_data.get("classname", r_data.get("className", room_doc.id)),
                        "buildingId": r_data.get("buildingId", "Not Specified"),
                        "floorNum": r_data.get("floorNum", "Not Specified"),
                        "description": b_data.get("description", r_data.get("description", "No description available."))
                    })

        return {
            "status": "success",
            "device_tracked": x_device_id,
            "total_bookmarks": len(detailed_bookmarks),
            "data": detailed_bookmarks
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to pull profile configurations: {str(e)}")

@app.post("/bookmarks/add")
async def add_bookmark(payload: BookmarkPayload, x_device_id: str = Header(None, alias="x-device-id")):
    if not x_device_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Required hardware identification token 'x-device-id' header is missing.")
            
    if db is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database connectivity dropped.")

    try:
        room_ref = db.collection('ClassRoom').document(payload.roomId)
        room_doc = room_ref.get()
        if not room_doc.exists:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Target room identifier '{payload.roomId}' does not map to any database entity.")

        room_data = room_doc.to_dict()
        room_description = room_data.get('description', 'قاعة دراسية')

        user_doc_ref = db.collection('User').document(x_device_id)
        user_doc_ref.set({"isActive": True}, merge=True)

        custom_bookmark_id = f"{x_device_id}_{payload.roomId}"

        new_bookmark_data = {
            "classId": payload.roomId,
            "userId": x_device_id,
            "description": room_description
        }
        
        db.collection('Bookmark').document(custom_bookmark_id).set(new_bookmark_data)

        return {
            "status": "success",
            "message": f"Room '{payload.roomId}' successfully mapped with custom ID '{custom_bookmark_id}'.",
            "device": x_device_id
        }
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to complete transactional updates: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)
