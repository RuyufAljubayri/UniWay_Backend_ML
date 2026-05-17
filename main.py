import io
import os
import sys
import asyncio
import numpy as np
from PIL import Image
import nest_asyncio
import cv2
import urllib.request

from fastapi import FastAPI, Header, HTTPException, status, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import firebase_admin
from firebase_admin import credentials, firestore

from ultralytics import YOLO
import easyocr

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
models_loaded = False  # Flag to track if background initialization is complete

class BookmarkPayload(BaseModel):
    roomId: str

def normalize_ml_text(text: str) -> str:
    if not text:
        return ""
    text = str(text).strip().lower()
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ة", "ه").replace("ى", "ي")
    text = text.replace(" ", "").replace("\\t", "").replace("\\n", "")
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

def download_file_from_drive(file_id, output_path):
    if not os.path.exists(output_path):
        print(f"[DOWNLOAD] Asset {output_path} not found locally. Fetching from Google Drive cloud core...")
        url = f"https://docs.google.com/uc?export=download&id={file_id}&confirm=t"
        try:
            urllib.request.urlretrieve(url, output_path)
            print(f"[DOWNLOAD] Success! Asset {output_path} synchronized securely.")
        except Exception as e:
            print(f"[CRITICAL] Download failed for asset {output_path}: {str(e)}")
    else:
        print(f"[LOCAL] Asset {output_path} discovered in root registry. Skipping download sequence.")

async def initialize_ml_models_background():
    """تشغيل تحميل وتهيئة الموديلات الثقيلة في الخلفية لتفادي توقف البورت في السيرفر السحابي"""
    global detection_model, reader, models_loaded
    print("[BACKGROUND INIT] Starting heavy ML asset synchronization from Drive...")
    
    # Run synchronous downloads in a separate thread context to avoid blocking the loop
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, download_file_from_drive, "1EC86KnVqaDQT1ipgkWyYVwie1xU4TWVD", "best_uqu_v1.pt")
    await loop.run_in_executor(None, download_file_from_drive, "1mmvmR4Lhp_Zc6EdFqI7PsfwXul8tYrHi", "best_accuracy.pth")
    
    try:
        model_path = os.environ.get("YOLO_MODEL_PATH", "best_uqu_v1.pt")
        if os.path.exists(model_path):
            detection_model = YOLO(model_path)
            print(f"[BACKGROUND INIT] YOLOv8 weight matrix loaded securely.")
        else:
            print(f"[BACKGROUND WARNING] Weight matrix file not found.")
    except Exception as yolo_err:
        print(f"[BACKGROUND CRITICAL] Spatial framework configuration locked: {str(yolo_err)}")

    try:
        reader = easyocr.Reader(['ar', 'en'], gpu=False, model_storage_directory=".", user_network_directory=".")
        print("[BACKGROUND INIT] Bilingual EasyOCR pipelines fully generated.")
    except Exception as ocr_err:
        print(f"[BACKGROUND CRITICAL] Linguistic pipeline failed to compile: {str(ocr_err)}")
        
    models_loaded = True
    print("[BACKGROUND INIT] All ML systems operational and active.")

@app.on_event("startup")
async def startup_event():
    global db
    print("[INIT] Igniting cloud resources initialization sequence...")
    
    try:
        cred_path = os.environ.get("FIREBASE_CREDENTIALS_PATH", "serviceAccountKey.json")
        if not firebase_admin._apps:
            if os.path.exists(cred_path):
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
                print("[INIT] Firebase Administrative SDK bound successfully.")
            else:
                print(f"[CRITICAL] Firebase key missing at {cred_path}.")
        db = firestore.client()
    except Exception as fb_err:
        print(f"[CRITICAL] Firebase administrative handshake ruptured: {str(fb_err)}")

    # Fire and forget the heavy ML downloads and setups into the async background pipeline immediately
    asyncio.create_task(initialize_ml_models_background())
    print("[INIT] Port binding released to Render core. Server is opening ports globally.")

@app.post("/predict")
async def predict_signage(file: UploadFile = File(...)):
    if not models_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="AI Core is still caching weight blocks from Google Drive in the background. Please retry in a few moments."
        )
        
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
            return {"status": "error", "error_code": "MODEL_UNAVAILABLE", "message": "YOLOv8 weights are unavailable."}

        detection_results = detection_model(image, conf=0.3)
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

        clean_str = clean_str.replace('ا', 'أ').replace('إ', 'أ').replace('آ', 'أ').replace('A', 'أ').replace('a', 'أ')
        clean_str = clean_str.replace('D', 'د').replace('d', 'د')

        raw_digits = "".join(filter(str.isdigit, clean_str))
        letters_part = "".join(filter(lambda x: not x.isdigit(), clean_str)).replace(" ", "")
        digits_part = enforce_uqu_room_constraints(raw_digits)

        if not letters_part:
            letters_part = "أ"

        target_room_id = f"{digits_part}{letters_part}"

        if db is None:
            return {"status": "error", "error_code": "FIRESTORE_OFFLINE", "message": "Database link unavailable."}

        try:
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
            
            normalized_ocr_query = normalize_ml_text(raw_text)
            all_classrooms = db.collection('ClassRoom').stream()
            
            for room_doc in all_classrooms:
                room_data = room_doc.to_dict()
                db_classname = room_data.get("classname", room_data.get("className", ""))
                normalized_db_classname = normalize_ml_text(db_classname)
                
                if normalized_db_classname and ((normalized_db_classname in normalized_ocr_query) or (normalized_ocr_query in normalized_db_classname)):
                    return {
                        "status": "success", "message": "Match Found via High-Tolerance Classroom Name Attribute Lookup",
                        "data": {
                            "detected_text_raw": raw_text, "processed_room_id": room_doc.id, "className": db_classname,
                            "buildingId": room_data.get("buildingId", "Not Specified"), "floorNum": room_data.get("floorNum", "Not Specified"),
                            "description": room_data.get("description", "No description available.")
                        }
                    }

            fallback_id = f"{target_room_id} "
            doc_ref_fb = db.collection('ClassRoom').document(fallback_id)
            doc_fb = doc_ref_fb.get()

            if doc_fb.exists:
                data_fb = doc_fb.to_dict()
                return {
                    "status": "success", "message": "Match Found directly via Document ID Lookup (with padding)",
                    "data": {
                        "detected_text_raw": raw_text, "processed_room_id": doc_fb.id, "className": target_room_id,
                        "buildingId": data_fb.get("buildingId", "Not Specified"), "floorNum": data_fb.get("floorNum", "Not Specified"),
                        "description": data_fb.get("description", "No description available.")
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
        user_doc_ref = db.collection('Users').document(x_device_id)
        user_doc = user_doc_ref.get()

        if not user_doc.exists:
            return {"status": "success", "device_tracked": x_device_id, "total_bookmarks": 0, "data": []}

        user_data = user_doc.to_dict()
        bookmark_ids = user_data.get('bookmarks', [])

        if not bookmark_ids:
            return {"status": "success", "device_tracked": x_device_id, "total_bookmarks": 0, "data": []}

        detailed_bookmarks = []
        for room_id in bookmark_ids:
            room_ref = db.collection('ClassRoom').document(str(room_id).strip())
            room_doc = room_ref.get()
            
            if room_doc.exists:
                r_data = room_doc.to_dict()
                detailed_bookmarks.append({
                    "roomId": room_doc.id,
                    "className": r_data.get("classname", r_data.get("className", room_doc.id)),
                    "buildingId": r_data.get("buildingId", "Not Specified"),
                    "floorNum": r_data.get("floorNum", "Not Specified"),
                    "description": r_data.get("description", "No description available.")
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
        if not room_ref.get().exists:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Target room identifier '{payload.roomId}' does not map to any database entity.")

        user_doc_ref = db.collection('Users').document(x_device_id)
        user_doc = user_doc_ref.get()

        if user_doc.exists:
            user_data = user_doc.to_dict()
            current_bookmarks = user_data.get('bookmarks', [])
            if payload.roomId not in current_bookmarks:
                current_bookmarks.append(payload.roomId)
                user_doc_ref.update({'bookmarks': current_bookmarks})
        else:
            user_doc_ref.set({'bookmarks': [payload.roomId]})

        return {
            "status": "success",
            "message": f"Room '{payload.roomId}' securely appended to device profile arrays.",
            "device": x_device_id
        }
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to complete transactional updates: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
