# UniWay Backend & ML Engine

This repository contains the production backend architecture, REST APIs, and Machine Learning pipelines for UniWay indoor navigation.

---

## System Data Flow
* **Mobile App (Unity):** Sends localized signage images directly to the dual-stage Machine Learning pipelines.
* **ML Pipelines Engine:** Executes real-time YOLOv8 Sign Detection, crops target tensors, and passes arrays to EasyOCR for text extraction.
* **Mobile App (React Native):** Dispatches asynchronous HTTP requests natively to the unified backend core framework.
* **FastAPI Backend:** Coordinates request routing logic, handles endpoint processes, and synchronizes real-time state configurations.
* **Firebase Cloud Infrastructure:** Persists active session tokens, manages document queries, and caches device data.

---

## 1. Machine Learning Pipelines

* **UniWay_Detection_YOLOv8.ipynb (Signage Object Detection):** 
  * Custom trained YOLOv8 model to detect and localize indoor signs, room plaques, and directory boards within campus corridors.
* **UniWay_Recognition_OCR.ipynb (Text Recognition):**
  * Integrated EasyOCR pipeline to extract alphanumeric text from the YOLOv8 cropped bounding boxes, converting physical signs into string IDs (CLASS_ID).

---

## 2. Backend & API Architecture (main.py)

* **Asynchronous Engine:** Built using FastAPI (async/await) to handle concurrent database and routing operations with sub-second latency.
* **Cloud Infrastructure (serviceAccountKey.json):** Leverages Firebase Admin SDK for real-time database state persistence and device tracking.
* **API_Architecture_and_Testing.ipynb:** Dedicated test suite inside the repo to validate JSON schemas, request headers, and network performance.

---

## 3. Production API Endpoints (Live)

* **Production Base URL:** https://rayouf0-uniway-backend-core.hf.space

### Endpoints Specifications:

* **POST /predict**
  * **Payload:** multipart/form-data (Key: file, Type: image)
  * **Action:** Executes YOLOv8 + EasyOCR pipeline and returns room identification.
* **GET /classrooms/search?query=CLASS_ID**
  * **Action:** Runs native substring searches over campus spatial records.
* **GET /bookmarks/my**
  * **Headers:** x-device-id: <YOUR_DEVICE_TOKEN>
  * **Action:** Fetches user's saved routes linked to their hardware token.
* **POST /bookmarks/add**
  * **Headers:** x-device-id: <YOUR_DEVICE_TOKEN>
  * **Payload (application/json):** {"roomId": "CLASS_ID"}
  * **Action:** Appends and saves customized user destination points.

---

## Repository Blueprint
* main.py -> FastAPI gateway setup and router configurations.
* UniWay_Detection_YOLOv8.ipynb -> YOLOv8 image processing and inference pipeline.
* UniWay_Recognition_OCR.ipynb -> EasyOCR character extraction and string parsing.
* API_Architecture_and_Testing.ipynb -> Integration testing scenarios for endpoint validation.
* serviceAccountKey.json -> Secure Firebase access configurations.
* requirements.txt -> Pinned package dependency matrix.
