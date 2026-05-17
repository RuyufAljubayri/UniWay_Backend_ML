---

## UniWay Centralized API Endpoints (Live Production)

The backend core is successfully deployed on Hugging Face Spaces with a 16GB RAM infrastructure. Use the following production URLs inside the mobile/AR client applications:

* **Base URL:** `https://rayouf0-uniway-backend-core.hf.space`

### 🔗 Available Route Mappings:

1. **Signage Localization & OCR (POST):**
   * **URL:** `https://rayouf0-uniway-backend-core.hf.space/predict`
   * **Payload:** Multipart/Form-Data (`file`: Image Binary)

2. **Global Classroom Search (GET):**
   * **URL:** `https://rayouf0-uniway-backend-core.hf.space/classrooms/search?query=YOUR_ROOM_ID`

3. **Fetch Device Bookmarks (GET):**
   * **URL:** `https://rayouf0-uniway-backend-core.hf.space/bookmarks/my`
   * **Header Required:** `x-device-id`: `YOUR_DEVICE_UUID`

4. **Add Classroom to Bookmarks (POST):**
   * **URL:** `https://rayouf0-uniway-backend-core.hf.space/bookmarks/add`
   * **Header Required:** `x-device-id`: `YOUR_DEVICE_UUID`
   * **Body (JSON):** `{"roomId": "ROOM_ID"}`

---
