# UniWay - Backend Core Production

This repository contains the backend core source code for the UniWay navigation system. The server is currently deployed and fully active on Hugging Face Spaces utilizing a 16GB RAM infrastructure.

---

## Production API Endpoints (Live)

The following URLs represent the active production endpoints for integrating the mobile application and AR client services:

* **Base URL:** `https://rayouf0-uniway-backend-core.hf.space`

### Available Endpoints and Routes

1. **Signage Prediction and OCR (POST):**
   * **URL:** `https://rayouf0-uniway-backend-core.hf.space/predict`
   * **Payload:** Form-Data (Key: `file`, Type: `image`)

2. **Location Search Engine (GET):**
   * **URL:** `https://rayouf0-uniway-backend-core.hf.space/classrooms/search?query=ROOM_ID`

3. **Fetch Device Bookmarks (GET):**
   * **URL:** `https://rayouf0-uniway-backend-core.hf.space/bookmarks/my`
   * **Header:** `x-device-id`: `YOUR_DEVICE_TOKEN`

4. **Insert New Bookmark (POST):**
   * **URL:** `https://rayouf0-uniway-backend-core.hf.space/bookmarks/add`
   * **Header:** `x-device-id`: `YOUR_DEVICE_TOKEN`
   * **Body (JSON):** `{"roomId": "ROOM_ID"}`
