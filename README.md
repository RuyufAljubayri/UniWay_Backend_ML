## Production API Endpoints (Live)

The following URLs represent the active production endpoints for integrating the mobile application and AR client services:

**Base URL:** `https://rayouf0-uniway-backend-core.hf.space`

---

### Available Endpoints and Routes

#### 1. Signage Prediction and OCR (POST)
* **Description:** Used in the Camera/Scanner screen to capture and process signage images.
* **URL:** `https://rayouf0-uniway-backend-core.hf.space/predict`
* **Payload:** Form-Data (Key: `file`, Type: image)

#### 2. Location Search Engine (GET)
* **Description:** Used in the Search screen to query classrooms natively across all buildings.
* **URL:** `https://rayouf0-uniway-backend-core.hf.space/classrooms/search?query=CLASS_ID`

#### 3. Fetch Device Bookmarks (GET)
* **Description:** Used in the Saved Routes/Bookmarks dashboard to retrieve the user's saved locations.
* **URL:** `https://rayouf0-uniway-backend-core.hf.space/bookmarks/my`
* **Header:** `x-device-id`: `YOUR_DEVICE_TOKEN`

#### 4. Insert New Bookmark (POST)
* **Description:** Used when hitting the "Save/Bookmark" button to map a custom composite ID and activate user status.
* **URL:** `https://rayouf0-uniway-backend-core.hf.space/bookmarks/add`
* **Header:** `x-device-id`: `YOUR_DEVICE_TOKEN`
* **Body (JSON):** ```json
{
    "roomId": "CLASS_ID"
}
