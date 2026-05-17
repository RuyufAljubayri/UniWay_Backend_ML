---

## Production API Endpoints (Hugging Face Server)

The backend layer is fully active on Hugging Face Spaces. Below are the finalized web service routes for integration:

- Base System URL: https://rayouf0-uniway-backend-core.hf.space

Available Endpoints for Mobile Development:

1. Signage Prediction (POST):
   URL: https://rayouf0-uniway-backend-core.hf.space/predict
   Payload: Form-Data Key named "file" (binary image)

2. Location Search Engine (GET):
   URL: https://rayouf0-uniway-backend-core.hf.space/classrooms/search?query=ROOM_ID

3. Fetch User Bookmarks (GET):
   URL: https://rayouf0-uniway-backend-core.hf.space/bookmarks/my
   Header required: "x-device-id" containing the device token

4. Insert New Bookmark (POST):
   URL: https://rayouf0-uniway-backend-core.hf.space/bookmarks/add
   Header required: "x-device-id" containing the device token
   Body (JSON format): {"roomId": "ROOM_ID"}

---
