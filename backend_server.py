import base64
import json
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from backend.hand_detector import HandDetector
from backend.model_utils import SignLanguageClassifier
from backend.inference_pipeline import (
    PredictionStabilizer,
    build_debug_overlay,
    encode_image_jpeg,
    CONFIDENCE_THRESHOLD,
)

app = FastAPI(title="SignBridge Real-Time API Server")

# Enable CORS for React Dev Client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engines once on startup
detector = None
classifier = None

@app.on_event("startup")
def startup_event():
    global detector, classifier
    print("[SERVER] Initializing SignLanguageClassifier...")
    classifier = SignLanguageClassifier()
    print("[SERVER] Initializing HandDetector...")
    detector = HandDetector()
    print("[SERVER] All engines loaded and ready.")
    print(f"[SERVER] Inference fixes active: padding=0.45, rotation=90CW, "
          f"confidence>={CONFIDENCE_THRESHOLD}, stabilization=15 frames")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "use_mediapipe": detector.use_mediapipe if detector else False,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "crop_padding": 0.45,
        "stabilization_window": 15,
    }

@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    debug: bool = Query(False, description="Enable debug overlay in responses"),
):
    await websocket.accept()
    stabilizer = PredictionStabilizer()
    print(f"[WS] Client connected (debug={debug}).")
    try:
        while True:
            # Receive base64 frame string from React
            data = await websocket.receive_text()
            
            # Remove base64 header data:image/jpeg;base64, if present
            if "," in data:
                data = data.split(",")[1]
            
            # Decode frame
            img_data = base64.b64decode(data)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                await websocket.send_json({"detected": False, "error": "Invalid frame decode"})
                continue
            
            # Run detection
            results = detector.find_hand_landmarks(frame, mode="Skin Color")
            cropped_roi, bbox, _ = detector.get_hand_roi(frame, results, mode="Skin Color")
            
            response = {"detected": False}
            
            # Process landmarks if hand detected
            if results and hasattr(results, 'hand_landmarks') and results.hand_landmarks:
                response["detected"] = True
                
                # Get all hands' 21 landmarks
                all_hands = []
                for hand_lms in results.hand_landmarks:
                    hand_points = []
                    for lm in hand_lms:
                        hand_points.append([float(lm.x * frame.shape[1]), float(lm.y * frame.shape[0])])
                    all_hands.append(hand_points)
                
                response["hands"] = all_hands
                response["landmarks"] = all_hands[0]  # backward compatibility
                
                if bbox is not None:
                    response["bbox"] = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                
                # Predict character through fixed inference pipeline
                if cropped_roi is not None and cropped_roi.size > 0:
                    result = stabilizer.predict_from_roi(classifier, cropped_roi, apply_rotation=True)
                    response["letter"] = result["letter"]
                    response["confidence"] = float(result["confidence"])
                    response["raw_letter"] = result["raw_letter"]
                    response["raw_confidence"] = float(result["raw_confidence"])
                    response["top_3"] = [
                        {"letter": lbl, "confidence": conf}
                        for lbl, conf in result.get("top_3", [])
                    ]

                    if debug:
                        overlay = build_debug_overlay(
                            frame=frame,
                            bbox=bbox,
                            cropped_roi=cropped_roi,
                            rotated_roi=result["rotated_roi"],
                            model_input=result["model_input"],
                            letter=result["letter"],
                            confidence=result["confidence"],
                            raw_letter=result["raw_letter"],
                            raw_confidence=result["raw_confidence"],
                        )
                        response["debug_overlay"] = encode_image_jpeg(overlay)
                        response["debug"] = {
                            "cropped_roi": encode_image_jpeg(cropped_roi, max_width=160),
                            "rotated_roi": encode_image_jpeg(result["rotated_roi"], max_width=160),
                            "model_input": encode_image_jpeg(result["model_input"], max_width=128),
                        }
                else:
                    stabilizer.reset()
                    response["letter"] = "Unknown"
                    response["confidence"] = 0.0
            else:
                stabilizer.reset()
            
            # Send prediction packet back to client
            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        print("[WS] Client disconnected.")
    except Exception as e:
        print(f"[WS] Error: {e}")
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    uvicorn.run("backend_server:app", host="0.0.0.0", port=8000, reload=False)
