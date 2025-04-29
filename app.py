# from flask import Flask, render_template, Response
# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import img_to_array
# import torch
# from PIL import Image

# app = Flask(__name__)

# # Load models
# emotion_model = load_model("emotion_model.h5")
# emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
# yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

# camera = cv2.VideoCapture(0)

# # Detection function
# def gen_frames():
#     frame_count = 0
#     yolo_results = None

#     while True:
#         success, frame = camera.read()
#         if not success:
#             break

#         # Face detection for emotion
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#         faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#         for (x, y, w, h) in faces:
#             roi = gray[y:y+h, x:x+w]
#             roi = cv2.resize(roi, (48, 48))
#             roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
#             roi = roi.astype("float32") / 255.0
#             roi = img_to_array(roi)
#             roi = np.expand_dims(roi, axis=0)
#             preds = emotion_model.predict(roi, verbose=0)[0]
#             label = emotion_labels[np.argmax(preds)]
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#             cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

#         # Run YOLOv5 every 5 frames for performance
#         if frame_count % 5 == 0:
#             pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#             yolo_results = yolo_model(pil_img, size=640)

#         if yolo_results is not None:
#             for *box, conf, cls in yolo_results.xyxy[0]:
#                 x1, y1, x2, y2 = map(int, box)
#                 label = f"{yolo_model.names[int(cls)]} {conf:.2f}"
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#                 cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

#         frame_count += 1

#         # Stream to browser
#         _, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# # Routes
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/video')
# def video():
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# # Start app
# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import torch
from PIL import Image
import json
import time

app = Flask(__name__)

# Load models
emotion_model = load_model("emotion_model.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

camera = cv2.VideoCapture(0)

# Stream real-time frames
def gen_frames():
    frame_count = 0
    yolo_results = None
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
            roi = roi.astype("float32") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            preds = emotion_model.predict(roi, verbose=0)[0]
            label = emotion_labels[np.argmax(preds)]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        if frame_count % 5 == 0:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            yolo_results = yolo_model(pil_img, size=640)

        if yolo_results is not None:
            for *box, conf, cls in yolo_results.xyxy[0]:
                x1, y1, x2, y2 = map(int, box)
                label = f"{yolo_model.names[int(cls)]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        frame_count += 1

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def capture_for_30_seconds():
    detected_emotions = set()
    detected_objects = set()

    start_time = time.time()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while time.time() - start_time < 30:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48))
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)
            roi = roi.astype("float32") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            preds = emotion_model.predict(roi, verbose=0)[0]
            label = emotion_labels[np.argmax(preds)]
            detected_emotions.add(label)

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        yolo_results = yolo_model(pil_img, size=640)

        if yolo_results is not None:
            for *box, conf, cls in yolo_results.xyxy[0]:
                label = yolo_model.names[int(cls)]
                detected_objects.add(label)

    # Final result (only unique values)
    results = {
        "emotions": list(detected_emotions),
        "objects": list(detected_objects)
    }

    # Save to JSON file only
    with open("static/detection_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture')
def capture():
    results = capture_for_30_seconds()
    return jsonify({
        "status": "success",
        "message": "Capture completed!",
        "emotions": results["emotions"],
        "objects": results["objects"]
    })

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
