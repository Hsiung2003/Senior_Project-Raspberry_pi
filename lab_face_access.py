import argparse
import os
import signal
import sys
from threading import Condition, Lock, Thread
from unittest.mock import MagicMock

import cv2
from flask import Flask, Response, jsonify, render_template_string
from picamera2 import Picamera2

sys.modules["pkg_resources"] = MagicMock()
import face_recognition


class FaceRecognitionCamera:
    def __init__(self, width: int = 640, height: int = 480, known_image_path: str = "test.jpg"):
        self.picam2 = Picamera2()
        self.frame_condition = Condition()
        self.lock = Lock()
        self.running = False
        self.thread = None
        self.camera_started = False

        self.latest_jpeg = None
        self.latest_access_message = "No face detected"
        self.latest_access_level = "idle"

        self.face_locations = []
        self.face_names = []
        self.process_this_frame = 0

        self.known_face_encodings = []
        self.known_face_names = []

        if os.path.exists(known_image_path):
            img = face_recognition.load_image_file(known_image_path)
            encodings = face_recognition.face_encodings(img)
            if encodings:
                self.known_face_encodings.append(encodings[0])
                self.known_face_names.append("Winnie")
            else:
                print(f"No face found in {known_image_path}.")
        else:
            print(f"Known face image not found: {known_image_path}")

        config = self.picam2.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls={"Sharpness": 1.0},
        )
        self.picam2.configure(config)

    def start(self):
        with self.lock:
            if self.running:
                return
            self.running = True

        self.picam2.start()
        self.camera_started = True
        self.thread = Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print("Camera thread started.")

    def stop(self):
        with self.lock:
            was_started = self.camera_started
            self.running = False
            self.camera_started = False

        if self.thread is not None:
            self.thread.join(timeout=2.0)
            self.thread = None

        if was_started:
            self.picam2.stop()
            print("Camera stopped.")

    def _capture_loop(self):
        while True:
            with self.lock:
                if not self.running:
                    break

            rgb_frame = self.picam2.capture_array()

            if self.process_this_frame % 3 == 0:
                small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)
                self.face_locations = face_recognition.face_locations(small_frame)
                face_encodings = face_recognition.face_encodings(small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings, face_encoding, tolerance=0.5
                    )
                    name = "Unknown"
                    if True in matches:
                        name = self.known_face_names[matches.index(True)]
                    self.face_names.append(name)

                if not self.face_names:
                    self.latest_access_message = "No face detected"
                    self.latest_access_level = "idle"
                elif all(name == "Winnie" for name in self.face_names):
                    self.latest_access_message = "Welcome"
                    self.latest_access_level = "allow"
                else:
                    self.latest_access_message = "Denied"
                    self.latest_access_level = "deny"

            self.process_this_frame += 1

            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2

                if name == "Winnie":
                    color = (0, 255, 0)
                    status_label = "Winnie: ACCESS GRANTED"
                else:
                    color = (0, 0, 255)
                    status_label = "UNKNOWN: ACCESS DENIED"

                cv2.rectangle(bgr_frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(bgr_frame, (left, top - 30), (right, top), color, cv2.FILLED)
                cv2.putText(
                    bgr_frame,
                    status_label,
                    (left + 5, top - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

            ok, jpeg = cv2.imencode(".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                continue

            with self.frame_condition:
                self.latest_jpeg = jpeg.tobytes()
                self.frame_condition.notify_all()

    def mjpeg_generator(self):
        while True:
            with self.lock:
                if not self.running:
                    break

            with self.frame_condition:
                if self.latest_jpeg is None:
                    self.frame_condition.wait(timeout=1.0)
                frame = self.latest_jpeg

            if frame is None:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )


def create_app(camera: FaceRecognitionCamera):
    app = Flask(__name__)
    template = """
    <!doctype html>
    <html>
      <head>
        <title>ESOE Lab Access</title>
        <style>
          body {
            background: #1a1a1a;
            color: white;
            text-align: center;
            font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
          }
          .status-bar {
            padding: 10px;
            margin-bottom: 20px;
            background: #333;
            font-weight: bold;
          }
          img {
            border: 4px solid #444;
            box-shadow: 0 0 20px rgba(0,0,0,0.5);
            width: 85%;
            max-width: 900px;
          }
          #access-status {
            margin-top: 16px;
            font-size: 2rem;
            font-weight: bold;
          }
          #access-status.allow { color: #00e676; }
          #access-status.deny { color: #ff5252; }
          #access-status.idle { color: #ffd54f; }
        </style>
      </head>
      <body>
        <div class="status-bar">LIVE: SMART LAB MONITORING SYSTEM</div>
        <img src="/video_stream" alt="Live camera stream" />
        <div id="access-status" class="idle">No face detected</div>
        <p>System Ver 1.0 - National Taiwan University</p>
        <script>
          async function refreshStatus() {
            try {
              const resp = await fetch("/status");
              const data = await resp.json();
              const node = document.getElementById("access-status");
              node.textContent = data.message;
              node.className = data.level;
            } catch (e) {
              const node = document.getElementById("access-status");
              node.textContent = "Status unavailable";
              node.className = "deny";
            }
          }
          setInterval(refreshStatus, 500);
          refreshStatus();
        </script>
      </body>
    </html>
    """

    @app.route("/", methods=["GET"])
    def index():
        return render_template_string(template)

    @app.route("/video_stream", methods=["GET"])
    def video_stream():
        return Response(
            camera.mjpeg_generator(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/status", methods=["GET"])
    def status():
        return jsonify(
            {
                "message": camera.latest_access_message,
                "level": camera.latest_access_level,
            }
        )

    return app


def main():
    parser = argparse.ArgumentParser(description="Lab Face Recognition System")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--known-image", default="test.jpg", help="Path to authorized face image")
    args = parser.parse_args()

    camera = FaceRecognitionCamera(width=640, height=480, known_image_path=args.known_image)
    app = create_app(camera)

    def shutdown_handler(signum, frame):
        camera.stop()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    camera.start()
    try:
        app.run(host=args.host, port=args.port, threaded=True)
    finally:
        camera.stop()


if __name__ == "__main__":
    main()
