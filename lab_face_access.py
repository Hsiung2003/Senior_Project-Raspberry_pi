import argparse
import os
import signal
import sys
from threading import Condition, Lock, Thread
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import cv2
from flask import Flask, Response, jsonify, render_template_string
from picamera2 import Picamera2

sys.modules["pkg_resources"] = MagicMock()
import face_recognition


class FaceRecognitionCamera:
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        known_image_path: str = "test.jpg",
        known_faces: Optional[Dict[str, List[str]]] = None,
        target_fps: int = 30,
    ):
        self.picam2 = Picamera2()
        self.frame_condition = Condition()
        self.recognition_condition = Condition()
        self.lock = Lock()
        self.running = False
        self.capture_thread = None
        self.recognition_thread = None
        self.camera_started = False

        self.latest_jpeg = None
        self.latest_access_message = "No face detected"
        self.latest_access_level = "idle"

        self.pending_recognition_frame = None
        self.detections = []
        self.process_this_frame = 0
        self.target_fps = target_fps
        self.recognition_interval = 4
        self.recognition_scale = 0.25
        self.jpeg_quality = 65

        self.known_face_encodings = []
        self.known_face_names = []

        if known_faces is None:
            known_faces = {"Winnie": [known_image_path]}
        self._load_known_faces(known_faces)
        print(f"[INFO] Loaded {len(self.known_face_encodings)} known face encodings.")

        frame_time_us = max(8333, int(1_000_000 / max(1, self.target_fps)))

        config = self.picam2.create_video_configuration(
            main={"size": (width, height), "format": "BGR888"},
            controls={
                "Sharpness": 1.0,
                "FrameDurationLimits": (frame_time_us, frame_time_us),
            },
        )
        self.picam2.configure(config)

    def _load_known_faces(self, known_faces: Dict[str, List[str]]):
        for person_name, image_paths in known_faces.items():
            for image_path in image_paths:
                if not os.path.exists(image_path):
                    print(f"[WARN] File not found: {image_path}")
                    continue

                img = face_recognition.load_image_file(image_path)
                encodings = face_recognition.face_encodings(img)
                if not encodings:
                    print(f"[WARN] No face found in {image_path}")
                    continue

                self.known_face_encodings.append(encodings[0])
                self.known_face_names.append(person_name)

    def start(self):
        with self.lock:
            if self.running:
                return
            self.running = True

        self.picam2.start()
        self.picam2.set_controls({"AeEnable": True, "AwbEnable": True})
        self.camera_started = True

        self.capture_thread = Thread(target=self._capture_loop, daemon=True)
        self.recognition_thread = Thread(target=self._recognition_loop, daemon=True)
        self.capture_thread.start()
        self.recognition_thread.start()
        print("Camera threads started.")

    def stop(self):
        with self.lock:
            was_started = self.camera_started
            self.running = False
            self.camera_started = False

        with self.frame_condition:
            self.frame_condition.notify_all()
        with self.recognition_condition:
            self.recognition_condition.notify_all()

        if self.capture_thread is not None:
            self.capture_thread.join(timeout=2.0)
            self.capture_thread = None
        if self.recognition_thread is not None:
            self.recognition_thread.join(timeout=2.0)
            self.recognition_thread = None

        if was_started:
            self.picam2.stop()
            print("Camera stopped.")

    def _capture_loop(self):
        while True:
            with self.lock:
                if not self.running:
                    break

            bgr_frame = self.picam2.capture_array("main")
            self.process_this_frame += 1

            if self.process_this_frame % self.recognition_interval == 0:
                with self.lock:
                    self.pending_recognition_frame = bgr_frame.copy()
                with self.recognition_condition:
                    self.recognition_condition.notify()

            with self.lock:
                detections = list(self.detections)
            self._draw_detections(bgr_frame, detections)

            ok, jpeg = cv2.imencode(".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if not ok:
                continue

            with self.frame_condition:
                self.latest_jpeg = jpeg.tobytes()
                self.frame_condition.notify_all()

    def _recognition_loop(self):
        while True:
            with self.recognition_condition:
                self.recognition_condition.wait(timeout=0.1)

            with self.lock:
                if not self.running and self.pending_recognition_frame is None:
                    break
                frame_for_recognition = self.pending_recognition_frame
                self.pending_recognition_frame = None

            if frame_for_recognition is None:
                continue

            small_bgr = cv2.resize(
                frame_for_recognition,
                (0, 0),
                fx=self.recognition_scale,
                fy=self.recognition_scale,
                interpolation=cv2.INTER_LINEAR,
            )
            small_rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(small_rgb, model="hog")
            face_encodings = face_recognition.face_encodings(small_rgb, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, face_encoding, tolerance=0.5
                )
                name = "Unknown"
                if True in matches:
                    name = self.known_face_names[matches.index(True)]
                face_names.append(name)

            scale_restore = int(round(1 / self.recognition_scale))
            detections: List[Tuple[int, int, int, int, str]] = []
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                detections.append(
                    (
                        top * scale_restore,
                        right * scale_restore,
                        bottom * scale_restore,
                        left * scale_restore,
                        name,
                    )
                )

            message, level = self._resolve_access_status(face_names)
            with self.lock:
                self.detections = detections
                self.latest_access_message = message
                self.latest_access_level = level

    @staticmethod
    def _draw_detections(bgr_frame, detections: List[Tuple[int, int, int, int, str]]):
        for top, right, bottom, left, name in detections:
            if name != "Unknown":
                color = (0, 255, 0)
                status_label = f"{name}: ACCESS GRANTED"
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

    @staticmethod
    def _resolve_access_status(face_names):
        if not face_names:
            return "No face detected", "idle"
        if any(name != "Unknown" for name in face_names):
            return "Welcome", "allow"
        return "Denied", "deny"

    def get_status(self):
        with self.lock:
            return self.latest_access_message, self.latest_access_level

    def mjpeg_generator(self):
        while True:
            with self.lock:
                if not self.running:
                    break

            with self.frame_condition:
                if self.latest_jpeg is None:
                    self.frame_condition.wait(timeout=0.2)
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
              const resp = await fetch("/status", { cache: "no-store" });
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
          setInterval(refreshStatus, 250);
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
        resp = Response(
            camera.mjpeg_generator(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["X-Accel-Buffering"] = "no"
        return resp

    @app.route("/status", methods=["GET"])
    def status():
        message, level = camera.get_status()
        resp = jsonify(
            {
                "message": message,
                "level": level,
            }
        )
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        resp.headers["Pragma"] = "no-cache"
        return resp

    return app


def main():
    parser = argparse.ArgumentParser(description="Lab Face Recognition System")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--known-image", default="test.jpg", help="Path to authorized face image")
    parser.add_argument("--target-fps", default=30, type=int, help="Target camera FPS")
    parser.add_argument(
        "--recognition-interval",
        default=4,
        type=int,
        help="Run face recognition once every N frames",
    )
    parser.add_argument(
        "--jpeg-quality",
        default=65,
        type=int,
        help="MJPEG quality (lower is faster)",
    )
    args = parser.parse_args()

    camera = FaceRecognitionCamera(
        width=640,
        height=480,
        known_image_path=args.known_image,
        target_fps=args.target_fps,
    )
    camera.recognition_interval = max(1, args.recognition_interval)
    camera.jpeg_quality = max(40, min(90, args.jpeg_quality))
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
