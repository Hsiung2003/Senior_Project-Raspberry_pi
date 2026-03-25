import argparse
import os
import signal
import time
from threading import Condition, Lock, Thread

import cv2
from flask import Flask, Response, render_template_string
from libcamera import Transform, controls
from picamera2 import Picamera2


def load_face_detector():
    cascade_name = "haarcascade_frontalface_default.xml"
    candidate_files = []

    # Newer OpenCV builds expose cv2.data.haarcascades
    cv2_data = getattr(cv2, "data", None)
    haar_dir = getattr(cv2_data, "haarcascades", None) if cv2_data else None
    if haar_dir:
        candidate_files.append(os.path.join(haar_dir, cascade_name))

    # Common Linux / Raspberry Pi OpenCV cascade locations
    candidate_files.extend(
        [
            f"/usr/share/opencv4/haarcascades/{cascade_name}",
            f"/usr/share/opencv/haarcascades/{cascade_name}",
            f"/usr/local/share/opencv4/haarcascades/{cascade_name}",
            f"/usr/local/share/opencv/haarcascades/{cascade_name}",
        ]
    )

    # Optional override if environment variable is provided
    custom_dir = os.environ.get("OPENCV_HAAR_DIR")
    if custom_dir:
        candidate_files.append(os.path.join(custom_dir, cascade_name))

    checked = []
    for cascade_path in candidate_files:
        if cascade_path in checked:
            continue
        checked.append(cascade_path)
        if not os.path.isfile(cascade_path):
            continue
        detector = cv2.CascadeClassifier(cascade_path)
        if not detector.empty():
            return detector, cascade_path

    # Last chance: let OpenCV resolve by filename if packaged internally
    detector = cv2.CascadeClassifier(cascade_name)
    if not detector.empty():
        return detector, cascade_name

    raise RuntimeError(
        "Failed to load Haar cascade for face detection. "
        f"Checked paths: {checked}"
    )


class FaceDetectionCamera:
    def __init__(self, width=1280, height=720, vflip=True):
        self.picam2 = Picamera2()
        self.frame_condition = Condition()
        self.latest_jpeg = None
        self.running = False
        self.thread = None
        self.lock = Lock()
        self.camera_started = False

        self.face_detector, self.face_cascade_path = load_face_detector()

        config = self.picam2.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            transform=Transform(vflip=1 if vflip else 0),
            controls={
                "NoiseReductionMode": controls.draft.NoiseReductionModeEnum.HighQuality,
                "Sharpness": 1.5,
            },
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

    def _capture_loop(self):
        while True:
            with self.lock:
                if not self.running:
                    break

            rgb_frame = self.picam2.capture_array()
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)

            faces = self.face_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(bgr_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            ok, jpeg = cv2.imencode(".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                continue

            with self.frame_condition:
                self.latest_jpeg = jpeg.tobytes()
                self.frame_condition.notify_all()

            time.sleep(0.01)

    def mjpeg_generator(self):
        while True:
            with self.frame_condition:
                self.frame_condition.wait(timeout=1.0)
                frame = self.latest_jpeg

            if frame is None:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )


def create_app(camera: FaceDetectionCamera):
    app = Flask(__name__)

    template = """
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8">
        <title>Raspberry Pi Face Detection Stream</title>
      </head>
      <body>
        <h2>OpenCV Face Detection (Raspberry Pi)</h2>
        <img src="{{ url_for('video_stream') }}" width="100%">
      </body>
    </html>
    """

    @app.route("/", methods=["GET"])
    def index():
        return render_template_string(template)

    @app.route("/api/stream", methods=["GET"])
    def video_stream():
        return Response(
            camera.mjpeg_generator(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    return app


def main():
    parser = argparse.ArgumentParser(
        description="Raspberry Pi real-time face detection and MJPEG stream"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--width", default=1280, type=int)
    parser.add_argument("--height", default=720, type=int)
    parser.add_argument("--no-vflip", action="store_true")
    args = parser.parse_args()

    camera = FaceDetectionCamera(
        width=args.width, height=args.height, vflip=not args.no_vflip
    )
    app = create_app(camera)

    def shutdown_handler(signum, frame):  # noqa: ARG001
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
