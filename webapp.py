import base64
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import cv2
import mediapipe as mp
import numpy as np

from openface import (
    CALIBRATION_CAPTURE_FRAMES,
    FACE_TRACKING_GRACE_FRAMES,
    HAND_MODEL_URL,
    LETTERS,
    MIN_SAMPLES_PER_LETTER,
    SignTextAccumulator,
    classify_gesture,
    detect_face,
    draw_hand,
    get_model_path,
    get_templates_path,
    get_windows_compatible_path,
    gesture_features,
    load_templates,
    resample_sequence,
    save_templates,
)


ROOT = Path(__file__).resolve().parent
MAX_REQUEST_SIZE = 4 * 1024 * 1024
SESSION_TTL_SECONDS = 30 * 60
MAX_ACTIVE_SESSIONS = 4
ALLOWED_ORIGINS = {
    origin.strip().rstrip("/")
    for origin in os.environ.get(
        "CORS_ORIGINS",
        "http://127.0.0.1:8000,http://localhost:8000",
    ).split(",")
    if origin.strip()
}


class RequestError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


class SignRecognizer:
    def __init__(self, session_id):
        self.lock = threading.RLock()
        templates_path = get_templates_path()
        self.templates_path = (
            templates_path.parent / "web-sessions" / session_id / templates_path.name
        )
        self.templates, self.template_labels = load_templates(self.templates_path)
        face_model_path = ROOT / "haarcascade_frontalface_default.xml"
        compatible_face_model_path = get_windows_compatible_path(face_model_path)
        self.face_detector = cv2.CascadeClassifier(str(compatible_face_model_path))
        if self.face_detector.empty():
            raise RuntimeError(f"Не удалось загрузить модель распознавания лица: {face_model_path}")

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(get_model_path("hand_landmarker.task", HAND_MODEL_URL))
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.65,
            min_hand_presence_confidence=0.65,
            min_tracking_confidence=0.65,
        )
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self.calibration_mode = False
        self.calibration_index = 0
        self.recording_sequence = None
        self.sequence_buffer = []
        self.recognized_text = ""
        self.detected_letter = None
        self.hand_detected = False
        self.last_feature_vector = None
        self.last_face_box = None
        self.missed_face_frames = 0
        self.last_timestamp_ms = 0
        self.text_accumulator = SignTextAccumulator()
        self.last_used = time.monotonic()

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = max(time.monotonic_ns() // 1_000_000, self.last_timestamp_ms + 1)
        self.last_timestamp_ms = timestamp_ms
        results = self.landmarker.detect_for_video(image, timestamp_ms)
        hand_landmarks = results.hand_landmarks[0] if results.hand_landmarks else None
        face_box = detect_face(frame, self.face_detector)
        if face_box is not None:
            self.last_face_box = face_box
            self.missed_face_frames = 0
        elif (
            self.last_face_box is not None
            and self.missed_face_frames < FACE_TRACKING_GRACE_FRAMES
        ):
            face_box = self.last_face_box
            self.missed_face_frames += 1
        else:
            self.last_face_box = None

        self.detected_letter = None
        self.hand_detected = hand_landmarks is not None
        self.last_feature_vector = None
        if face_box is not None:
            face_x, face_y, face_width, face_height = face_box
            cv2.rectangle(
                frame,
                (face_x, face_y),
                (face_x + face_width, face_y + face_height),
                (80, 180, 255),
                2,
            )

        if hand_landmarks:
            draw_hand(frame, hand_landmarks)
            self.last_feature_vector = gesture_features(
                hand_landmarks,
                frame.shape,
                face_box,
            )

        if self.calibration_mode:
            if self.recording_sequence is not None and self.last_feature_vector is not None:
                self.recording_sequence.append(self.last_feature_vector.copy())
                if len(self.recording_sequence) >= CALIBRATION_CAPTURE_FRAMES:
                    self.templates = np.concatenate(
                        (
                            self.templates,
                            resample_sequence(self.recording_sequence)[None, :, :],
                        ),
                        axis=0,
                    )
                    self.template_labels = np.concatenate(
                        (
                            self.template_labels,
                            np.asarray([LETTERS[self.calibration_index]], dtype="U1"),
                        )
                    )
                    save_templates(
                        self.templates_path,
                        self.templates,
                        self.template_labels,
                    )
                    self.recording_sequence = None
        elif self.last_feature_vector is not None:
            self.sequence_buffer.append(self.last_feature_vector)
            if len(self.sequence_buffer) > CALIBRATION_CAPTURE_FRAMES:
                del self.sequence_buffer[0]
            if len(self.sequence_buffer) == CALIBRATION_CAPTURE_FRAMES and len(self.templates):
                self.detected_letter, _ = classify_gesture(
                    resample_sequence(self.sequence_buffer),
                    self.templates,
                    self.template_labels,
                )
        else:
            self.sequence_buffer.clear()

        if not self.calibration_mode and len(self.templates):
            added_letter = self.text_accumulator.update(self.detected_letter)
            if added_letter:
                self.recognized_text += added_letter

        return frame

    def status(self):
        letter = LETTERS[self.calibration_index] if self.calibration_mode else None
        sample_count = (
            int(np.count_nonzero(self.template_labels == letter)) if letter else 0
        )
        completed_letters = sum(
            np.count_nonzero(self.template_labels == item) >= MIN_SAMPLES_PER_LETTER
            for item in LETTERS
        )
        return {
            "templates_count": len(self.templates),
            "detected_letter": self.detected_letter,
            "recognized_text": self.recognized_text,
            "calibration_mode": self.calibration_mode,
            "calibration_letter": letter,
            "sample_count": sample_count,
            "recording_frames": (
                len(self.recording_sequence)
                if self.recording_sequence is not None
                else 0
            ),
            "completed_letters": int(completed_letters),
            "letters_total": len(LETTERS),
            "hand_detected": self.hand_detected,
        }

    def action(self, name):
        if name == "start_calibration":
            counts = {
                letter: np.count_nonzero(self.template_labels == letter)
                for letter in LETTERS
            }
            self.calibration_index = next(
                (
                    index
                    for index, letter in enumerate(LETTERS)
                    if counts[letter] < MIN_SAMPLES_PER_LETTER
                ),
                0,
            )
            self.calibration_mode = True
            self.recording_sequence = None
            self.sequence_buffer.clear()
            self.text_accumulator.reset()
        elif name == "cancel_calibration":
            self.calibration_mode = False
            self.recording_sequence = None
            self.sequence_buffer.clear()
            self.text_accumulator.reset()
        elif name == "finish_calibration":
            if not self.calibration_mode:
                raise RequestError("Сейчас настройка не выполняется.", 409)
            self.calibration_mode = False
            self.recording_sequence = None
            self.sequence_buffer.clear()
            self.text_accumulator.reset()
        elif name == "record":
            if not self.calibration_mode:
                raise RequestError("Сначала начните настройку.")
            if self.recording_sequence is not None:
                raise RequestError("Запись жеста уже выполняется.", 409)
            if self.last_feature_vector is None:
                raise RequestError("Сначала покажите руку в камеру.", 409)
            self.recording_sequence = []
        elif name in ("previous_letter", "next_letter"):
            if not self.calibration_mode:
                raise RequestError("Сначала начните настройку.")
            if self.recording_sequence is not None:
                raise RequestError("Дождитесь окончания записи жеста.", 409)
            direction = -1 if name == "previous_letter" else 1
            self.calibration_index = (self.calibration_index + direction) % len(LETTERS)
        elif name == "add_space":
            if not self.calibration_mode and self.recognized_text and not self.recognized_text.endswith(" "):
                self.recognized_text += " "
        elif name == "backspace":
            if not self.calibration_mode:
                self.recognized_text = self.recognized_text[:-1]
        elif name == "clear_text":
            if not self.calibration_mode:
                self.recognized_text = ""
        else:
            raise RequestError("Неизвестная команда.")


recognizers = {}
recognizer_lock = threading.Lock()


def parse_session_id(value):
    if not isinstance(value, str):
        raise RequestError("В запросе отсутствует идентификатор браузера.")
    try:
        return str(uuid.UUID(value))
    except ValueError as error:
        raise RequestError("Некорректный идентификатор браузера.") from error


def get_recognizer(session_id):
    now = time.monotonic()
    with recognizer_lock:
        expired_sessions = [
            key
            for key, app in recognizers.items()
            if now - app.last_used >= SESSION_TTL_SECONDS
        ]
        for key in expired_sessions:
            app = recognizers.pop(key)
            with app.lock:
                app.landmarker.close()

        app = recognizers.get(session_id)
        if app is None:
            if len(recognizers) >= MAX_ACTIVE_SESSIONS:
                raise RequestError(
                    "Сервер занят. Попробуйте открыть сайт позже.",
                    503,
                )
            app = SignRecognizer(session_id)
            recognizers[session_id] = app
        app.last_used = now
        return app


class WebHandler(BaseHTTPRequestHandler):
    def _send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        origin = self.headers.get("Origin")
        if origin and origin.rstrip("/") in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Vary", "Origin")
        self.end_headers()
        self.wfile.write(body)

    def _origin_is_allowed(self):
        origin = self.headers.get("Origin")
        return not origin or origin.rstrip("/") in ALLOWED_ORIGINS

    def _read_json(self):
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise RequestError("Некорректный размер запроса.") from error
        if content_length <= 0 or content_length > MAX_REQUEST_SIZE:
            raise RequestError("Пустой запрос или слишком большой кадр.")
        try:
            payload = json.loads(self.rfile.read(content_length))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise RequestError("Некорректный JSON.") from error
        if not isinstance(payload, dict):
            raise RequestError("Ожидался JSON-объект.")
        return payload

    def do_GET(self):
        if not self._origin_is_allowed():
            self._send_json({"error": "Источник запроса не разрешён."}, 403)
            return
        if self.path == "/":
            page = (ROOT / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)
            return
        parsed_path = urlsplit(self.path)
        if parsed_path.path == "/api/status":
            try:
                query = parse_qs(parsed_path.query)
                session_id = parse_session_id(
                    query.get("session_id", [None])[0]
                )
                app = get_recognizer(session_id)
                with app.lock:
                    self._send_json(app.status())
            except RequestError as error:
                self._send_json({"error": str(error)}, error.status)
            except Exception as error:
                self._send_json({"error": str(error)}, 500)
            return
        self._send_json({"error": "Страница не найдена."}, 404)

    def do_OPTIONS(self):
        if not self._origin_is_allowed():
            self._send_json({"error": "Источник запроса не разрешён."}, 403)
            return
        self.send_response(204)
        origin = self.headers.get("Origin")
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Max-Age", "600")
            self.send_header("Vary", "Origin")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        if not self._origin_is_allowed():
            self._send_json({"error": "Источник запроса не разрешён."}, 403)
            return
        try:
            payload = self._read_json()
            session_id = parse_session_id(payload.get("session_id"))
            app = get_recognizer(session_id)
            with app.lock:
                if self.path == "/api/frame":
                    encoded_frame = payload.get("image")
                    if not isinstance(encoded_frame, str):
                        raise RequestError("В запросе отсутствует изображение.")
                    try:
                        frame_bytes = base64.b64decode(encoded_frame, validate=True)
                    except (ValueError, base64.binascii.Error) as error:
                        raise RequestError("Не удалось прочитать изображение.") from error
                    frame = cv2.imdecode(
                        np.frombuffer(frame_bytes, dtype=np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                    if frame is None:
                        raise RequestError("Получен некорректный кадр камеры.")
                    processed_frame = app.process_frame(frame)
                    success, jpeg = cv2.imencode(
                        ".jpg",
                        processed_frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 78],
                    )
                    if not success:
                        raise RuntimeError("Не удалось подготовить кадр для браузера.")
                    response = {
                        "image": base64.b64encode(jpeg).decode("ascii"),
                        "status": app.status(),
                    }
                    self._send_json(response)
                elif self.path == "/api/action":
                    name = payload.get("action")
                    if not isinstance(name, str):
                        raise RequestError("В запросе отсутствует команда.")
                    app.action(name)
                    self._send_json(app.status())
                else:
                    self._send_json({"error": "Страница не найдена."}, 404)
        except RequestError as error:
            self._send_json({"error": str(error)}, error.status)
        except Exception as error:
            self._send_json({"error": str(error)}, 500)

    def log_message(self, format_string, *args):
        print(f"[web] {self.address_string()} - {format_string % args}")


def main():
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer((host, port), WebHandler)
    print(f"Сайт запущен: http://{host}:{port}")
    print("Для остановки нажмите Ctrl+C.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nСервер остановлен.")
    finally:
        server.server_close()
        for app in recognizers.values():
            with app.lock:
                app.landmarker.close()
        recognizers.clear()


if __name__ == "__main__":
    main()
