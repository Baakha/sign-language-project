import ctypes
import os
import shutil
import time
import urllib.request
from functools import lru_cache
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont


HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
LETTERS = tuple("АӘБВГҒДЕЁЖЗИЙКҚЛМНҢОӨПРСТУҰҮФХҺЦЧШЩЪЫІЬЭЮЯ")
MIN_SAMPLES_PER_LETTER = 5
HAND_FEATURE_COUNT = 63
POSITION_FEATURE_COUNT = 5
FEATURE_SIZE = HAND_FEATURE_COUNT + POSITION_FEATURE_COUNT
SEQUENCE_FRAMES = 20
CALIBRATION_CAPTURE_FRAMES = 40
NEAREST_TEMPLATES = 3
MAX_MATCH_DISTANCE = 0.65
MIN_MATCH_MARGIN = 0.08
STABLE_LETTER_FRAMES = 4
RELEASE_FRAMES = 6
FACE_TRACKING_GRACE_FRAMES = 15
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


def get_windows_compatible_path(path):
    if os.name != "nt":
        return path

    get_short_path_name = ctypes.WinDLL("kernel32").GetShortPathNameW
    get_short_path_name.argtypes = (
        ctypes.c_wchar_p,
        ctypes.POINTER(ctypes.c_wchar),
        ctypes.c_uint32,
    )
    get_short_path_name.restype = ctypes.c_uint32
    short_path_buffer = ctypes.create_unicode_buffer(32768)
    if get_short_path_name(str(path), short_path_buffer, len(short_path_buffer)):
        return Path(short_path_buffer.value)
    return path


def get_model_path(filename, model_url):
    app_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    model_path = app_data / "KazakhSignAlphabet" / filename
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        download_path = model_path.with_suffix(".task.download")
        try:
            with urllib.request.urlopen(model_url, timeout=60) as response:
                with download_path.open("wb") as model_file:
                    shutil.copyfileobj(response, model_file)
            os.replace(download_path, model_path)
        except OSError as error:
            download_path.unlink(missing_ok=True)
            raise RuntimeError(
                "Не удалось загрузить модель кисти MediaPipe. "
                "Проверьте интернет-соединение."
            ) from error

    return get_windows_compatible_path(model_path)


def get_templates_path():
    app_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    return app_data / "KazakhSignAlphabet" / "gesture_sequences.npz"


def hand_features(landmarks):
    points = np.array(
        [(point.x, point.y, point.z) for point in landmarks],
        dtype=np.float32,
    )
    points -= points[0]
    scale = float(np.linalg.norm(points[9]))
    if scale < 1e-6:
        raise ValueError("Не удалось нормализовать точки кисти.")
    points /= scale
    return points.reshape(-1)


def detect_face(frame, face_detector):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray_frame.shape
    scale = min(1.0, 320 / width)
    if scale < 1.0:
        gray_frame = cv2.resize(
            gray_frame,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    faces = face_detector.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(max(1, round(60 * scale)), max(1, round(60 * scale))),
    )
    if len(faces) == 0:
        return None
    face = max(faces, key=lambda item: item[2] * item[3])
    if scale < 1.0:
        return tuple(round(value / scale) for value in face)
    return face


def gesture_features(landmarks, frame_shape, face_box):
    height, width = frame_shape[:2]
    wrist_x = float(landmarks[0].x)
    wrist_y = float(landmarks[0].y)
    hand_center_x = float(np.mean([point.x for point in landmarks]))
    hand_center_y = float(np.mean([point.y for point in landmarks]))
    if face_box is None:
        face_relative_x = hand_center_x
        face_relative_y = hand_center_y
        face_detected = 0.0
    else:
        face_x, face_y, face_width, face_height = face_box
        face_relative_x = (
            hand_center_x * width - (face_x + face_width / 2)
        ) / max(face_width, 1)
        face_relative_y = (
            hand_center_y * height - (face_y + face_height / 2)
        ) / max(face_height, 1)
        face_detected = 1.0

    position_features = np.array(
        (
            wrist_x * 2,
            wrist_y * 2,
            face_relative_x * 0.5,
            face_relative_y * 0.5,
            face_detected,
        ),
        dtype=np.float32,
    )
    return np.concatenate((hand_features(landmarks), position_features))


def resample_sequence(sequence):
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 2 or sequence.shape[1] != FEATURE_SIZE or len(sequence) < 2:
        raise ValueError("Последовательность жеста имеет неверный формат.")

    source_positions = np.linspace(0, len(sequence) - 1, num=len(sequence))
    target_positions = np.linspace(0, len(sequence) - 1, num=SEQUENCE_FRAMES)
    sampled = np.empty((SEQUENCE_FRAMES, FEATURE_SIZE), dtype=np.float32)
    for feature_index in range(FEATURE_SIZE):
        sampled[:, feature_index] = np.interp(
            target_positions,
            source_positions,
            sequence[:, feature_index],
        )
    return sampled


def load_templates(path):
    if not path.exists():
        return (
            np.empty((0, SEQUENCE_FRAMES, FEATURE_SIZE), dtype=np.float32),
            np.empty((0,), dtype="U1"),
        )

    with np.load(path, allow_pickle=False) as data:
        features = np.asarray(data["features"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype="U1")

    if (
        features.ndim != 3
        or features.shape[1:] != (SEQUENCE_FRAMES, FEATURE_SIZE)
    ):
        raise RuntimeError(f"Файл шаблонов имеет неверный формат: {path}")
    if labels.ndim != 1 or len(labels) != len(features):
        raise RuntimeError(f"Метки в файле шаблонов повреждены: {path}")
    if not np.isfinite(features).all():
        raise RuntimeError(f"Файл шаблонов содержит некорректные данные: {path}")
    if not set(labels).issubset(set(LETTERS)):
        raise RuntimeError(f"Файл содержит неизвестные метки букв: {path}")
    return features, labels


def save_templates(path, features, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary_path,
        features=np.asarray(features, dtype=np.float32),
        labels=np.asarray(labels, dtype="U1"),
    )
    os.replace(temporary_path, path)


def classify_gesture(sequence, templates, labels):
    if len(templates) == 0:
        return None, 0.0

    hand_distances = np.linalg.norm(
        templates[:, :, :HAND_FEATURE_COUNT]
        - sequence[None, :, :HAND_FEATURE_COUNT],
        axis=2,
    ) / np.sqrt(21)
    position_distances = np.linalg.norm(
        templates[:, :, HAND_FEATURE_COUNT:]
        - sequence[None, :, HAND_FEATURE_COUNT:],
        axis=2,
    ) / np.sqrt(POSITION_FEATURE_COUNT)
    distances = (0.8 * hand_distances + 0.2 * position_distances).mean(axis=1)
    class_distances = {
        str(label): float(
            np.sort(distances[labels == label])[:NEAREST_TEMPLATES].mean()
        )
        for label in np.unique(labels)
    }
    ranked = sorted(class_distances.items(), key=lambda item: item[1])
    best_label, best_distance = ranked[0]
    second_distance = ranked[1][1] if len(ranked) > 1 else float("inf")
    margin = second_distance - best_distance
    if best_distance > MAX_MATCH_DISTANCE or (
        len(ranked) > 1 and margin < MIN_MATCH_MARGIN
    ):
        return None, best_distance
    return best_label, best_distance


class SignTextAccumulator:
    def __init__(self):
        self.candidate = None
        self.candidate_frames = 0
        self.last_added = None
        self.release_frames = 0

    def reset(self):
        self.candidate = None
        self.candidate_frames = 0
        self.last_added = None
        self.release_frames = 0

    def update(self, letter):
        if letter is None:
            self.candidate = None
            self.candidate_frames = 0
            self.release_frames += 1
            if self.release_frames >= RELEASE_FRAMES:
                self.last_added = None
            return None

        self.release_frames = 0
        if letter == self.candidate:
            self.candidate_frames += 1
        else:
            self.candidate = letter
            self.candidate_frames = 1

        if (
            self.candidate_frames >= STABLE_LETTER_FRAMES
            and letter != self.last_added
        ):
            self.last_added = letter
            return letter
        return None


def draw_hand(frame, landmarks):
    height, width = frame.shape[:2]
    points = [
        (int(point.x * width), int(point.y * height))
        for point in landmarks
    ]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (0, 220, 100), 2)
    for point in points:
        cv2.circle(frame, point, 4, (30, 80, 255), -1)


@lru_cache(maxsize=8)
def load_font(size):
    fonts_dir = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    for filename in ("segoeui.ttf", "arial.ttf"):
        font_path = os.path.join(fonts_dir, filename)
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, size)
    raise RuntimeError("Не найден шрифт Windows с поддержкой кириллицы.")


def wrap_text(draw, text, font, max_width):
    lines = []
    current_line = ""
    for word in text.split():
        if draw.textbbox((0, 0), word, font=font)[2] > max_width:
            if current_line:
                lines.append(current_line)
                current_line = ""
            part = ""
            for character in word:
                candidate = part + character
                if part and draw.textbbox((0, 0), candidate, font=font)[2] > max_width:
                    lines.append(part)
                    part = character
                else:
                    part = candidate
            current_line = part
            continue

        candidate = f"{current_line} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current_line = candidate
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)
    return lines


def draw_interface(
    camera_frame,
    templates_count,
    detected_letter,
    recognized_text,
    calibration_mode=False,
    calibration_letter=None,
    sample_count=0,
    recording_frames=0,
    completed_letters=0,
    hand_detected=False,
):
    camera_height, camera_width = camera_frame.shape[:2]
    gap = max(14, camera_height // 32)
    sidebar_width = max(340, int(camera_width * 0.48))
    canvas_width = camera_width + sidebar_width + gap * 3
    canvas_height = camera_height + gap * 2
    canvas = Image.new("RGB", (canvas_width, canvas_height), (13, 17, 24))
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(max(23, camera_height // 20))
    heading_font = load_font(max(14, camera_height // 34))
    body_font = load_font(max(15, camera_height // 31))
    small_font = load_font(max(13, camera_height // 38))

    camera_x = gap
    camera_y = gap
    camera_card = (
        camera_x,
        camera_y,
        camera_x + camera_width,
        camera_y + camera_height,
    )
    draw.rounded_rectangle(
        (camera_x - 2, camera_y - 2, camera_card[2] + 2, camera_card[3] + 2),
        radius=16,
        fill=(36, 44, 58),
    )
    camera_image = Image.fromarray(cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB))
    camera_mask = Image.new("L", (camera_width, camera_height), 0)
    ImageDraw.Draw(camera_mask).rounded_rectangle(
        (0, 0, camera_width - 1, camera_height - 1),
        radius=14,
        fill=255,
    )
    canvas.paste(camera_image, (camera_x, camera_y), camera_mask)

    live_color = (84, 203, 143) if hand_detected else (246, 190, 75)
    live_text = "РУКА В КАДРЕ" if hand_detected else "ОЖИДАНИЕ РУКИ"
    badge_font = small_font
    badge_box = draw.textbbox((0, 0), live_text, font=badge_font)
    badge_width = badge_box[2] + 48
    badge_height = badge_font.size + 18
    badge_x = camera_x + 16
    badge_y = camera_y + 16
    draw.rounded_rectangle(
        (badge_x, badge_y, badge_x + badge_width, badge_y + badge_height),
        radius=badge_height // 2,
        fill=(13, 17, 24),
    )
    dot_y = badge_y + badge_height // 2
    draw.ellipse(
        (badge_x + 12, dot_y - 4, badge_x + 20, dot_y + 4),
        fill=live_color,
    )
    draw.text(
        (badge_x + 28, badge_y + (badge_height - badge_font.size) // 2 - 1),
        live_text,
        font=badge_font,
        fill=(239, 243, 249),
    )

    panel_x = camera_x + camera_width + gap
    panel_y = gap
    panel_right = panel_x + sidebar_width
    panel_bottom = panel_y + camera_height
    draw.rounded_rectangle(
        (panel_x, panel_y, panel_right, panel_bottom),
        radius=18,
        fill=(22, 28, 38),
        outline=(39, 48, 62),
        width=1,
    )

    inner_gap = max(16, sidebar_width // 18)
    left = panel_x + inner_gap
    right = panel_right - inner_gap
    content_width = right - left
    draw.text((left, panel_y + 16), "Дактиль", font=title_font, fill=(246, 248, 252))
    title_width = draw.textbbox((0, 0), "Дактиль", font=title_font)[2]
    draw.text(
        (left + title_width + 8, panel_y + 22),
        "ҚАЗ",
        font=heading_font,
        fill=(97, 211, 157),
    )
    subtitle_y = panel_y + 20 + title_font.size
    draw.text(
        (left, subtitle_y),
        "Распознавание казахского дактиля",
        font=small_font,
        fill=(143, 156, 175),
    )

    hero_y = subtitle_y + small_font.size + 20
    hero_height = max(150, int(camera_height * 0.38))
    hero_bottom = min(panel_bottom - 142, hero_y + hero_height)
    letter_font = load_font(max(64, min(camera_height // 6, hero_height // 3)))
    draw.rounded_rectangle(
        (left, hero_y, right, hero_bottom),
        radius=14,
        fill=(29, 37, 50),
    )
    if calibration_mode:
        status_text = "НАСТРОЙКА"
        status_color = (246, 190, 75)
        main_label = "ПОКАЖИТЕ БУКВУ"
        main_letter = calibration_letter or "?"
        letter_color = (255, 255, 255)
        subtitle_status = (
            "Рука найдена" if hand_detected else "Покажите кисть в камеру"
        )
    elif templates_count == 0:
        status_text = "ТРЕБУЕТСЯ НАСТРОЙКА"
        status_color = (246, 190, 75)
        main_label = "ШАБЛОНЫ НЕ ЗАПИСАНЫ"
        main_letter = "—"
        letter_color = (132, 145, 162)
        subtitle_status = "Нажмите Tab, чтобы начать калибровку"
    else:
        status_text = "РАСПОЗНАНИЕ"
        status_color = (84, 203, 143) if detected_letter else (143, 156, 175)
        main_label = "ТЕКУЩАЯ БУКВА"
        main_letter = detected_letter or "—"
        letter_color = (255, 255, 255) if detected_letter else (132, 145, 162)
        subtitle_status = (
            "Знак распознан" if detected_letter else "Покажите знак и держите руку"
        )

    pill_y = hero_y + 12
    pill_box = draw.textbbox((0, 0), status_text, font=small_font)
    pill_width = pill_box[2] + 24
    draw.rounded_rectangle(
        (left + 12, pill_y, left + 12 + pill_width, pill_y + small_font.size + 12),
        radius=small_font.size,
        fill=(38, 48, 62),
    )
    draw.text(
        (left + 24, pill_y + 5),
        status_text,
        font=small_font,
        fill=status_color,
    )

    main_label_box = draw.textbbox((0, 0), main_label, font=small_font)
    main_label_y = hero_y + 46
    main_label_x = left + (content_width - main_label_box[2]) // 2
    draw.text(
        (main_label_x, main_label_y),
        main_label,
        font=small_font,
        fill=(143, 156, 175),
    )
    letter_box = draw.textbbox((0, 0), main_letter, font=letter_font)
    letter_x = left + (content_width - letter_box[2]) // 2
    letter_y = main_label_y + small_font.size + 2
    draw.text((letter_x, letter_y), main_letter, font=letter_font, fill=letter_color)
    status_y = hero_bottom - body_font.size - 13
    status_box = draw.textbbox((0, 0), subtitle_status, font=small_font)
    status_x = left + (content_width - status_box[2]) // 2
    draw.text(
        (status_x, status_y),
        subtitle_status,
        font=small_font,
        fill=(203, 212, 224),
    )

    if calibration_mode:
        detail_y = hero_bottom + 14
        detail_text = (
            f"{sample_count} / {MIN_SAMPLES_PER_LETTER} жестов"
            if recording_frames == 0
            else (
                f"Запись жеста · {recording_frames} / "
                f"{CALIBRATION_CAPTURE_FRAMES} кадров"
            )
        )
        detail_font = small_font
        detail_color = (143, 156, 175)
        instructions = (
            "Пробел  ·  начать запись жеста",
            "Покажите движение целиком около 1 секунды",
            "Для букв у лица держите лицо в кадре",
            "[ / ]  ·  выбрать другую букву",
            "Enter  ·  закончить настройку",
            "Esc  ·  выйти из настройки",
        )
    elif templates_count == 0:
        detail_y = hero_bottom + 14
        detail_text = (
            f"{len(LETTERS)} букв · достаточно {MIN_SAMPLES_PER_LETTER} "
            "образцов на каждую"
        )
        detail_font = small_font
        detail_color = (143, 156, 175)
        instructions = (
            "Tab  ·  начать настройку",
            "Будут сохранены только новые образцы выбранных букв",
            "Снимайте руку целиком, при хорошем освещении",
            "Esc  ·  выход",
        )
    else:
        detail_y = hero_bottom + 14
        detail_text = (
            f"ТЕКСТ  ·  {len(recognized_text)} знаков · "
            f"{len(recognized_text.split())} слов"
        )
        detail_font = small_font
        detail_color = (143, 156, 175)
        instructions = (
            "Enter  ·  вывести текст и очистить",
            "Пробел  ·  разделить слова",
            "Backspace  ·  удалить знак",
            "Tab  ·  настроить шаблоны заново",
            "Делайте жест полностью, включая движение",
            "Уберите руку, чтобы повторить ту же букву",
        )

    draw.text(
        (left + 2, detail_y),
        detail_text,
        font=detail_font,
        fill=detail_color,
    )

    if calibration_mode:
        progress_y = detail_y + small_font.size + 8
        draw.text(
            (left + 2, progress_y),
            f"{completed_letters} из {len(LETTERS)} букв обучено",
            font=small_font,
            fill=(143, 156, 175),
        )
        progress_y += small_font.size + 7
        draw.rounded_rectangle(
            (left, progress_y, right, progress_y + 9),
            radius=5,
            fill=(48, 58, 73),
        )
        progress_width = int(
            content_width * min(completed_letters, len(LETTERS)) / len(LETTERS)
        )
        if progress_width:
            draw.rounded_rectangle(
                (left, progress_y, left + progress_width, progress_y + 9),
                radius=5,
                fill=(84, 203, 143),
            )
        controls_y = progress_y + 20
    elif templates_count > 0:
        text_y = detail_y + small_font.size + 8
        text_bottom = min(panel_bottom - 126, text_y + max(48, camera_height // 8))
        draw.rounded_rectangle(
            (left, text_y, right, text_bottom),
            radius=12,
            fill=(29, 37, 50),
        )
        text_content = recognized_text[-80:] or "Распознанные слова появятся здесь"
        text_color = (245, 247, 251) if recognized_text else (117, 131, 150)
        text_lines = wrap_text(draw, text_content, body_font, content_width - 24)
        max_lines = max(1, (text_bottom - text_y - 16) // (body_font.size + 5))
        line_y = text_y + 10
        for line in text_lines[-max_lines:]:
            draw.text((left + 12, line_y), line, font=body_font, fill=text_color)
            line_y += body_font.size + 5
        controls_y = text_bottom + 14
    else:
        controls_y = detail_y + small_font.size + 16

    available_height = max(1, panel_bottom - controls_y - inner_gap)
    wrapped_instructions = [
        line
        for instruction in instructions
        for line in wrap_text(draw, instruction, small_font, content_width)
    ]
    heading_height = heading_font.size + 14
    line_height = small_font.size + 8
    if heading_height + len(wrapped_instructions) * line_height > available_height:
        max_instruction_lines = max(0, (available_height - heading_height) // line_height)
        wrapped_instructions = wrapped_instructions[:max_instruction_lines]
    draw.line(
        (left, controls_y, right, controls_y),
        fill=(48, 58, 73),
        width=1,
    )
    controls_y += 12
    draw.text(
        (left, controls_y),
        "БЫСТРЫЕ КЛАВИШИ",
        font=heading_font,
        fill=(143, 156, 175),
    )
    controls_y += heading_height
    for instruction in wrapped_instructions:
        draw.text(
            (left, controls_y),
            instruction,
            font=small_font,
            fill=(215, 222, 232),
        )
        controls_y += line_height

    output = np.asarray(canvas)
    return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)


def main():
    templates_path = get_templates_path()
    templates, template_labels = load_templates(templates_path)

    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        webcam.release()
        raise RuntimeError("Не удалось открыть камеру.")

    calibration_index = 0
    calibration_mode = False
    recording_sequence = None
    sequence_buffer = []
    recognized_text = ""
    text_accumulator = SignTextAccumulator()

    try:
        face_model_path = Path(__file__).with_name("haarcascade_frontalface_default.xml")
        compatible_face_model_path = get_windows_compatible_path(face_model_path)
        face_detector = cv2.CascadeClassifier(str(compatible_face_model_path))
        if face_detector.empty():
            raise RuntimeError(
                f"Не удалось загрузить модель распознавания лица: {face_model_path}"
            )
        last_face_box = None
        missed_face_frames = 0

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
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            cv2.namedWindow("Казахский дактиль", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Казахский дактиль", 1120, 680)
            while True:
                success, frame = webcam.read()
                if not success:
                    raise RuntimeError("Не удалось получить изображение с камеры.")

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = time.monotonic_ns() // 1_000_000
                results = landmarker.detect_for_video(image, timestamp_ms)
                hand_landmarks = results.hand_landmarks[0] if results.hand_landmarks else None
                face_box = detect_face(frame, face_detector)
                if face_box is not None:
                    last_face_box = face_box
                    missed_face_frames = 0
                elif (
                    last_face_box is not None
                    and missed_face_frames < FACE_TRACKING_GRACE_FRAMES
                ):
                    face_box = last_face_box
                    missed_face_frames += 1
                else:
                    last_face_box = None
                if face_box is not None:
                    face_x, face_y, face_width, face_height = face_box
                    cv2.rectangle(
                        frame,
                        (face_x, face_y),
                        (face_x + face_width, face_y + face_height),
                        (80, 180, 255),
                        2,
                    )

                detected_letter = None
                feature_vector = None

                if hand_landmarks:
                    draw_hand(frame, hand_landmarks)
                    feature_vector = gesture_features(
                        hand_landmarks,
                        frame.shape,
                        face_box,
                    )

                if calibration_mode:
                    if recording_sequence is not None and feature_vector is not None:
                        recording_sequence.append(feature_vector.copy())
                        if len(recording_sequence) >= CALIBRATION_CAPTURE_FRAMES:
                            templates = np.concatenate(
                                (
                                    templates,
                                    resample_sequence(recording_sequence)[None, :, :],
                                ),
                                axis=0,
                            )
                            template_labels = np.concatenate(
                                (
                                    template_labels,
                                    np.asarray(
                                        [LETTERS[calibration_index]],
                                        dtype="U1",
                                    ),
                                )
                            )
                            save_templates(templates_path, templates, template_labels)
                            recording_sequence = None
                elif feature_vector is not None:
                    sequence_buffer.append(feature_vector)
                    if len(sequence_buffer) > CALIBRATION_CAPTURE_FRAMES:
                        del sequence_buffer[0]
                    if (
                        len(sequence_buffer) == CALIBRATION_CAPTURE_FRAMES
                        and len(templates)
                    ):
                        detected_letter, _ = classify_gesture(
                            resample_sequence(sequence_buffer),
                            templates,
                            template_labels,
                        )
                else:
                    sequence_buffer.clear()

                if not calibration_mode and len(templates):
                    added_letter = text_accumulator.update(detected_letter)
                    if added_letter:
                        recognized_text += added_letter

                calibration_letter = None
                sample_count = 0
                if calibration_mode:
                    calibration_letter = LETTERS[calibration_index]
                    sample_count = int(np.count_nonzero(template_labels == calibration_letter))
                recording_frames = (
                    len(recording_sequence) if recording_sequence is not None else 0
                )
                display_frame = draw_interface(
                    camera_frame=frame,
                    templates_count=len(templates),
                    detected_letter=detected_letter,
                    recognized_text=recognized_text,
                    calibration_mode=calibration_mode,
                    calibration_letter=calibration_letter,
                    sample_count=sample_count,
                    recording_frames=recording_frames,
                    completed_letters=sum(
                        np.count_nonzero(template_labels == letter)
                        >= MIN_SAMPLES_PER_LETTER
                        for letter in LETTERS
                    ),
                    hand_detected=hand_landmarks is not None,
                )
                cv2.imshow("Казахский дактиль", display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    if calibration_mode:
                        calibration_mode = False
                        recording_sequence = None
                    else:
                        break
                if key in (8, 127) and not calibration_mode:
                    recognized_text = recognized_text[:-1]
                elif key in (10, 13) and not calibration_mode:
                    if recognized_text.strip():
                        print(
                            f"\nРаспознанный текст: {recognized_text.strip()}",
                            flush=True,
                        )
                    recognized_text = ""
                if key == 9 and not calibration_mode:
                    calibration_mode = True
                    counts = {
                        letter: np.count_nonzero(template_labels == letter)
                        for letter in LETTERS
                    }
                    calibration_index = next(
                        (
                            index
                            for index, letter in enumerate(LETTERS)
                            if counts[letter] < MIN_SAMPLES_PER_LETTER
                        ),
                        0,
                    )
                    recording_sequence = None
                    sequence_buffer.clear()
                    text_accumulator.reset()
                elif (
                    not calibration_mode
                    and key == ord(" ")
                    and recognized_text
                    and not recognized_text.endswith(" ")
                ):
                    recognized_text += " "
                elif (
                    calibration_mode
                    and key == ord(" ")
                    and recording_sequence is None
                    and feature_vector is not None
                ):
                    recording_sequence = []
                elif calibration_mode and recording_sequence is None and key in (
                    ord("["),
                    ord("]"),
                ):
                    direction = -1 if key == ord("[") else 1
                    calibration_index = (
                        calibration_index + direction
                    ) % len(LETTERS)
                elif calibration_mode and key in (10, 13):
                    calibration_mode = False
                    sequence_buffer.clear()
                    text_accumulator.reset()
    finally:
        webcam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
