"""MediaPipe Face Landmarker based facial-emotion recognizer for Week 8.

The node uses Face Landmarker blendshapes plus geometry from facial landmarks.
This is intentionally deterministic: it does not claim a generic emotion model
has understood the user's personality. It maps observable facial cues to the
labels required by the assignment and stabilizes them over several frames.
"""
from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, String


LABELS = ("Happy", "Sad", "Neutral", "Angry", "Surprise", "Fear")


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(np.clip(value, low, high))


def mean_score(scores: Dict[str, float], *names: str) -> float:
    values = [float(scores.get(name, 0.0)) for name in names]
    return float(np.mean(values)) if values else 0.0


class EmotionRecognitionNode(Node):
    """Read a webcam frame, estimate facial cues and publish a stable label."""

    def __init__(self) -> None:
        super().__init__("emotion_recognition_node")

        self.declare_parameter("camera_index", 0)
        self.declare_parameter("backend", "mediapipe")
        self.declare_parameter("show_window", True)
        self.declare_parameter("publish_image", True)
        self.declare_parameter("draw_landmarks", True)
        self.declare_parameter("show_debug_cues", False)
        self.declare_parameter("analysis_every_n_frames", 2)
        self.declare_parameter("stable_window", 7)
        self.declare_parameter("min_stable_votes", 3)
        self.declare_parameter("min_confidence", 0.45)
        self.declare_parameter("mirror", True)
        self.declare_parameter("face_model", "")
        self.declare_parameter("hand_model", "")

        self.camera_index = int(self.get_parameter("camera_index").value)
        self.show_window = bool(self.get_parameter("show_window").value)
        self.publish_image_enabled = bool(self.get_parameter("publish_image").value)
        self.draw_landmarks = bool(self.get_parameter("draw_landmarks").value)
        self.show_debug_cues = bool(self.get_parameter("show_debug_cues").value)
        self.analysis_period = max(
            1, int(self.get_parameter("analysis_every_n_frames").value)
        )
        self.stable_window = max(3, int(self.get_parameter("stable_window").value))
        self.min_stable_votes = max(
            2, int(self.get_parameter("min_stable_votes").value)
        )
        self.min_confidence = float(self.get_parameter("min_confidence").value)
        self.mirror = bool(self.get_parameter("mirror").value)

        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(Image, "/facial_emotion/image", 1)
        self.emotion_pub = self.create_publisher(String, "/facial_emotion", 10)
        self.confidence_pub = self.create_publisher(
            Float32, "/facial_emotion_confidence", 10
        )
        self.status_pub = self.create_publisher(
            String, "/facial_emotion_status", 10
        )

        self.capture: Optional[cv2.VideoCapture] = None
        self.last_open_attempt = 0.0
        self.frame_count = 0
        self.last_result: Tuple[str, float, Dict[str, float]] = (
            "Neutral",
            0.0,
            {},
        )
        self.history: deque[Tuple[str, float]] = deque(maxlen=self.stable_window)
        self.pending_label = ""
        self.pending_votes = 0
        self.pending_confidences: deque[float] = deque(maxlen=self.min_stable_votes)
        self.stable_label = "Neutral"
        self.stable_confidence = 0.0

        self.face_landmarker: Optional[vision.FaceLandmarker] = None
        self.hand_landmarker: Optional[vision.HandLandmarker] = None
        self.face_detector = cv2.CascadeClassifier(
            str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        )
        self._init_landmarkers()

        self.timer = self.create_timer(0.03, self.process_frame)
        self.retry_timer = self.create_timer(2.0, self.ensure_camera)
        self.ensure_camera()

        self.get_logger().info(
            "Emotion webcam node san sang; MediaPipe Face Landmarker + blendshapes."
        )

    def _model_path(self, parameter_name: str, default_name: str) -> Path:
        configured = str(self.get_parameter(parameter_name).value).strip()
        if configured:
            return Path(configured).expanduser()
        share = Path(get_package_share_directory("tb4_week8_emotion"))
        return share / "models" / default_name

    def _init_landmarkers(self) -> None:
        face_path = self._model_path("face_model", "face_landmarker.task")
        hand_path = self._model_path("hand_model", "hand_landmarker.task")
        try:
            face_options = vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(face_path)
                ),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.60,
                min_face_presence_confidence=0.60,
                min_tracking_confidence=0.60,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=False,
            )
            self.face_landmarker = vision.FaceLandmarker.create_from_options(
                face_options
            )
            self.get_logger().info(f"Face model: {face_path}")
        except Exception as exc:
            self.get_logger().error(
                f"Khong nap duoc face_landmarker.task ({face_path}): {exc}"
            )
            self.face_landmarker = None

        try:
            hand_options = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(hand_path)
                ),
                running_mode=vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.55,
                min_hand_presence_confidence=0.55,
                min_tracking_confidence=0.55,
            )
            self.hand_landmarker = vision.HandLandmarker.create_from_options(
                hand_options
            )
            self.get_logger().info(f"Hand model: {hand_path}")
        except Exception as exc:
            self.get_logger().warning(
                f"Khong nap duoc hand_landmarker.task ({hand_path}); Fear se dung face cue: {exc}"
            )
            self.hand_landmarker = None

    def _safe_publish(self, publisher: Any, message: Any) -> None:
        if not rclpy.ok():
            return
        try:
            publisher.publish(message)
        except Exception as exc:
            self.get_logger().debug(f"Bo qua publish khi ROS dang shutdown: {exc}")

    def publish_status(self, text: str) -> None:
        msg = String()
        msg.data = text
        self._safe_publish(self.status_pub, msg)

    def ensure_camera(self) -> None:
        if self.capture is not None and self.capture.isOpened():
            return
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self.last_open_attempt < 1.5:
            return
        self.last_open_attempt = now
        if self.capture is not None:
            self.capture.release()

        capture = cv2.VideoCapture(self.camera_index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not capture.isOpened():
            capture.release()
            self.capture = None
            self.publish_status("Webcam chua san sang; dang thu lai.")
            return
        self.capture = capture
        self.publish_status("Webcam da san sang cho nhan dien cam xuc.")

    @staticmethod
    def _largest_face(
        faces: Iterable[Tuple[int, int, int, int]]
    ) -> Optional[Tuple[int, int, int, int]]:
        faces = list(faces)
        if not faces:
            return None
        x, y, w, h = max(faces, key=lambda item: int(item[2]) * int(item[3]))
        return int(x), int(y), int(w), int(h)

    @staticmethod
    def _point(landmarks: list, index: int) -> np.ndarray:
        point = landmarks[index]
        return np.array([float(point.x), float(point.y)], dtype=np.float32)

    @classmethod
    def _distance(cls, landmarks: list, first: int, second: int) -> float:
        return float(np.linalg.norm(cls._point(landmarks, first) - cls._point(landmarks, second)))

    @staticmethod
    def _blendshapes(result: Any) -> Dict[str, float]:
        output: Dict[str, float] = {}
        if not getattr(result, "face_blendshapes", None):
            return output
        for category in result.face_blendshapes[0]:
            name = str(category.category_name)
            output[name] = max(output.get(name, 0.0), float(category.score))
        return output

    @staticmethod
    def _blendshape(scores: Dict[str, float], *names: str) -> float:
        return mean_score(scores, *names)

    def _mouth_pixels(
        self,
        frame: np.ndarray,
        landmarks: list,
        face_box: Tuple[int, int, int, int],
    ) -> Tuple[float, float]:
        """Return visible-teeth and clenched-teeth estimates from the mouth ROI."""
        height, width = frame.shape[:2]
        x, y, w, h = face_box
        mouth_indices = (61, 291, 13, 14, 78, 308)
        points = np.array(
            [
                [
                    int(np.clip(self._point(landmarks, index)[0] * width, 0, width - 1)),
                    int(np.clip(self._point(landmarks, index)[1] * height, 0, height - 1)),
                ]
                for index in mouth_indices
            ],
            dtype=np.int32,
        )
        left = max(0, int(points[:, 0].min()) - max(2, w // 35))
        right = min(width, int(points[:, 0].max()) + max(2, w // 35))
        top = max(0, int(points[:, 1].min()) - max(2, h // 45))
        bottom = min(height, int(points[:, 1].max()) + max(2, h // 35))
        if right <= left or bottom <= top:
            return 0.0, 0.0

        roi = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
        if roi.size == 0:
            return 0.0, 0.0
        roi = cv2.GaussianBlur(roi, (3, 3), 0)

        # White, low-saturation pixels in the mouth opening are a teeth cue.
        hsv = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2HSV)
        bright = (roi > 155) & (hsv[:, :, 1] < 105)
        bright_ratio = float(np.mean(bright))

        # Exclude the outer lips and cheeks; clenched teeth are a bright,
        # mostly horizontal band while the mouth opening remains small.
        inner = bright[
            max(0, int(bright.shape[0] * 0.18)): max(1, int(bright.shape[0] * 0.82)),
            max(0, int(bright.shape[1] * 0.08)): max(1, int(bright.shape[1] * 0.92)),
        ]
        teeth_band = float(np.max(np.mean(inner, axis=1))) if inner.size else 0.0
        visible = clamp(0.45 * (bright_ratio - 0.025) / 0.16 + 0.55 * teeth_band)
        return visible, clamp(visible * 0.75 + (1.0 - visible) * 0.0)

    def _hand_near_face(
        self,
        frame_width: int,
        frame_height: int,
        result: Any,
        face_box: Tuple[int, int, int, int],
    ) -> float:
        if result is None or not getattr(result, "hand_landmarks", None):
            return 0.0

        x, y, w, h = face_box
        fx, fy = x / frame_width, y / frame_height
        fw, fh = w / frame_width, h / frame_height
        mouth = np.array([fx + 0.50 * fw, fy + 0.68 * fh], dtype=np.float32)
        scores = []
        for hand in result.hand_landmarks:
            points = np.array(
                [[float(point.x), float(point.y)] for point in hand],
                dtype=np.float32,
            )
            distances = np.linalg.norm(points - mouth, axis=1)
            nearest = float(np.min(distances))
            near_mouth = clamp(
                1.0 - nearest / max(0.22 * fw + 0.12 * fh, 1e-4)
            )
            inside_lower_face = (
                (points[:, 0] >= fx - 0.18 * fw)
                & (points[:, 0] <= fx + 1.18 * fw)
                & (points[:, 1] >= fy + 0.32 * fh)
                & (points[:, 1] <= fy + 1.05 * fh)
            )
            coverage = clamp(float(np.mean(inside_lower_face)) / 0.35)
            scores.append(clamp(0.72 * near_mouth + 0.28 * coverage))
        return max(scores, default=0.0)

    def _features(
        self,
        frame: np.ndarray,
        face_landmarks: list,
        blendshapes: Dict[str, float],
        hands_near_face: float,
        face_box: Tuple[int, int, int, int],
    ) -> Dict[str, float]:
        left_eye_width = self._distance(face_landmarks, 33, 133)
        right_eye_width = self._distance(face_landmarks, 362, 263)
        left_eye_open = (
            self._distance(face_landmarks, 159, 145)
            + self._distance(face_landmarks, 158, 153)
        ) / (2.0 * max(left_eye_width, 1e-6))
        right_eye_open = (
            self._distance(face_landmarks, 386, 374)
            + self._distance(face_landmarks, 385, 380)
        ) / (2.0 * max(right_eye_width, 1e-6))
        geometric_eye_open = clamp((left_eye_open + right_eye_open) / 0.42)

        mouth_width = self._distance(face_landmarks, 61, 291)
        mouth_open_ratio = self._distance(face_landmarks, 13, 14) / max(mouth_width, 1e-6)
        jaw_open = self._blendshape(blendshapes, "jawOpen")
        mouth_o = clamp(
            0.58 * clamp((mouth_open_ratio - 0.18) / 0.27)
            + 0.42 * jaw_open
        )

        eye_wide_shape = self._blendshape(blendshapes, "eyeWideLeft", "eyeWideRight")
        eye_squint_shape = max(
            self._blendshape(blendshapes, "eyeSquintLeft", "eyeSquintRight"),
            self._blendshape(blendshapes, "eyeBlinkLeft", "eyeBlinkRight"),
        )
        eye_wide = clamp(0.72 * eye_wide_shape + 0.28 * clamp((geometric_eye_open - 0.50) / 0.34))
        eye_squint = clamp(
            0.68 * eye_squint_shape
            + 0.32 * clamp((0.38 - geometric_eye_open) / 0.27)
        )

        corner_left = self._point(face_landmarks, 61)
        corner_right = self._point(face_landmarks, 291)
        lip_center = (
            self._point(face_landmarks, 13) + self._point(face_landmarks, 14)
        ) / 2.0

        # Remove head-roll influence before measuring a crooked mouth.
        left_eye_center = (
            self._point(face_landmarks, 33) + self._point(face_landmarks, 133)
        ) / 2.0
        right_eye_center = (
            self._point(face_landmarks, 362) + self._point(face_landmarks, 263)
        ) / 2.0
        roll = float(
            (right_eye_center[1] - left_eye_center[1])
            / max(right_eye_center[0] - left_eye_center[0], 1e-6)
        )
        face_mid_x = float((left_eye_center[0] + right_eye_center[0]) / 2.0)

        def corrected_y(point: np.ndarray) -> float:
            return float(point[1] - roll * (point[0] - face_mid_x))

        left_corner_y = corrected_y(corner_left)
        right_corner_y = corrected_y(corner_right)
        center_lip_y = corrected_y(lip_center)
        corner_y = (left_corner_y + right_corner_y) / 2.0
        corner_raise = clamp((center_lip_y - corner_y) / 0.075)
        corner_down = clamp((corner_y - center_lip_y - 0.003) / 0.060)
        mouth_asymmetry = clamp(
            abs(left_corner_y - right_corner_y)
            / max(mouth_width, 1e-6)
            / 0.16
        )

        smile_shape = self._blendshape(blendshapes, "mouthSmileLeft", "mouthSmileRight")
        frown_shape = self._blendshape(blendshapes, "mouthFrownLeft", "mouthFrownRight")
        pucker_shape = self._blendshape(blendshapes, "mouthPucker")
        mouth_press = self._blendshape(
            blendshapes, "mouthPressLeft", "mouthPressRight"
        )
        mouth_stretch = self._blendshape(
            blendshapes, "mouthStretchLeft", "mouthStretchRight"
        )
        mouth_shrug = self._blendshape(
            blendshapes, "mouthShrugLower", "mouthShrugUpper"
        )
        brow_down = self._blendshape(blendshapes, "browDownLeft", "browDownRight")
        brow_up = self._blendshape(
            blendshapes, "browInnerUp", "browOuterUpLeft", "browOuterUpRight"
        )
        nose_sneer = self._blendshape(
            blendshapes, "noseSneerLeft", "noseSneerRight"
        )
        cheek_puff = self._blendshape(blendshapes, "cheekPuff")
        teeth_visible, teeth_clenched = self._mouth_pixels(
            frame, face_landmarks, face_box
        )

        surprise = clamp(
            0.62 * mouth_o + 0.18 * eye_wide + 0.20 * brow_up
        )
        # The supplied Angry examples show a tense, pursed/pressed mouth
        # more clearly than visible teeth. Keep that cue separate from O-mouth
        # Surprise: a strong pucker with eyes open is Angry, while a large jaw
        # opening remains Surprise.
        angry_pucker = clamp(0.55 * eye_wide + 0.45 * pucker_shape)
        # In the second supplied Angry image the model overestimates
        # mouthSmile. The discriminating cue is the combination of unusually
        # wide eyes and a tense, not-actually-raised mouth corner.
        angry_eye_mouth = clamp(
            0.55 * clamp((eye_wide - 0.25) / 0.12)
            + 0.45 * clamp((smile_shape - 0.55) / 0.25)
        ) if (
            eye_wide >= 0.28
            and smile_shape >= 0.65
            and mouth_o < 0.35
            and corner_raise < 0.42
        ) else 0.0
        angry = max(
            angry_pucker,
            angry_eye_mouth,
            clamp(
                0.26 * eye_wide
                + 0.20 * brow_down
                + 0.17 * nose_sneer
                + 0.15 * mouth_press
                + 0.10 * mouth_stretch
                + 0.07 * teeth_clenched
                + 0.05 * mouth_asymmetry
            ),
        )
        sad = clamp(
            0.60 * eye_squint
            + 0.18 * max(frown_shape, corner_down, mouth_shrug)
            + 0.10 * mouth_asymmetry
            + 0.07 * pucker_shape
            + 0.05 * mouth_press
        )
        happy = clamp(
            0.60 * smile_shape
            + 0.22 * corner_raise
            + 0.13 * teeth_visible
            + 0.05 * self._blendshape(
                blendshapes, "cheekSquintLeft", "cheekSquintRight"
            )
        )
        fear = clamp(
            0.50 * hands_near_face
            + 0.22 * eye_wide
            + 0.16 * brow_up
            + 0.12 * mouth_o
        )

        return {
            "mouth_o": mouth_o,
            "mouth_open_ratio": clamp(mouth_open_ratio / 0.55),
            "eye_wide": eye_wide,
            "eye_squint": eye_squint,
            "mouth_asymmetry": mouth_asymmetry,
            "corner_raise": corner_raise,
            "corner_down": corner_down,
            "teeth_visible": teeth_visible,
            "teeth_clenched": teeth_clenched,
            "brow_down": brow_down,
            "brow_up": brow_up,
            "mouth_press": mouth_press,
            "mouth_stretch": mouth_stretch,
            "mouth_shrug": mouth_shrug,
            "hands_near_face": hands_near_face,
            "surprise_score": surprise,
            "angry_score": angry,
            "angry_pucker": angry_pucker,
            "angry_eye_mouth": angry_eye_mouth,
            "sad_score": sad,
            "happy_score": happy,
            "fear_score": fear,
            "smile_shape": smile_shape,
            "frown_shape": frown_shape,
            "jaw_open": jaw_open,
            "pucker": pucker_shape,
            "cheek_puff": cheek_puff,
        }

    @staticmethod
    def _classify(features: Dict[str, float]) -> Tuple[str, float]:
        scores = {
            "Happy": features["happy_score"],
            "Sad": features["sad_score"],
            "Angry": features["angry_score"],
            "Surprise": features["surprise_score"],
            "Fear": features["fear_score"],
            "Neutral": 0.46,
        }

        # Specific cues have priority. O-shaped mouth is Surprise unless a
        # hand is covering the face, which is the Fear example in this task.
        if (
            features["hands_near_face"] >= 0.55
            and features["fear_score"] >= 0.44
        ):
            label = "Fear"
        elif (
            features["mouth_o"] >= 0.58
            and features["surprise_score"] >= features["angry_score"] - 0.04
        ):
            label = "Surprise"
        elif (
            features["eye_wide"] >= 0.28
            and features["mouth_o"] < 0.35
            and features["pucker"] >= 0.55
        ):
            label = "Angry"
        elif (
            features["eye_wide"] >= 0.40
            and features["mouth_o"] < 0.58
            and (
                features["teeth_clenched"] >= 0.20
                or features["brow_down"] >= 0.34
                or features["mouth_press"] >= 0.32
                or features["mouth_stretch"] >= 0.34
                or features["angry_score"] >= 0.48
            )
        ):
            label = "Angry"
        elif (
            features["sad_score"] >= 0.52
            and features["eye_squint"] >= 0.38
            and features["happy_score"] < 0.55
        ):
            label = "Sad"
        elif features["happy_score"] >= 0.55:
            label = "Happy"
        else:
            label = max(scores, key=scores.get)

        confidence = clamp(scores[label])
        if label == "Neutral":
            confidence = clamp(0.58 + 0.20 * (1.0 - max(scores.values())))
        return label, confidence

    def _detect(self, frame: np.ndarray) -> Tuple[Optional[Dict[str, Any]], Any]:
        if self.face_landmarker is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(
                gray, 1.1, 5, minSize=(80, 80)
            )
            face = self._largest_face(faces)
            if face is None:
                return None, None
            return {"box": face}, None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        face_result = self.face_landmarker.detect(image)
        if not face_result.face_landmarks:
            return None, None

        landmarks = face_result.face_landmarks[0]
        height, width = frame.shape[:2]
        xs = [float(point.x) for point in landmarks]
        ys = [float(point.y) for point in landmarks]
        x0 = int(np.clip(min(xs) * width, 0, width - 1))
        y0 = int(np.clip(min(ys) * height, 0, height - 1))
        x1 = int(np.clip(max(xs) * width, x0 + 1, width))
        y1 = int(np.clip(max(ys) * height, y0 + 1, height))
        box = (x0, y0, x1 - x0, y1 - y0)

        hand_result = (
            self.hand_landmarker.detect(image)
            if self.hand_landmarker is not None
            else None
        )
        return {
            "box": box,
            "landmarks": landmarks,
            "blendshapes": self._blendshapes(face_result),
        }, hand_result

    def stable_update(self, label: str, confidence: float) -> None:
        # Sad/Fear in the supplied screenshots are valid cues around 0.48-0.52.
        # Do not let the old 0.54 cutoff silently turn them into Neutral.
        accepted = confidence >= self.min_confidence
        if label in {"Sad", "Fear", "Angry"} and confidence >= 0.44:
            accepted = True
        if not accepted:
            label = "Neutral"

        self.history.append((label, confidence))
        if label == self.pending_label:
            self.pending_votes += 1
        else:
            self.pending_label = label
            self.pending_votes = 1
            self.pending_confidences.clear()
        self.pending_confidences.append(float(confidence))

        # Use consecutive frames instead of a majority that can be dominated
        # by the previous expression. Three consistent frames are enough.
        if self.pending_votes >= self.min_stable_votes:
            self.stable_label = label
            self.stable_confidence = sum(self.pending_confidences) / max(
                1, len(self.pending_confidences)
            )

        msg = String()
        msg.data = self.stable_label
        self._safe_publish(self.emotion_pub, msg)
        conf = Float32()
        conf.data = float(self.stable_confidence)
        self._safe_publish(self.confidence_pub, conf)

    def _draw_landmarks(self, image: np.ndarray, landmarks: list) -> None:
        height, width = image.shape[:2]
        for point in landmarks:
            px = int(np.clip(point.x * width, 0, width - 1))
            py = int(np.clip(point.y * height, 0, height - 1))
            cv2.circle(image, (px, py), 1, (255, 190, 0), -1)

    def process_frame(self) -> None:
        self.ensure_camera()
        if self.capture is None or not self.capture.isOpened():
            return
        ok, frame = self.capture.read()
        if not ok or frame is None:
            self.publish_status("Khong doc duoc frame webcam; dang thu lai.")
            return

        self.frame_count += 1
        if self.mirror:
            frame = cv2.flip(frame, 1)
        display = frame.copy()

        detected_for_draw = None
        if self.frame_count % self.analysis_period == 0:
            try:
                detected_for_draw, hand_result = self._detect(frame)
                if detected_for_draw is None:
                    self.history.clear()
                    self.pending_label = ""
                    self.pending_votes = 0
                    self.pending_confidences.clear()
                    self.stable_label = "Neutral"
                    self.stable_confidence = 0.0
                    unknown = String()
                    unknown.data = "Unknown"
                    self._safe_publish(self.emotion_pub, unknown)
                elif "landmarks" in detected_for_draw:
                    hands_near_face = self._hand_near_face(
                        frame.shape[1],
                        frame.shape[0],
                        hand_result,
                        detected_for_draw["box"],
                    )
                    features = self._features(
                        frame,
                        detected_for_draw["landmarks"],
                        detected_for_draw["blendshapes"],
                        hands_near_face,
                        detected_for_draw["box"],
                    )
                    label, confidence = self._classify(features)
                    self.last_result = (label, confidence, features)
                    self.stable_update(label, confidence)
                else:
                    self.last_result = ("Neutral", 0.55, {})
                    self.stable_update("Neutral", 0.55)
            except Exception as exc:
                self.get_logger().warning(f"Loi xu ly frame: {exc}")
                self.publish_status("Loi xu ly frame; dang tiep tuc.")
        else:
            try:
                detected_for_draw, _ = self._detect(frame)
            except Exception as exc:
                self.get_logger().debug(f"Bo qua frame hien thi: {exc}")

        if detected_for_draw is not None:
            x, y, w, h = detected_for_draw["box"]
            color = (0, 220, 0) if self.stable_confidence >= self.min_confidence else (0, 180, 255)
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
            if self.draw_landmarks and "landmarks" in detected_for_draw:
                self._draw_landmarks(display, detected_for_draw["landmarks"])
            text = f"{self.stable_label}: {self.stable_confidence:.2f}"
            cv2.putText(
                display, text, (x, max(25, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2
            )
            if self.show_debug_cues:
                features = self.last_result[2]
                cue = (
                    "O-mouth -> Surprise"
                    if features.get("mouth_o", 0.0) >= 0.58
                    else "eyes + brow + clenched mouth -> Angry"
                    if features.get("angry_score", 0.0) >= 0.45
                    else "closed eyes + downturned pout -> Sad"
                    if features.get("sad_score", 0.0) >= 0.50
                    else "smile -> Happy"
                    if features.get("happy_score", 0.0) >= 0.50
                    else "hand near face -> Fear"
                    if features.get("hands_near_face", 0.0) >= 0.45
                    else "face"
                )
                cue_y = min(display.shape[0] - 38, y + h + 24)
                cv2.putText(
                    display, f"cue: {cue}",
                    (x, cue_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 0), 1
                )
                debug_scores = (
                    f"S:{features.get('sad_score', 0.0):.2f} "
                    f"A:{features.get('angry_score', 0.0):.2f} "
                    f"F:{features.get('fear_score', 0.0):.2f} "
                    f"hand:{features.get('hands_near_face', 0.0):.2f}"
                )
                cv2.putText(
                    display, debug_scores,
                    (x, min(display.shape[0] - 12, cue_y + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 0), 1
                )
        else:
            cv2.putText(
                display, "No face / Unknown", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 255), 2
            )

        cv2.putText(
            display,
            "MediaPipe: Happy | Sad | Angry | Surprise | Fear | Neutral",
            (10, display.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
        )

        if self.publish_image_enabled:
            try:
                image_msg = self.bridge.cv2_to_imgmsg(display, encoding="bgr8")
                self._safe_publish(self.image_pub, image_msg)
            except Exception as exc:
                self.get_logger().debug(f"Khong publish duoc anh webcam: {exc}")

        if self.show_window:
            try:
                cv2.imshow("TB4 Week 8 - MediaPipe Emotion", display)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    rclpy.shutdown()
            except cv2.error as exc:
                self.show_window = False
                self.get_logger().warning(f"Khong mo duoc cua so OpenCV: {exc}")

    def destroy_node(self):
        if self.capture is not None:
            self.capture.release()
        if self.face_landmarker is not None:
            self.face_landmarker.close()
        if self.hand_landmarker is not None:
            self.hand_landmarker.close()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EmotionRecognitionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
