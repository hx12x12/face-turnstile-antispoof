import time
from typing import Any, Dict, List, Set, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

from src.anti_spoofing import PassiveAntiSpoofing
from src.face_recognition import FaceRecognizer


class RealTimeTurnstilePipeline:
    """
    Real-Time Turnstile System with Persistent User ID Mapping, 
    Strict Unauthorized Label Overrides, and Anti-Spoofing Verification.
    """
    def __init__(
        self,
        yolo_model_path: str = "models/yolov8n-face.pt",
        frame_skip: int = 3,
        conf_thresh: float = 0.5
    ) -> None:
        self.frame_skip = frame_skip
        self.conf_thresh = conf_thresh

        self.detector = YOLO(yolo_model_path)
        self.anti_spoofing = PassiveAntiSpoofing()
        self.recognizer = FaceRecognizer(storage_path="models/authorized_users.pkl", similarity_threshold=0.70)

        self.track_cache: Dict[int, Dict[str, Any]] = {}
        self.last_boxes: List[Tuple[int, Tuple[int, int, int, int]]] = []

        # Persistent identity to ID mapping
        self.user_id_map: Dict[str, int] = {}
        self.next_user_id: int = 1

        # Populate user_id_map from existing database records
        for identity in self.recognizer.known_database.keys():
            self.user_id_map[identity] = self.next_user_id
            self.next_user_id += 1

        self.last_frame_time: float = time.perf_counter()
        self.fps_smoothed: float = 0.0
        self.latest_face_crop: Any = None
        self.status_message: str = "Press 'R' to Register & Save Face | Press 'C' to Reset"

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> Tuple[np.ndarray, float]:
        current_time = time.perf_counter()
        elapsed = current_time - self.last_frame_time
        self.last_frame_time = current_time

        if elapsed > 0:
            instant_fps = 1.0 / elapsed
            self.fps_smoothed = (
                instant_fps if self.fps_smoothed == 0.0 
                else (0.85 * self.fps_smoothed + 0.15 * instant_fps)
            )

        height, width = frame.shape[:2]
        is_detection_frame = (frame_idx % self.frame_skip == 0)

        if is_detection_frame:
            results = self.detector.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
                conf=self.conf_thresh
            )[0]

            self.last_boxes = []
            active_track_ids: Set[int] = set()

            if results.boxes is not None and len(results.boxes) > 0 and results.boxes.id is not None:
                boxes_xyxy = results.boxes.xyxy.cpu().numpy()
                track_ids = results.boxes.id.int().cpu().numpy()

                for box, track_id in zip(boxes_xyxy, track_ids):
                    tid = int(track_id)
                    x1, y1, x2, y2 = map(int, box)
                    self.last_boxes.append((tid, (x1, y1, x2, y2)))
                    active_track_ids.add(tid)

            stale_ids = [tid for tid in self.track_cache if tid not in active_track_ids]
            for tid in stale_ids:
                del self.track_cache[tid]

        # Reset active frame counter for unauthorized detections
        unauthorized_counter = 1

        for track_id, (x1, y1, x2, y2) in self.last_boxes:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            face_crop = frame[y1:y2, x1:x2]
            self.latest_face_crop = face_crop.copy()

            if is_detection_frame or track_id not in self.track_cache:
                is_live, liveness_score = self.anti_spoofing.check_liveness(face_crop)
                is_authorized, identity, match_score = self.recognizer.match_face(face_crop)

                authorized = is_live and is_authorized

                if authorized:
                    if identity not in self.user_id_map:
                        self.user_id_map[identity] = self.next_user_id
                        self.next_user_id += 1
                    display_id = self.user_id_map[identity]
                    display_identity = identity
                else:
                    # ALWAYS force identity to 'Unauthorized' on Red Box / ACCESS DENIED
                    display_identity = "Unauthorized"
                    display_id = None

                self.track_cache[track_id] = {
                    "identity": display_identity,
                    "display_id": display_id,
                    "authorized": authorized,
                    "is_live": is_live,
                    "match_score": match_score
                }

            track_info = self.track_cache.get(track_id, {
                "identity": "Unauthorized", "display_id": None, "authorized": False, "is_live": False, "match_score": 0.0
            })

            # Rendering logic
            if track_info["authorized"]:
                box_color = (0, 255, 0)
                status_text = "ACCESS GRANTED"
                id_label = f"ID:{track_info['display_id']}"
                identity_label = track_info["identity"]
            else:
                box_color = (0, 0, 255)
                status_text = "ACCESS DENIED"
                id_label = f"ID:{unauthorized_counter}"
                identity_label = "Unauthorized"
                unauthorized_counter += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            label = f"{id_label} | {identity_label} | {status_text}"
            cv2.putText(frame, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        cv2.putText(frame, f"FPS: {self.fps_smoothed:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, self.status_message, (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        return frame, self.fps_smoothed

    def run_video_stream(self, source: Any = 0) -> None:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video source: {source}")

        window_name = "Real-Time Whitelisted Face Turnstile"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        frame_idx = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame, _ = self.process_frame(frame, frame_idx)
                cv2.imshow(window_name, processed_frame)
                frame_idx += 1

                key = cv2.waitKey(1) & 0xFF

                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

                if key in (ord('r'), ord('R')):
                    if self.latest_face_crop is not None:
                        user_name = f"Authorized_User_{self.next_user_id}"
                        success = self.recognizer.register_user(user_name, self.latest_face_crop)
                        if success:
                            self.user_id_map[user_name] = self.next_user_id
                            self.status_message = f"[+] Saved '{user_name}' to disk storage! (ID:{self.next_user_id})"
                            self.next_user_id += 1
                        else:
                            self.status_message = "[-] Registration Failed: Invalid face crop."
                    else:
                        self.status_message = "[-] No face detected to register."

                elif key in (ord('c'), ord('C')):
                    self.recognizer.clear_database()
                    self.track_cache.clear()
                    self.user_id_map.clear()
                    self.next_user_id = 1
                    self.status_message = "[!] Database reset. All faces marked Unauthorized."

                elif key in (ord('q'), ord('Q'), 27):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()