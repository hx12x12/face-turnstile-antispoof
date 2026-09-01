import os
import pickle
from typing import Dict, List, Optional, Set, Tuple
import cv2
import numpy as np


class FaceRecognizer:
    """
    Extracts facial feature embeddings, persists authorized user templates to disk,
    and performs cosine similarity matching against saved authorized vectors.
    """
    def __init__(
        self,
        storage_path: str = "models/authorized_users.pkl",
        similarity_threshold: float = 0.70
    ) -> None:
        self.storage_path = storage_path
        self.similarity_threshold = similarity_threshold
        self.known_database: Dict[str, List[np.ndarray]] = {}
        self.allowed_users: Set[str] = set()
        
        # Load persistent database on initialization
        self.load_database()

    def _extract_embedding(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """Extracts an L2-normalized 4096-dimensional spatial vector."""
        if face_crop is None or face_crop.size == 0 or face_crop.shape[0] < 20 or face_crop.shape[1] < 20:
            return None

        resized = cv2.resize(face_crop, (64, 64))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)

        vec = equalized.astype(np.float32).flatten()
        norm = np.linalg.norm(vec)
        if norm == 0:
            return None
        return vec / norm

    def load_database(self) -> None:
        """Loads authorized user feature templates from local storage."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "rb") as file:
                    self.known_database = pickle.load(file)
                    self.allowed_users = set(self.known_database.keys())
                print(f"[+] Loaded {len(self.known_database)} user(s) from '{self.storage_path}'")
            except Exception as error:
                print(f"[-] Failed to load face database ({error}). Starting fresh.")
                self.known_database = {}
                self.allowed_users = set()

    def save_database(self) -> None:
        """Saves authorized user feature templates to local storage."""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, "wb") as file:
            pickle.dump(self.known_database, file)
        print(f"[+] Authorized face database saved to '{self.storage_path}'")

    def register_user(self, identity: str, face_crop: np.ndarray) -> bool:
        """Enrolls a face template for an authorized user and updates disk storage."""
        vec = self._extract_embedding(face_crop)
        if vec is None:
            return False

        if identity not in self.known_database:
            self.known_database[identity] = []

        self.known_database[identity].append(vec)
        self.allowed_users.add(identity)
        self.save_database()
        return True

    def clear_database(self) -> None:
        """Clears all enrolled records in memory and deletes storage file."""
        self.known_database.clear()
        self.allowed_users.clear()
        if os.path.exists(self.storage_path):
            os.remove(self.storage_path)

    def match_face(self, face_crop: np.ndarray) -> Tuple[bool, str, float]:
        """
        Compares input face embedding against disk-stored authorized templates.
        Returns: (is_authorized: bool, identity: str, similarity_score: float)
        """
        vec = self._extract_embedding(face_crop)
        if vec is None or not self.known_database:
            return False, "Unknown", 0.0

        best_match = "Unknown"
        best_score = -1.0

        for identity, templates in self.known_database.items():
            for ref_vec in templates:
                score = float(np.dot(vec, ref_vec))
                if score > best_score:
                    best_score = score
                    best_match = identity

        # Match must meet similarity threshold AND exist in saved authorized users
        if best_score >= self.similarity_threshold and best_match in self.allowed_users:
            return True, best_match, best_score

        return False, "Unknown", max(0.0, best_score)