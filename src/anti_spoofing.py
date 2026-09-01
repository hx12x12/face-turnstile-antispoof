from typing import Tuple
import cv2
import numpy as np


class PassiveAntiSpoofing:
    """
    Multi-Factor Anti-Spoofing evaluating 2D FFT Moiré patterns, 
    Laplacian sharpness, and YCrCb chrominance consistency to block screen photos.
    """
    def __init__(
        self,
        min_laplacian: float = 15.0,
        max_laplacian: float = 350.0,
        max_fft_ratio: float = 0.35,
        min_chroma_std: float = 4.0
    ) -> None:
        self.min_laplacian = min_laplacian
        self.max_laplacian = max_laplacian
        self.max_fft_ratio = max_fft_ratio
        self.min_chroma_std = min_chroma_std

    def check_liveness(self, face_crop: np.ndarray) -> Tuple[bool, float]:
        if (
            face_crop is None 
            or face_crop.size == 0 
            or len(face_crop.shape) != 3 
            or face_crop.shape[2] != 3 
            or face_crop.shape[0] < 20 
            or face_crop.shape[1] < 20
        ):
            return False, 0.0

        normalized_crop = cv2.resize(face_crop, (128, 128))
        gray = cv2.cvtColor(normalized_crop, cv2.COLOR_BGR2GRAY)

        # 1. Laplacian Sharpness Bounds
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        valid_sharpness = self.min_laplacian <= laplacian_var <= self.max_laplacian

        # 2. 2D FFT High-Frequency Spectral Ratio (Rejects mobile screen grid Moiré noise)
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)

        rows, cols = gray.shape
        center_row, center_col = rows // 2, cols // 2
        mask = np.ones((rows, cols), dtype=np.float32)
        cv2.circle(mask, (center_col, center_row), 16, 0, -1)

        high_freq_energy = np.sum(magnitude_spectrum * mask)
        total_energy = np.sum(magnitude_spectrum) + 1e-6
        high_freq_ratio = float(high_freq_energy / total_energy)
        valid_fft = high_freq_ratio <= self.max_fft_ratio

        # 3. YCrCb Chrominance Consistency
        ycrcb = cv2.cvtColor(normalized_crop, cv2.COLOR_BGR2YCrCb)
        chroma_std = float(np.std(ycrcb[:, :, 1]) + np.std(ycrcb[:, :, 2])) / 2.0
        valid_chroma = chroma_std >= self.min_chroma_std

        # Requires passing all liveness criteria to reject screen playbacks
        is_live = valid_sharpness and valid_fft and valid_chroma
        score = (0.4 if valid_sharpness else 0.0) + (0.4 if valid_fft else 0.0) + (0.2 if valid_chroma else 0.0)

        return is_live, float(score)