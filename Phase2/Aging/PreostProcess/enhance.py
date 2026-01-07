import cv2
import dlib
import numpy as np


class HeadEnhancer:
    """
    Stable head crop
    CPU-only, identity-preserving
    """

    def __init__(
        self,
        predictor_path: str,
        head_radius_mult: float = 3,    # slightly reduced (safer)
        head_top_shift: float = 0.30,   # matches de-aging crop better
        neck_extension_mult: float = 0.4  # extra neck coverage to avoid cutting
    ):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

        self.head_radius_mult = head_radius_mult
        self.head_top_shift = head_top_shift
        self.neck_extension_mult = neck_extension_mult

    # -------------------------------------------------
    # STABLE HEAD CROP (DE-AGING COMPATIBLE)
    # -------------------------------------------------
    def crop_head(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        if not faces:
            raise RuntimeError("No face detected")

        face = max(faces, key=lambda r: r.width() * r.height())
        shape = self.predictor(gray, face)

        pts = np.array([(p.x, p.y) for p in shape.parts()],
                       dtype=np.float32)

        # --- Stable anchors ---
        left_eye  = pts[36:42].mean(axis=0)
        right_eye = pts[42:48].mean(axis=0)

        eye_center = (left_eye + right_eye) / 2.0
        eye_dist   = np.linalg.norm(left_eye - right_eye)

        # --- Head size ---
        radius = int(self.head_radius_mult * eye_dist)
        neck_extra = int(radius * self.neck_extension_mult)

        cx = int(eye_center[0])
        cy = int(eye_center[1] - self.head_top_shift * radius)

        # --- Square crop with extra neck room (bottom-extended) ---
        x1 = int(cx - radius)
        y1 = int(cy - radius)
        x2 = int(cx + radius)
        y2 = int(cy + radius + neck_extra)

        # --- Clamp safely ---
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.shape[1], x2)
        y2 = min(img.shape[0], y2)

        cropped = img[y1:y2, x1:x2]

        if cropped.size == 0:
            raise RuntimeError("Invalid crop region")

        return cropped

    # -------------------------------------------------
    # FULL PIPELINE
    # -------------------------------------------------
    def process(self, img: np.ndarray) -> np.ndarray:
        head = self.crop_head(img)
        return head
