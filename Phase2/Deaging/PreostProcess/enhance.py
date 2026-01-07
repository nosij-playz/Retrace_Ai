import cv2
import dlib
import numpy as np


class HeadEnhancer:
    """
    Geometric head crop + wrinkle-safe enhancement
    CPU-only, identity-preserving
    """

    def __init__(
        self,
        predictor_path: str,
        head_radius_mult: float = 2.8,
        head_top_shift: float = 0.35
    ):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

        self.head_radius_mult = head_radius_mult
        self.head_top_shift = head_top_shift

    # -------------------------------------------------
    # GEOMETRIC HEAD CROP (ROBUST)
    # -------------------------------------------------
    def crop_head(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        if not faces:
            raise RuntimeError("No face detected")

        face = max(faces, key=lambda r: r.width() * r.height())
        shape = self.predictor(gray, face)

        pts = np.array([(p.x, p.y) for p in shape.parts()])

        # Eye centers (stable anchor)
        left_eye = pts[36:42].mean(axis=0)
        right_eye = pts[42:48].mean(axis=0)

        eye_center = (left_eye + right_eye) / 2
        eye_dist = np.linalg.norm(left_eye - right_eye)

        # Head radius estimation
        radius = int(self.head_radius_mult * eye_dist)

        # Shift upward to include hair
        center_x = int(eye_center[0])
        center_y = int(eye_center[1] - self.head_top_shift * radius)

        x1 = max(0, center_x - radius)
        y1 = max(0, center_y - radius)
        x2 = min(img.shape[1], center_x + radius)
        y2 = min(img.shape[0], center_y + radius)

        return img[y1:y2, x1:x2]

    # -------------------------------------------------
    # SAFE WRINKLE-PRESERVING ENHANCEMENT
    # -------------------------------------------------
    def enhance_face(self, face: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Mild contrast enhancement (wrinkle-safe)
        clahe = cv2.createCLAHE(clipLimit=1.15, tileGridSize=(8, 8))
        l = clahe.apply(l)

        lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Edge-guided reinforcement (NOT sharpening)
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 80)
        edges = cv2.GaussianBlur(edges, (5, 5), 0)
        edges = edges.astype(np.float32) / 255.0
        edges = cv2.merge([edges, edges, edges])

        final = (
            enhanced * (1 - 0.12 * edges) +
            face * (0.12 * edges)
        ).astype(np.uint8)

        return final

    # -------------------------------------------------
    # FULL PIPELINE
    # -------------------------------------------------
    def process(self, img: np.ndarray) -> np.ndarray:
        head = self.crop_head(img)
        enhanced = self.enhance_face(head)
        return enhanced


# =====================================================
# USAGE
# =====================================================
