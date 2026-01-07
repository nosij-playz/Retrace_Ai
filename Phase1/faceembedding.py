import insightface
from insightface.app import FaceAnalysis
import cv2
import numpy as np
from numpy.linalg import norm
import os


class FaceEmbedder:
    def __init__(self, model_name="buffalo_l", det_size=(640, 640), crop_dir="static/uploads/cropped_faces", crop_margin=0.25):
        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=0, det_size=det_size)
        self.crop_dir = crop_dir
        self.crop_margin = crop_margin
        os.makedirs(self.crop_dir, exist_ok=True)

        # Lightweight pre-check to reject non-face images early
        self._haar_face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    # --------------------------------------
    # Utility: L2 Normalization
    # --------------------------------------
    @staticmethod
    def l2norm(x):
        return x / np.linalg.norm(x)

    # --------------------------------------
    # Utility: Normalize value to 0-1 range
    # --------------------------------------
    @staticmethod
    def normalize(x, lo, hi):
        return float(np.clip((x - lo) / (hi - lo), 0, 1))

    # --------------------------------------
    # Presentation Only: Color vs Grayscale
    # --------------------------------------
    @staticmethod
    def classify_color_type(img_bgr, gray_thr=2.0, mixed_thr=10.0):
        """Classify image as 'color', 'mixed', or 'grayscale' based on channel diversity."""
        if img_bgr is None:
            return "grayscale"
        if len(img_bgr.shape) < 3 or img_bgr.shape[2] < 3:
            return "grayscale"

        b, g, r = cv2.split(img_bgr)
        r = r.astype(np.int16)
        g = g.astype(np.int16)
        b = b.astype(np.int16)

        diff_rg = float(np.mean(np.abs(r - g)))
        diff_rb = float(np.mean(np.abs(r - b)))
        diff_gb = float(np.mean(np.abs(g - b)))
        mean_diff = (diff_rg + diff_rb + diff_gb) / 3.0

        if mean_diff < gray_thr:
            return "grayscale"
        if mean_diff < mixed_thr:
            return "mixed"
        return "color"

    @staticmethod
    def quality_display_score(raw_quality, color_type):
        """Map raw model quality (0-1) to a UI-friendly score range, without affecting decisions."""
        if raw_quality is None:
            return None

        ranges = {
            "color": (60.0, 90.0),
            "mixed": (50.0, 60.0),
            "grayscale": (40.0, 50.0),
        }
        lo, hi = ranges.get(color_type, (50.0, 60.0))
        return float(lo + float(raw_quality) * (hi - lo))

    # --------------------------------------
    # Quality Component: Embedding Signal Strength (ESS)
    # --------------------------------------
    @staticmethod
    def embedding_signal_strength(emb):
        """Variance of embedding - measures information content"""
        return float(np.var(emb))

    # --------------------------------------
    # Quality Component: Self-Consistency Score (SC)
    # --------------------------------------
    @staticmethod
    def self_consistency_score(emb_list):
        """Average cosine similarity between multiple embeddings of same face"""
        if len(emb_list) < 2:
            return 0.0
        sims = []
        for i in range(len(emb_list)):
            for j in range(i + 1, len(emb_list)):
                sims.append(np.dot(emb_list[i], emb_list[j]))
        return float(np.mean(sims))

    # --------------------------------------
    # Quality Component: Pose Penalty
    # --------------------------------------
    @staticmethod
    def pose_penalty(face):
        """Penalty for extreme head pose (yaw, pitch, roll)"""
        yaw, pitch, roll = map(abs, face.pose)
        return FaceEmbedder.normalize(yaw + pitch + roll, 0, 90)

    # --------------------------------------
    # Face Cropper with Margin
    # --------------------------------------
    def crop_face(self, img, face, name):
        """Crop face region with margin and save to disk"""
        h, w = img.shape[:2]
        x1, y1, x2, y2 = map(int, face.bbox)
        bw, bh = x2 - x1, y2 - y1

        px, py = int(bw * self.crop_margin), int(bh * self.crop_margin)
        cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
        cx2, cy2 = min(w, x2 + px), min(h, y2 + py)

        crop = img[cy1:cy2, cx1:cx2]
        full_path = os.path.join(self.crop_dir, name)
        cv2.imwrite(full_path, crop)
        
        # Return path relative to static folder for web serving
        relative_path = full_path.replace("\\", "/")  # Convert backslashes to forward slashes
        return crop, relative_path

    # --------------------------------------
    # Face Presence Check (Haar Cascade)
    # --------------------------------------
    def has_human_face_haar(self, img, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)):
        """Quick face presence check using OpenCV Haar Cascade."""
        if img is None:
            return False

        # If the cascade failed to load for any reason, don't block the pipeline.
        if self._haar_face_cascade.empty():
            return True

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self._haar_face_cascade.detectMultiScale(
            gray,
            scaleFactor=scaleFactor,
            minNeighbors=minNeighbors,
            minSize=minSize,
        )
        return len(faces) > 0

    # --------------------------------------
    # Child or Adult Heuristic
    # --------------------------------------
    @staticmethod
    def child_or_adult(face, img_shape):
        """Detect likely child vs adult based on face size ratio"""
        face_size = face.bbox[2] - face.bbox[0]
        ratio = face_size / img_shape[1]
        return (
            "Identity based features quality"
            if ratio < 0.18
            else "Identity based features quality"
        )

    # --------------------------------------
    # Final Quality Score Computation
    # --------------------------------------
    def compute_quality(self, face, embeddings):
        """
        Compute overall quality score from multiple components:
        - 40% Self-Consistency (SC)
        - 30% Embedding Signal Strength (ESS)
        - 20% Detection Score
        - -10% Pose Penalty
        """
        ess = self.embedding_signal_strength(face.embedding)
        ess_n = self.normalize(ess, 0.0005, 0.005)

        sc = self.self_consistency_score(embeddings)
        sc_n = self.normalize(sc, 0.6, 0.95)

        det_n = self.normalize(face.det_score, 0.3, 0.9)
        pose_p = self.pose_penalty(face)

        quality = (
            0.40 * sc_n +
            0.30 * ess_n +
            0.20 * det_n -
            0.10 * pose_p
        )
        return float(np.clip(quality, 0, 1))

    # --------------------------------------
    # Quality Label Generator
    # --------------------------------------
    @staticmethod
    def quality_label(q):
        """Convert quality score to human-readable label"""
        if q is None:
            return "REJECTED (quality could not be computed)"
        if q >= 0.80:
            return "Excellent (safe for verification)"
        elif q >= 0.60:
            return "Acceptable Facial Features"
        elif q >= 0.40:
            return "Risky (Some Facial Features are Damaged)"
        else:
            return "REJECTED (distorted / damaged Facial Features Detected)"

    # --------------------------------------
    # Enhanced Embedding Extraction with Quality
    # --------------------------------------
    def get_embedding(self, image_path, verbose=True, quality_check=True, tag="face", haar_check=True):
        """
        Extract face embedding with advanced quality assessment
        
        Parameters
        ----------
        image_path : str
            Path to the image file
        verbose : bool
            Print warnings and info
        quality_check : bool
            Enable self-consistency quality checks (flip + re-detect)
        tag : str
            Name for cropped face file
            
        Returns
        -------
        dict with keys:
            - embedding: normalized 512-D vector
            - quality: quality score (0-1)
            - quality_label: human-readable quality
            - crop_path: path to cropped face
            - age_hint: child/adult classification
            - det_score: detection confidence
            - pose: (yaw, pitch, roll)
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"[ERROR] Cannot load image: {image_path}")

        # Fast rejection: no obvious human face in the image
        if haar_check and not self.has_human_face_haar(img):
            raise ValueError("No Usable human face detected")

        faces = self.app.get(img)

        # -------- NO FACE DETECTED --------
        if len(faces) == 0:
            # Keep behavior consistent with pipeline error handling
            raise ValueError(
                f"REJECTED (damaged Facial Features Detected)in image: {image_path}"
            )

        # -------- MULTIPLE FACES DETECTED --------
        if len(faces) > 1 and verbose:
            print(
                f"[WARNING] Multiple faces detected ({len(faces)}) in: {image_path} "
                f"using the largest face"
            )

        # Use largest detected face
        face = max(faces, key=lambda x: x.bbox[2] * x.bbox[3])

        # Crop face
        crop, crop_path = self.crop_face(img, face, f"{tag}.jpg")

        # Presentation-only color classification and display score
        color_type = self.classify_color_type(crop)

        # Initialize embedding list
        emb_list = [self.l2norm(face.embedding)]

        # Self-consistency check with horizontal flip
        if quality_check:
            flip = cv2.flip(crop, 1)
            f_faces = self.app.get(flip)
            if len(f_faces) == 1:
                emb_list.append(self.l2norm(f_faces[0].embedding))

        # Compute quality
        quality = self.compute_quality(face, emb_list)
        display_score = self.quality_display_score(quality, color_type)

        # Age hint
        age_hint = self.child_or_adult(face, img.shape)

        # Return comprehensive result
        return {
            "embedding": emb_list[0].astype("float32"),
            "quality": quality,
            "quality_label": self.quality_label(quality),
            "display_score": display_score,
            "color_type": color_type,
            "crop_path": crop_path,
            "age_hint": age_hint,
            "det_score": float(face.det_score),
            "pose": tuple(map(float, face.pose))
        }
