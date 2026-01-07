import cv2
import dlib
import numpy as np
import math
import sys


class FaceQualityValidator:
    def __init__(
        self,
        image_path,
        predictor_path,
        center_thr=0.15,
        blur_thr=100,
        roll_thr=11.0,
        light_low=60,
        light_high=200,
        appearance_sat_thr=15,
        mouth_open_thr=0.25,
        eye_ear_thr=0.20,
        nose_edge_thr=12,
        mouth_texture_thr=10,
    ):
        self.image_path = image_path
        self.predictor_path = predictor_path

        # thresholds
        self.CENTER_THR = center_thr
        self.BLUR_THR = blur_thr
        self.ROLL_THR = roll_thr
        self.LIGHT_LOW = light_low
        self.LIGHT_HIGH = light_high
        self.APPEARANCE_SAT_THR = appearance_sat_thr
        self.MOUTH_OPEN_THR = mouth_open_thr
        self.EYE_EAR_THR = eye_ear_thr
        self.NOSE_EDGE_THR = nose_edge_thr
        self.MOUTH_TEXTURE_THR = mouth_texture_thr

        # init models
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.predictor_path)

    # ==============================
    # HELPERS
    # ==============================
    @staticmethod
    def shape_to_np(shape):
        return np.array([(p.x, p.y) for p in shape.parts()], dtype=np.float32)

    @staticmethod
    def angle_between(p1, p2):
        return abs(math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])))

    @staticmethod
    def mouth_open_ratio(lm):
        upper_lip = lm[62]
        lower_lip = lm[66]
        left_mouth = lm[48]
        right_mouth = lm[54]
        return np.linalg.norm(upper_lip - lower_lip) / np.linalg.norm(left_mouth - right_mouth)

    @staticmethod
    def eye_aspect_ratio(eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C)

    @staticmethod
    def region_variance(gray, points, pad=6):
        x, y = points.mean(axis=0).astype(int)
        h, w = gray.shape
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w, x + pad), min(h, y + pad)
        roi = gray[y1:y2, x1:x2]
        return roi.var() if roi.size > 0 else 0

    # ==============================
    # MAIN PIPELINE
    # ==============================
    def run(self):
        img = cv2.imread(self.image_path)
        if img is None:
            raise RuntimeError("❌ Image not found")

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Appearance quality
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        appearance_ok = hsv[:, :, 1].mean() > self.APPEARANCE_SAT_THR

        # Face detection
        faces = self.detector(gray)
        if len(faces) == 0:
            print("❌ FACE NOT DETECTED Either Face is too small or Covered")
            return False

        face = max(faces, key=lambda r: r.width() * r.height())
        shape = self.predictor(gray, face)
        lm = self.shape_to_np(shape)

        # Centering
        face_cx = (face.left() + face.right()) / 2
        face_cy = (face.top() + face.bottom()) / 2
        center_offset = np.linalg.norm([face_cx - w / 2, face_cy - h / 2]) / w
        centered = center_offset < self.CENTER_THR

        # Alignment
        left_eye_c = lm[36:42].mean(axis=0)
        right_eye_c = lm[42:48].mean(axis=0)
        roll = self.angle_between(left_eye_c, right_eye_c)
        frontal = roll <= self.ROLL_THR

        # Blur
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharp = blur_score > self.BLUR_THR

        # Lighting
        mean_luma = gray.mean()
        well_lit = self.LIGHT_LOW < mean_luma < self.LIGHT_HIGH

        # Mouth
        mouth_ratio = self.mouth_open_ratio(lm)
        mouth_closed = mouth_ratio < self.MOUTH_OPEN_THR

        # Eyes
        ear_left = self.eye_aspect_ratio(lm[36:42])
        ear_right = self.eye_aspect_ratio(lm[42:48])
        eyes_open = (ear_left > self.EYE_EAR_THR) and (ear_right > self.EYE_EAR_THR)

        # Mask
        nose_var = self.region_variance(gray, lm[27:36])
        mouth_var = self.region_variance(gray, lm[48:60])
        no_mask = (nose_var > self.NOSE_EDGE_THR) and (mouth_var > self.MOUTH_TEXTURE_THR)

        # Prepare a dictionary with the original printed values (strings/metrics), not booleans
        overall_pass = (
            appearance_ok
            and centered
            and frontal
            and sharp
            and well_lit
            and mouth_closed
            and eyes_open
            and no_mask
        )

        passed = {
            "appearance_quality": "OK" if appearance_ok else "NOT SUITABLE",
            "centered_face": "CENTERED" if centered else "OFF_CENTER",
            "center_offset": float(center_offset),
            "frontal_alignment": "FRONTAL" if frontal else "NON_FRONTAL",
            "roll_deg": float(roll),
            "sharpness": "SHARP" if sharp else "BLURRY",
            "blur_score": float(blur_score),
            "lighting": "WELL_LIT" if well_lit else "POOR",
            "mean_luma": float(mean_luma),
            "mouth_state": "CLOSED" if mouth_closed else "OPEN",
            "mouth_ratio": float(mouth_ratio),
            "eyes_state": "OPEN" if eyes_open else "CLOSED",
            "eye_left": float(ear_left),
            "eye_right": float(ear_right),
            "face_covering": "NOT DETECTED" if no_mask else "DETECTED",
            "result": True if overall_pass else False,
        }

        # ==============================
        # REPORT
        # ==============================
        print("\n====== FACIAL IMAGE QUALITY REPORT ======")
        print(f"Appearance Quality : {'OK' if appearance_ok else 'NOT SUITABLE'}")
        print(f"Centered Face      : {centered} (offset={center_offset:.3f})")
        print(f"Frontal Alignment  : {frontal} ({roll:.2f}°)")
        print(f"Image Sharpness    : {sharp} (score={blur_score:.1f})")
        print(f"Lighting Condition : {well_lit} (mean={mean_luma:.1f})")
        print(f"Mouth State        : {'CLOSED' if mouth_closed else 'OPEN'} (ratio={mouth_ratio:.2f})")
        print(f"Eyes State         : {'OPEN' if eyes_open else 'CLOSED'} (EYE L={ear_left:.2f}, R={ear_right:.2f})")
        print(f"Face Covering      : {'NOT DETECTED' if no_mask else 'DETECTED'}")
        print("--------------------------------")

        if passed["result"]:
            print("✅ BEST_FOR_AGING")
        else:
            print("❌ REJECT / Image Cannot Be Encoded")

        return passed
