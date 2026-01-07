import cv2
import dlib
import numpy as np
import os


class AgeConnector:
    """
    Connects de-aged and aged face images using
    landmark-aware, region-specific blending.
    """

    # ======================================================
    # INIT
    # ======================================================
    def __init__(self, predictor_path):
        if not os.path.exists(predictor_path):
            raise FileNotFoundError(predictor_path)

        self.detector  = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    # ======================================================
    # UTILITIES
    # ======================================================
    @staticmethod
    def read_img(path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Failed to load image: {path}")
        return img

    def get_landmarks(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        if not faces:
            raise RuntimeError("No face detected")

        face = max(faces, key=lambda r: r.width() * r.height())
        shape = self.predictor(gray, face)
        return np.array([(p.x, p.y) for p in shape.parts()], np.float32)

    def align(self, src, dst):
        lm_s = self.get_landmarks(src)
        lm_d = self.get_landmarks(dst)

        src_pts = np.array([
            lm_s[36:42].mean(0),
            lm_s[42:48].mean(0),
            lm_s[30]
        ])
        dst_pts = np.array([
            lm_d[36:42].mean(0),
            lm_d[42:48].mean(0),
            lm_d[30]
        ])

        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
        return cv2.warpAffine(
            src, M, (dst.shape[1], dst.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )

    # ======================================================
    # AGE → AGED DOMINANT
    # ======================================================
    @staticmethod
    def age_alpha(target_age):
        return float(np.clip(
            np.interp(target_age, [20, 22, 25], [0.20, 0.65, 0.95]),
            0.0, 1.0
        ))

    # ======================================================
    # MASKS
    # ======================================================
    @staticmethod
    def face_mask(h, w):
        Y, X = np.ogrid[:h, :w]
        cx, cy = w // 2, h // 2
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        mask = 1.0 - dist / dist.max()
        return cv2.GaussianBlur(mask, (0, 0), 25)[:, :, None]

    @staticmethod
    def identity_mask(lm, h, w):
        mask = np.zeros((h, w), np.float32)
        idxs = np.concatenate([
            np.arange(36, 48),  # eyes
            np.arange(27, 36),  # nose
            np.arange(48, 68)   # lips
        ])
        pts = lm[idxs].astype(np.int32)
        cv2.fillConvexPoly(mask, pts, 1.0)
        return cv2.GaussianBlur(mask, (0, 0), 15)[:, :, None]

    @staticmethod
    def jaw_mask(lm, h, w):
        mask = np.zeros((h, w), np.float32)
        jaw = lm[0:17].astype(np.int32)
        cv2.fillConvexPoly(mask, jaw, 1.0)
        return cv2.GaussianBlur(mask, (0, 0), 25)[:, :, None]

    @staticmethod
    def beard_mask(lm, h, w):
        mask = np.zeros((h, w), np.float32)

        nose = lm[30]
        mouth = lm[48:68]
        jaw = lm[6:11]

        y1 = int(nose[1])
        y2 = int(jaw[:, 1].max() + 25)
        y1 = max(0, y1)
        y2 = min(h, y2)

        mask[y1:y2, :] = 1.0

        lip_y = int(mouth[:, 1].mean())
        mask[y1:lip_y + 5, :] *= 0.65

        mask = cv2.GaussianBlur(mask, (0, 0), 35)
        return np.clip(mask, 0, 1)[:, :, None]

    # ======================================================
    # FINAL BLEND (JAW FROM AGED)
    # ======================================================
    def blend_faces(self, deaged, aged, target_age):
        alpha = self.age_alpha(target_age)

        aged = self.align(aged, deaged)

        de = deaged.astype(np.float32)
        ag = aged.astype(np.float32)

        h, w = de.shape[:2]
        lm = self.get_landmarks(deaged)

        fmask = self.face_mask(h, w)
        imask = self.identity_mask(lm, h, w)
        jmask = self.jaw_mask(lm, h, w)
        bmask = self.beard_mask(lm, h, w)

        # Global mix
        face_mix = de * (1 - alpha) + ag * alpha

        # Identity (age-biased)
        identity_alpha = 0.6 * alpha
        identity = (de * (1 - identity_alpha) + ag * identity_alpha) * imask

        # Jaw = fully aged
        jaw = ag * jmask

        # Beard
        beard_alpha = min(1.0, alpha * 1.15)
        beard = (de * (1 - beard_alpha) + ag * beard_alpha) * bmask
        beard = cv2.addWeighted(beard, 0.75, face_mix, 0.25, 0)

        result = (
            face_mix * fmask * (1 - imask) * (1 - jmask) * (1 - bmask) +
            identity +
            jaw +
            beard +
            de * (1 - fmask)
        )

        return np.clip(result, 0, 255).astype(np.uint8)

    # ======================================================
    # PUBLIC API
    # ======================================================
    def connect(self, deaged_path, aged_path, target_age, output_path):
        deaged = self.read_img(deaged_path)
        aged   = self.read_img(aged_path)

        out = self.blend_faces(deaged, aged, target_age)
        cv2.imwrite(output_path, out)

        print("✅ Output saved:", output_path)
