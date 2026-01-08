import numpy as np
import PIL
import PIL.Image
import scipy
import scipy.ndimage
import dlib
import cv2

# Pillow>=10 removed ANTIALIAS/BILINEAR; keep backward compat
_Resampling = getattr(PIL.Image, "Resampling", PIL.Image)


# ============================================================
# SCREENSHOT NORMALIZATION PIPELINE
# ============================================================

def is_likely_screenshot(img_np):
    """
    Detect "screenshot-like" images using high-frequency noise detection.
    Screenshots often have exaggerated edges from sharpening and aliasing.
    
    :param img_np: numpy array (RGB or BGR)
    :return: bool - True if likely a screenshot
    """
    # Convert to grayscale if needed
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
    
    # High-frequency noise detection via Laplacian
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    hf_energy = np.mean(np.abs(lap))
    
    # Screenshots often have exaggerated edges
    is_screenshot = hf_energy > 12.0
    
    return is_screenshot


def normalize_screenshot(img):
    """
    Normalize screenshot to behave like an original photo for face detection.
    This suppresses aliasing, smooths unnatural edges, and stabilizes landmarks.
    
    DOES NOT recreate the original - makes screenshot behave like one.
    
    :param img: PIL Image
    :return: PIL Image (normalized)
    """
    img_np = np.array(img).astype(np.float32)
    
    # 1. Remove oversharpening / aliasing
    img_np = cv2.GaussianBlur(img_np, (0, 0), sigmaX=0.6, sigmaY=0.6)
    
    # 2. Mild downscale + upscale (re-photograph effect)
    # This restores continuous gradients
    h, w = img_np.shape[:2]
    img_np = cv2.resize(img_np, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    img_np = cv2.resize(img_np, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # 3. Normalize contrast
    img_np = cv2.normalize(img_np, None, 0, 255, cv2.NORM_MINMAX)
    
    return PIL.Image.fromarray(img_np.astype(np.uint8))


def load_image_safely(filepath):
    """
    Load image and normalize if it's a screenshot.
    This is the integration point for the normalization pipeline.
    
    :param filepath: str - path to image
    :return: PIL Image (normalized if screenshot)
    """
    img = PIL.Image.open(filepath).convert("RGB")
    img_np = np.array(img)
    
    if is_likely_screenshot(img_np):
        print("ℹ️  Screenshot detected — normalizing before face processing")
        img = normalize_screenshot(img)
    
    return img


def get_landmark(filepath, predictor):
    """
    Get facial landmarks with dlib.
    Automatically normalizes screenshots before detection.
    
    :param filepath: str - path to image
    :param predictor: dlib shape predictor
    :return: np.array shape=(68, 2)
    """
    detector = dlib.get_frontal_face_detector()
    
    # Load and normalize if screenshot (BEFORE face detection)
    img_pil = load_image_safely(filepath)
    
    # Save normalized version temporarily for align_face to use
    import os
    import tempfile
    normalized_path = filepath
    
    # Check if normalization happened by comparing with original
    original_img = PIL.Image.open(filepath).convert("RGB")
    if np.array_equal(np.array(img_pil), np.array(original_img)) is False:
        # Screenshot was normalized - save it
        temp_dir = tempfile.gettempdir()
        base_name = os.path.basename(filepath)
        normalized_path = os.path.join(temp_dir, f"normalized_{base_name}")
        img_pil.save(normalized_path)
        print(f"   Saved normalized image for processing: {normalized_path}")
    
    # Convert PIL to numpy for dlib
    img = np.array(img_pil)
    
    dets = detector(img, 1)

    if len(dets) == 0:
        raise RuntimeError("No face detected in image")

    for k, d in enumerate(dets):
        shape = predictor(img, d)

    t = list(shape.parts())
    a = []
    for tt in t:
        a.append([tt.x, tt.y])
    lm = np.array(a)
    
    # Validate landmarks are reasonable
    if np.any(np.isnan(lm)) or np.any(np.isinf(lm)):
        raise RuntimeError("Invalid landmarks detected (NaN or Inf)")
    
    # Check landmarks are within image bounds
    if np.any(lm < 0) or np.any(lm[:, 0] >= img.shape[1]) or np.any(lm[:, 1] >= img.shape[0]):
        raise RuntimeError("Landmarks outside image bounds")
    
    return lm


def align_face(filepath, predictor):
    """
    Align face with screenshot normalization.
    Automatically detects and normalizes screenshots BEFORE alignment.
    
    :param filepath: str - path to image
    :param predictor: dlib shape predictor
    :return: PIL Image
    """

    lm = get_landmark(filepath, predictor)

    lm_chin = lm[0: 17]  # left-right
    lm_eyebrow_left = lm[17: 22]  # left-right
    lm_eyebrow_right = lm[22: 27]  # left-right
    lm_nose = lm[27: 31]  # top-down
    lm_nostrils = lm[31: 36]  # top-down
    lm_eye_left = lm[36: 42]  # left-clockwise
    lm_eye_right = lm[42: 48]  # left-clockwise
    lm_mouth_outer = lm[48: 60]  # left-clockwise
    lm_mouth_inner = lm[60: 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Load normalized image (screenshot normalization already applied in get_landmark)
    img = load_image_safely(filepath)

    output_size = 256
    transform_size = 256
    enable_padding = True

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, _Resampling.LANCZOS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), _Resampling.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), _Resampling.LANCZOS)

    # Return aligned image.
    return img
