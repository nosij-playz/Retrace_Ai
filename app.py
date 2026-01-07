from flask import Flask, render_template, request, jsonify
import os
import cv2
import dlib
from werkzeug.utils import secure_filename

from Phase1.pipeline import FaceVerifier
from Preprocess.rembg import BackgroundRemover
from Phase2.Aging.PreostProcess.checkup import FaceQualityValidator
from Phase2.age_transform import FullAgePipeline
from Phase2.Preprocess.sharpen import ImageSharpening

app = Flask(__name__)

# -------------------------------
# Config
# -------------------------------
UPLOAD_FOLDER = "static/uploads"
BG_FOLDER = os.path.join(UPLOAD_FOLDER, "bg_removed")
TRANSFORMED_FOLDER = os.path.join(UPLOAD_FOLDER, "transformed")
PREDICTOR_PATH = "Phase2/Aging/PreostProcess/shape_predictor_68_face_landmarks.dat"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(BG_FOLDER, exist_ok=True)
os.makedirs(TRANSFORMED_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["BG_FOLDER"] = BG_FOLDER
app.config["TRANSFORMED_FOLDER"] = TRANSFORMED_FOLDER

verifier = FaceVerifier()
bg_remover = BackgroundRemover()
age_pipeline = FullAgePipeline()
sharpener = ImageSharpening(strength=1.2)


# -------------------------------
# Helpers
# -------------------------------
def crop_face(image_path, predictor_path, output_path, margin=0.2):
    """Crop the largest detected face with optimized margin for full head including hair. Returns (path, error_message)."""
    img = cv2.imread(image_path)
    if img is None:
        return None, "Image not found for cropping"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    faces = detector(gray, 1)

    if len(faces) == 0:
        return None, "Face not detected for cropping or too small face was found"

    face = max(faces, key=lambda r: r.width() * r.height())
    predictor = dlib.shape_predictor(predictor_path)
    lm = predictor(gray, face)

    # Extract facial landmarks and compute tight bounding box
    xs = [p.x for p in lm.parts()]
    ys = [p.y for p in lm.parts()]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    
    face_w = x2 - x1
    face_h = y2 - y1

    h, w = img.shape[:2]
    # Optimized margin: Add margins to show full head with hair but exclude neck
    # More space on top for hair, less on bottom to avoid neck
    dx_left = int(face_w * margin)
    dx_right = int(face_w * margin)
    dy_top = int(face_h * 0.45)  # More space for hair (was 0.25)
    dy_bottom = int(face_h * 0.15)  # Less space at bottom to avoid neck (was 0.30)
    
    x1 = max(0, x1 - dx_left)
    y1 = max(0, y1 - dy_top)
    x2 = min(w, x2 + dx_right)
    y2 = min(h, y2 + dy_bottom)

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None, "Cropping failed (empty region)"

    cv2.imwrite(output_path, crop)
    return output_path.replace("\\", "/"), None

# -------------------------------
# Homepage
# -------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# -------------------------------
# Transform page
# -------------------------------
@app.route("/transform", methods=["GET", "POST"])
def transform():
    validation = None
    image_url = None  # display (cropped if available)
    original_image_url = None  # always the original upload for backend use
    transformed_image_url = None  # transformed image output
    error = None
    crop_message = None
    transform_error = None
    source_age = request.form.get("source_age")
    target_age = request.form.get("target_age")

    if request.method == "POST":
        photo = request.files.get("photo")
        photo_path_hidden = request.form.get("photo")  # Hidden field from age form

        if photo:
            # New file upload
            if not photo or photo.filename == "":
                error = "Please upload an image to process."
            else:
                filename = secure_filename(photo.filename)
                saved_path = os.path.join(UPLOAD_FOLDER, filename)
                photo.save(saved_path)
                original_image_url = saved_path.replace("\\", "/")

                # Validate on the ORIGINAL image (no crop) to avoid masking issues
                path_for_validation = saved_path

                # Attempt crop purely for display convenience (not used for backend logic)
                cropped_out = os.path.join(UPLOAD_FOLDER, f"crop_{filename}")
                cropped_path, crop_err = crop_face(saved_path, PREDICTOR_PATH, cropped_out)
                if cropped_path:
                    image_url = cropped_path
                else:
                    image_url = saved_path.replace("\\", "/")
                    crop_message = crop_err

                try:
                    validator = FaceQualityValidator(
                        image_path=path_for_validation,
                        predictor_path=PREDICTOR_PATH,
                    )
                    validation = validator.run()

                    # Handle boolean returns gracefully
                    if validation is False:
                        validation = {
                            "result": False,
                            "reason": crop_message or "Face not detected or image not suitable",
                        }
                    elif validation and not validation.get("result"):
                        validation["reason"] = validation.get("reason") or crop_message or "Image not usable"
                    elif validation and crop_message:
                        validation["note"] = crop_message
                except Exception as exc:
                    error = f"Validation failed: {exc}"
        elif photo_path_hidden:
            # Age transformation using existing image
            original_image_url = photo_path_hidden
            path_for_validation = photo_path_hidden
            
            try:
                validator = FaceQualityValidator(
                    image_path=path_for_validation,
                    predictor_path=PREDICTOR_PATH,
                )
                validation = validator.run()

                # Handle boolean returns gracefully
                if validation is False:
                    validation = {
                        "result": False,
                        "reason": "Face not detected or image not suitable",
                    }
                elif validation and not validation.get("result"):
                    validation["reason"] = validation.get("reason") or "Image not usable"
            except Exception as exc:
                error = f"Validation failed: {exc}"
        else:
            error = "Please upload an image to process."

        # If validation passed and age parameters are provided, run age transformation
        if validation and validation.get("result") and source_age and target_age and original_image_url:
            try:
                source_age_int = int(source_age)
                target_age_int = int(target_age)
                
                # Create output filename for transformed image
                base_filename = os.path.splitext(os.path.basename(original_image_url))[0]
                transformed_filename = f"{base_filename}_age_{source_age_int}_to_{target_age_int}.png"
                transformed_output_path = os.path.join(TRANSFORMED_FOLDER, transformed_filename)
                
                # Run age transformation pipeline on ORIGINAL (non-cropped) image
                final_image = age_pipeline.run(
                    input_image_path=original_image_url,
                    source_age=source_age_int,
                    target_age=target_age_int,
                    final_output_path=transformed_output_path,
                    aged_intermediate_path=os.path.join(TRANSFORMED_FOLDER, f"{base_filename}_intermediate.png")
                )
                
                transformed_image_url = transformed_output_path.replace("\\", "/")
            except ValueError:
                transform_error = "Invalid age values. Please enter valid numbers."
            except Exception as exc:
                transform_error = f"Age transformation failed: {str(exc)}"

    return render_template(
        "transform.html",
        validation=validation,
        image_url=image_url,
        original_image_url=original_image_url,
        transformed_image_url=transformed_image_url,
        error=error,
        transform_error=transform_error,
        crop_message=crop_message,
        source_age=source_age,
        target_age=target_age,
    )


# -------------------------------
# Verification page
# -------------------------------
@app.route("/verify", methods=["GET", "POST"])
def verify():
    result = None

    if request.method == "POST":
        child = request.files.get("child")
        adult = request.files.get("adult")

        child_bg = request.form.get("child_bg")
        adult_bg = request.form.get("adult_bg")

        if child and adult:
            child_name = secure_filename(child.filename)
            adult_name = secure_filename(adult.filename)

            child_path = os.path.join(UPLOAD_FOLDER, child_name)
            adult_path = os.path.join(UPLOAD_FOLDER, adult_name)

            child.save(child_path)
            adult.save(adult_path)

            # ✅ Use bg-removed images if available
            final_child = child_bg if child_bg else child_path
            final_adult = adult_bg if adult_bg else adult_path

            result = verifier.verify(final_child, final_adult)

            # Ensure correct images shown in result
            result["img1_path"] = final_child
            result["img2_path"] = final_adult

    # Always render verify.html
    return render_template("verify.html", result=result)

# -------------------------------
# Background removal API
# -------------------------------
@app.route("/bgremove", methods=["POST"])
def bgremove():
    image = request.files.get("image")

    if not image:
        return jsonify({
            "status": "error",
            "message": "No image provided"
        }), 400

    filename = secure_filename(image.filename)
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(input_path)

    output_name = f"bg_{filename}"
    output_path = os.path.join(BG_FOLDER, output_name)

    try:
        bg_remover.remove_background(input_path, output_path)
        return jsonify({
            "status": "ok",
            "output": output_path.replace("\\", "/")
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# -------------------------------
# Image Sharpening API
# -------------------------------
@app.route("/sharpen", methods=["POST"])
def sharpen_image():
    image = request.files.get("image")

    if not image:
        return jsonify({
            "status": "error",
            "message": "No image provided"
        }), 400

    filename = secure_filename(image.filename)
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    image.save(input_path)

    output_name = f"sharp_{filename}"
    output_path = os.path.join(UPLOAD_FOLDER, output_name)

    try:
        img = cv2.imread(input_path)
        if img is None:
            raise ValueError("Failed to read image")
        
        sharpened_img = sharpener.sharpen(img)
        cv2.imwrite(output_path, sharpened_img)
        
        return jsonify({
            "status": "ok",
            "output": output_path.replace("\\", "/")
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
