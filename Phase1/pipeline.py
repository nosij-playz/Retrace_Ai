import numpy as np
import warnings
import os
from contextlib import redirect_stdout, redirect_stderr

from Phase1.faceembedding import FaceEmbedder
from Phase1.featureextract import FaceFeatureExtractor
from Phase1.calibrator import FaceSimilarityGT


# --------------------------------------
# GLOBAL SILENCE
# --------------------------------------
warnings.filterwarnings("ignore")
os.environ["ORT_LOGGING_LEVEL"] = "4"
os.environ["XGBOOST_VERBOSE"] = "0"


class FaceVerifier:
    def __init__(self, match_thr=0.30, nomatch_thr=0.10, quality_reject_thr=0.40):
        """
        Initialize face verifier with threshold-based logic
        
        Parameters
        ----------
        match_thr : float
            Cosine similarity >= this → MATCH
        nomatch_thr : float
            Cosine similarity <= this → NO_MATCH  
        quality_reject_thr : float
            Quality score < this → UNUSABLE
        """
        with open(os.devnull, "w") as f, redirect_stdout(f), redirect_stderr(f):
            self.embedder = FaceEmbedder()
            self.extractor = FaceFeatureExtractor()
            self.calibrator = FaceSimilarityGT(
                match_thr=match_thr,
                nomatch_thr=nomatch_thr,
                quality_reject_thr=quality_reject_thr
            )

    def verify(self, img1_path, img2_path):
        result = {
            "status": "error",
            "message": "",
            "img1_path": img1_path,
            "img2_path": img2_path
        }

        try:
            # --------------------------------------
            # 1️⃣ EMBEDDING EXTRACTION WITH QUALITY
            # --------------------------------------
            with open(os.devnull, "w") as f, redirect_stdout(f), redirect_stderr(f):
                try:
                    emb_data1 = self.embedder.get_embedding(img1_path, tag="img1")
                except ValueError as e:
                    raise ValueError(f"Child image (img1): {e}")

                try:
                    emb_data2 = self.embedder.get_embedding(img2_path, tag="img2")
                except ValueError as e:
                    raise ValueError(f"Adult image (img2): {e}")

            emb1 = emb_data1["embedding"]
            emb2 = emb_data2["embedding"]
            q1 = emb_data1["quality"]
            q2 = emb_data2["quality"]
            q1_display = emb_data1.get("display_score")
            q2_display = emb_data2.get("display_score")
            img1_color_type = emb_data1.get("color_type")
            img2_color_type = emb_data2.get("color_type")

            # Lowest RAW quality (drives decisions)
            if q1 <= q2:
                worst_raw_quality = q1
                worst_raw_image = "img1"
                worst_raw_label = emb_data1["quality_label"]
            else:
                worst_raw_quality = q2
                worst_raw_image = "img2"
                worst_raw_label = emb_data2["quality_label"]

            # Lowest DISPLAY quality (presentation only)
            # If display scores are missing, fall back to raw ordering.
            if q1_display is None or q2_display is None:
                worst_display_score = q1_display if worst_raw_image == "img1" else q2_display
                worst_display_image = worst_raw_image
            elif q1_display <= q2_display:
                worst_display_score = q1_display
                worst_display_image = "img1"
            else:
                worst_display_score = q2_display
                worst_display_image = "img2"

            worst_display_color_type = img1_color_type if worst_display_image == "img1" else img2_color_type

            quality_info = {
                "q1": q1,
                "q2": q2,
                "q1_display": q1_display,
                "q2_display": q2_display,
                # Backward-compatible keys (RAW drives decisions)
                "worst_quality": worst_raw_quality,
                "worst_image": worst_raw_image,
                "worst_label": worst_raw_label,
                "worst_raw_quality": worst_raw_quality,
                "worst_raw_image": worst_raw_image,
                "worst_raw_label": worst_raw_label,
                "worst_display_score": worst_display_score,
                "worst_display_image": worst_display_image,
                "worst_display_color_type": worst_display_color_type,
                "img1_age_hint": emb_data1["age_hint"],
                "img2_age_hint": emb_data2["age_hint"],
                "img1_color_type": img1_color_type,
                "img2_color_type": img2_color_type,
                "img1_crop": emb_data1["crop_path"],
                "img2_crop": emb_data2["crop_path"]
            }

            # --------------------------------------
            # 2️⃣ FEATURE EXTRACTION (8D)
            # --------------------------------------
            features_8 = self.extractor.extract(
                emb1, emb2, strict=True
            )

            # --------------------------------------
            # 3️⃣ USE FIRST 7 FEATURES
            # --------------------------------------
            features_7 = features_8[:7]

            # --------------------------------------
            # 4️⃣ THRESHOLD-BASED PREDICTION
            # --------------------------------------
            with open(os.devnull, "w") as f, redirect_stdout(f), redirect_stderr(f):
                pred = self.calibrator.predict(features_7, quality_info=quality_info)

            # --------------------------------------
            # SUCCESS RESULT
            # --------------------------------------
            result.update({
                "status": "ok",
                "probability": pred["probability"],
                "similarity": pred["similarity"],
                "decision": pred["decision"],
                "label": pred.get("label"),
                "quality_info": quality_info
            })

        except ValueError as e:
            # Known face / embedding errors
            result["message"] = str(e)

        except Exception as e:
            # Unknown failure
            result["message"] = f"[INTERNAL ERROR] {str(e)}"

        # --------------------------------------
        # FINAL OUTPUT
        # --------------------------------------
        if result["status"] == "ok":
            qi = result.get("quality_info") or {}
            print("\n===== FACE VERIFICATION RESULT =====")
            
            if result["decision"] == "UNUSABLE":
                print(f"Decision    : {result['decision']}")
                print(f"Reason      : Quality below threshold")
                worst_img = qi.get("worst_image", qi.get("worst_raw_image"))
                worst_q = qi.get("worst_quality", qi.get("worst_raw_quality"))
                worst_label = qi.get("worst_label", qi.get("worst_raw_label"))
                print(f"Worst Image : {worst_img}")
                if worst_q is not None:
                    print(f"Worst Qual. : {float(worst_q):.3f}")
                print(f"Quality     : {worst_label}")
            else:
                print(f"Decision    : {result['decision']}")
                print(f"Probability : {result['probability']:.4f}")
                print(f"Similarity  : {result['similarity']:.2f}%")
                print(f"Label       : {result['label']}")
                print(f"\n----- Quality Analysis -----")
                if "q1" in qi:
                    print(f"Image 1 Qual: {float(qi.get('q1')):.3f} | {qi.get('img1_age_hint')}")
                if "q2" in qi:
                    print(f"Image 2 Qual: {float(qi.get('q2')):.3f} | {qi.get('img2_age_hint')}")

                worst_img = qi.get("worst_image", qi.get("worst_raw_image"))
                worst_q = qi.get("worst_quality", qi.get("worst_raw_quality"))
                worst_label = qi.get("worst_label", qi.get("worst_raw_label"))
                if worst_q is not None:
                    print(f"Worst Image : {worst_img} ({float(worst_q):.3f})")
                else:
                    print(f"Worst Image : {worst_img}")
                print(f"Quality Desc: {worst_label}")
        else:
            print("\n===== FACE VERIFICATION FAILED =====")
            print(result["message"])

        return result
