import cv2
import numpy as np
from Phase2.Aging.Perform.age import age_image
from Phase2.Aging.PreostProcess.enhance import HeadEnhancer


class FaceAgingHeadaPipeline:
    """
    End-to-end pipeline:
    1. Face aging (PIL output)
    2. PIL → OpenCV conversion
    3. Full-head enhancement with landmark safety
    """

    def __init__(
        self,
        predictor_path,
        head_radius_mult=2.8,
        head_top_shift=0.35,
        output_path="output_head_enhanced.jpg"
    ):
        self.predictor_path = predictor_path
        self.head_radius_mult = head_radius_mult
        self.head_top_shift = head_top_shift
        self.output_path = output_path

        # Initialize enhancer once (efficient)
        self.enhancer = HeadEnhancer(
            predictor_path=self.predictor_path,
            head_radius_mult=self.head_radius_mult,
            head_top_shift=self.head_top_shift
        )

    # --------------------------------------------------
    # Internal utility: PIL → OpenCV
    # --------------------------------------------------
    @staticmethod
    def _pil_to_cv(img_pil):
        img = np.array(img_pil)              # RGB
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # --------------------------------------------------
    # Main execution
    # --------------------------------------------------
    def run(
        self,
        input_image_path,
        source_age,
        target_age,
        aged_output_path="aged_photo.jpg"
    ):
        # -------------------------------
        # 1. Age the face
        # -------------------------------
        pil_result = age_image(
            input_image_path=input_image_path,
            source_age=source_age,
            target_age=target_age+10,
            output_path=aged_output_path
        )

        if pil_result is None:
            raise RuntimeError("❌ Age model returned None")

        # -------------------------------
        # 2. Convert to OpenCV
        # -------------------------------
        img_cv = self._pil_to_cv(pil_result)

        # -------------------------------
        # 3. Head enhancement
        # -------------------------------
        enhanced = self.enhancer.process(img_cv)

        # -------------------------------
        # 4. Save output
        # -------------------------------
        output_path = self.output_path or aged_output_path
        if output_path:
            cv2.imwrite(output_path, enhanced)

        print("✅ Full head preserved + safe enhancement completed")

        return enhanced
