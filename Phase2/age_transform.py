import warnings

# Suppress noisy deprecation warnings from dependencies that are out of our control.
warnings.filterwarnings("ignore", category=UserWarning, module="face_recognition_models")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"You are using `torch\.load` with `weights_only=False`"
)

import cv2
import numpy as np

from Phase2.pipelinemain import AgeTransformer
from Phase2.Preprocess.postprocess import FaceAgeProcessor


class FullAgePipeline:
    """
    Combines:
    1) AgeTransformer (aging / de-aging)
    2) FaceAgeProcessor (face parsing + age mask)

    Final output → NumPy image (BGR)
    """

    def __init__(
        self,
        age_model_path="Phase2/Preprocess/79999_iter.pth",
        device="cuda"
    ):
        self.transformer = AgeTransformer()
        self.face_processor = FaceAgeProcessor(
            model_path=age_model_path
        )

    def run(
        self,
        input_image_path,
        source_age,
        target_age,
        final_output_path="final.png",
        aged_intermediate_path=None
    ):
        """
        Returns:
            final_img (BGR NumPy array)
        """

        # --------------------------------------------------
        # STEP 1: AGE / DE-AGE IMAGE
        # --------------------------------------------------
        aged_img = self.transformer.run(
            input_image_path=input_image_path,
            source_age=source_age,
            target_age=target_age,
            aged_output_path=aged_intermediate_path,
            image_output_path=None   # return NumPy
        )

        if aged_img is None:
            raise RuntimeError("Age transformation failed")

        # Persist the transformed image to a known path for the face parser
        parse_path = aged_intermediate_path or "aged_tmp.png"
        cv2.imwrite(parse_path, aged_img)

        # --------------------------------------------------
        # STEP 2: FACE PARSING + AGE MASK
        # --------------------------------------------------
        rgba, mask = self.face_processor.process_image(
            parse_path,
            target_age=target_age
        )

        # --------------------------------------------------
        # STEP 3: SAVE FINAL IMAGE
        # --------------------------------------------------
        # `FaceAgeProcessor` builds the RGB head canvas (RGB + alpha).
        # OpenCV saves in BGR, so swap channels explicitly.
        rgb = rgba[:, :, :3]
        final_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(final_output_path, final_bgr)

        return final_bgr


