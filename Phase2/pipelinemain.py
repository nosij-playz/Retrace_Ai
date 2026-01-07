import cv2
import shutil
from unittest import runner

from Phase2.Aging.pipeline import FaceAgingHeadaPipeline
from Phase2.Aging.PreostProcess.checkup import FaceQualityValidator
from Phase2.Aging.PreostProcess.enhance import HeadEnhancer
from Phase2.Deaging.pipelineb import FaceAgingHeaddPipeline
from Phase2.Deaging.deage import DeAgeRunner

PIPELINE_PREDICTOR = "Phase2/Aging/PreostProcess/shape_predictor_68_face_landmarks.dat"


class AgeTransformer:
    def __init__(
        self,
        predictor_path=PIPELINE_PREDICTOR,
        head_radius_mult=2.8,
        head_top_shift=0.35
    ):
        self.predictor_path = predictor_path
        self.head_radius_mult = head_radius_mult
        self.head_top_shift = head_top_shift

        self.apipeline = FaceAgingHeadaPipeline(
            predictor_path=predictor_path,
            head_radius_mult=head_radius_mult,
            head_top_shift=head_top_shift,
            output_path=None
        )

        self.dpipeline = FaceAgingHeaddPipeline(
            predictor_path=predictor_path,
            head_radius_mult=head_radius_mult,
            head_top_shift=head_top_shift,
            output_path=None
        )

        self.enhancer = HeadEnhancer(
            predictor_path=predictor_path,
            head_radius_mult=head_radius_mult,
            head_top_shift=head_top_shift
        )

    # --------------------------------------------------
    # MAIN ENTRY
    # --------------------------------------------------
    def run(
        self,
        input_image_path,
        source_age,
        target_age,
        aged_output_path=None,
        image_output_path=None
    ):
        """
        Returns:
            np.ndarray (BGR image)
        """

        # -------------------------------
        # Load input image
        # -------------------------------
        img = cv2.imread(input_image_path)
        if img is None:
            raise FileNotFoundError(input_image_path)

        # -------------------------------
        # CASE 1: SAME AGE
        # -------------------------------
        if target_age == source_age:
            print("⚠️ Target age equals source age")

            enhanced = self.enhancer.process(img)

            if aged_output_path:
                cv2.imwrite(aged_output_path, img)
            if image_output_path:
                cv2.imwrite(image_output_path, enhanced)

            return enhanced

        # -------------------------------
        # CASE 2: CHILD DE-AGING
        # -------------------------------
        if 5 <= target_age <= 25 and target_age < source_age:
            print(f"✅ De-aging to child age: {source_age} → {target_age}")

            # Ensure a valid file path for the PIL save inside FaceDeAger
            if image_output_path is None:
                image_output_path = aged_output_path or "deaged_tmp.png"

            runner = DeAgeRunner(
                image_path=input_image_path,
                current_age=source_age,
                target_age=target_age,
                output_path=image_output_path
            )
            runner.run()

            result = cv2.imread(image_output_path)
            return result

        # -------------------------------
        # CASE 3: NORMAL DE-AGING
        # -------------------------------
        if target_age < source_age and target_age > 25:
            print(f"✅ De-aging enabled: {source_age} → {target_age}")

            self.dpipeline.run(
                input_image_path=input_image_path,
                source_age=source_age,
                target_age=target_age,
                aged_output_path=aged_output_path
            )

        # -------------------------------
        # CASE 4: AGING
        # -------------------------------
        elif target_age > source_age:
            print(f"✅ Aging enabled: {source_age} → {target_age}")

            self.apipeline.run(
                input_image_path=input_image_path,
                source_age=source_age,
                target_age=target_age,
                aged_output_path=aged_output_path
            )

        # -------------------------------
        # Load pipeline result
        # -------------------------------
        if aged_output_path:
            aged_img = cv2.imread(aged_output_path)
        else:
            raise RuntimeError("aged_output_path is required to load result")

        enhanced = self.enhancer.process(aged_img)

        if image_output_path:
            cv2.imwrite(image_output_path, enhanced)

        return enhanced
