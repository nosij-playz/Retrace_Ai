import warnings
from Phase2.Deaging.psp.deage import FaceDeAger


class DeAgeRunner:
    """
    Simple controller class to run FaceDeAger safely
    """

    def __init__(
        self,
        image_path,
        current_age,
        target_age,
        output_path="deaged.png",
        suppress_warnings=True
    ):
        self.image_path = image_path
        self.current_age = current_age
        self.target_age = target_age
        self.output_path = output_path

        if suppress_warnings:
            warnings.filterwarnings("ignore", category=FutureWarning)

        self.deager = FaceDeAger(
            image_path=self.image_path,
            current_age=self.current_age,
            target_age=self.target_age,
            output_path=self.output_path
        )

    def run(self):
        """
        Execute de-aging pipeline
        """
        self.deager.run()
