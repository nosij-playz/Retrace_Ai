from Phase2.Aging.PreostProcess.checkup import FaceQualityValidator

validator = FaceQualityValidator(
    image_path="images/Mohanlal-Biography.jpg",
    predictor_path="Phase2/Aging/PreostProcess/shape_predictor_68_face_landmarks.dat"
)

a=validator.run()

print("Face quality check result:", a["result"])