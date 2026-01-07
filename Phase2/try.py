from Phase2.Deaging.deage import DeAgeRunner


runner = DeAgeRunner(
    image_path="images/Mohanlal-Biography.jpg",
    current_age=50,
    target_age=18,
    output_path="deaged.png"
)

runner.run()
