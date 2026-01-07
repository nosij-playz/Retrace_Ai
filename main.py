from Phase2.age_transform import FullAgePipeline


if __name__ == "__main__":

    pipeline = FullAgePipeline()

    final_image = pipeline.run(
        input_image_path="images (14).jpg",
        source_age=55,
        target_age=80,
        final_output_path="images/final.png",
        aged_intermediate_path="images/aged_photo.jpg"  # optional
    )

    print("✅ Image processing complete! Saved to images/final.png")
