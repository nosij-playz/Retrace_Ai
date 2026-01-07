from Phase2.Aging.Perform.age import age_image

        # Process a single image
result = age_image(
            input_image_path="Mohanlal-Biography.jpg",
            source_age=50,
            target_age=90,
            output_path="aged_photo.jpg"
        )
