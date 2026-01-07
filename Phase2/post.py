from Preprocess import FaceAgeProcessor 
import cv2
    # Initialize processor
processor = FaceAgeProcessor(model_path="/content/79999_iter.pth")
    
    # Process image
rgba, mask = processor.process_image(
        "/content/images (9).jpg",
        target_age=50
    )
    
    # Save result
cv2.imwrite(
        "final.png",
        cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    )
    
print("Processing complete! Result saved as 'final.png'")