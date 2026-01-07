import torch
import numpy as np
from PIL import Image
from Phase2.Aging.Perform.models import UNet
from Phase2.Aging.Perform.test_functions import process_image

def age_image(input_image_path, source_age, target_age, output_path=None):
    """
    Process a single image for face re-aging

    Args:
        input_image_path (str): Path to input image
        source_age (int): Current age of the person
        target_age (int): Target age to transform to
        output_path (str, optional): Path to save output image

    Returns:
        PIL.Image: Processed image
    """
    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("[INFO] Using device:", device)
    # Load model
    model_path = 'Phase2/Aging/Perform/best_unet_model.pth'
    unet_model = UNet().to(device)
    unet_model.load_state_dict(torch.load(model_path, map_location=device))
    unet_model.eval()

    
    # Load and preprocess image
    image = Image.open(input_image_path)

    # Convert to RGB safely, preserving visibility for palette/alpha images
    if image.mode != 'RGB':
        print(f"Converting image from {image.mode} to RGB")
        if image.mode in ('RGBA', 'LA', 'P'):
            # Composite over white to avoid disappearing content after dropping alpha
            rgba = image.convert('RGBA')
            background = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, rgba).convert('RGB')
        else:
            image = image.convert('RGB')

    # Process image
    result_image = process_image(
        unet_model,
        image,
        video=False,
        source_age=source_age,
        target_age=target_age,
        window_size=512,
        stride=256
    )

    # Always keep the result in RGB (avoids alpha/palette issues downstream)
    if result_image.mode != 'RGB':
        print(f"Converting result image from {result_image.mode} to RGB")
        result_image = result_image.convert('RGB')

    # Save if output path provided
    if output_path:
        result_image.save(output_path)
        print(f"Image saved to: {output_path}")

    return result_image

# Example usage with error handling
