import cv2


class ImageSharpening:
    def __init__(self, strength=1.5):
        """
        strength: 1.0–2.0 (safe range)
        """
        self.strength = strength

    def sharpen(self, img, strength=None):
        """
        Sharpen an image using unsharp masking
        """
        if img is None:
            raise ValueError("Input image is None")

        s = strength if strength is not None else self.strength

        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0, sigmaY=1.0)
        sharpened = cv2.addWeighted(
            img, 1 + s,
            blurred, -s,
            0
        )
        return sharpened
