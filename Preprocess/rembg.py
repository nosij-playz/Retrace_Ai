import io
import os
from rembg import remove
from PIL import Image, ImageFilter


class BackgroundRemover:
    def __init__(self, output_dir="."):
        """Simple wrapper around rembg with a few quality tweaks."""
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def remove_background(
        self,
        input_path,
        output_name="output.png",
        *,
        alpha_matting: bool = True,
        alpha_matting_foreground_threshold: int = 240,
        alpha_matting_background_threshold: int = 10,
        alpha_matting_erode_size: int = 10,
        edge_smooth_radius: float = 1.5,
    ):
        """
        Remove background from an image with optional matting and edge smoothing.

        Parameters
        ----------
        input_path : str
            Path to input image.
        output_name : str
            Output file name (PNG recommended).
        alpha_matting : bool
            Enable rembg's alpha matting refinement (better hair/edges).
        alpha_matting_foreground_threshold : int
            Foreground threshold for matting (0-255).
        alpha_matting_background_threshold : int
            Background threshold for matting (0-255).
        alpha_matting_erode_size : int
            Erode size used by matting to tighten the trimap.
        edge_smooth_radius : float
            Extra Gaussian-like blur (using PIL) applied to the alpha to reduce halos.

        Returns
        -------
        str
            Output file path.
        """
        output_path = os.path.join(self.output_dir, output_name)

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input image not found: {input_path}")

        with open(input_path, "rb") as i:
            input_image = i.read()
            output_image = remove(
                input_image,
                alpha_matting=alpha_matting,
                alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                alpha_matting_background_threshold=alpha_matting_background_threshold,
                alpha_matting_erode_size=alpha_matting_erode_size,
            )

        # Open the in-memory PNG returned by rembg
        img = Image.open(io.BytesIO(output_image)).convert("RGBA")
        if edge_smooth_radius and edge_smooth_radius > 0:
            r, g, b, a = img.split()
            a = a.filter(ImageFilter.GaussianBlur(radius=edge_smooth_radius))
            img = Image.merge("RGBA", (r, g, b, a))

        img.save(output_path, format="PNG")

        return output_path