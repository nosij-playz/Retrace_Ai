# ============================================================
# AGE DE-AGING USING e4e + INTERFACEGAN (CLASS VERSION)
# ============================================================

import torch
import numpy as np
import dlib
from PIL import Image
from torchvision import transforms
from argparse import Namespace
import sys


class FaceDeAger:
    """
    Face De-aging using e4e + InterfaceGAN (age direction)

    Usage:
        deager = FaceDeAger(...)
        deager.run()
    """

    def __init__(
        self,
        image_path,
        current_age,
        target_age,
        output_path="deaged.png",
        e4e_path="Phase2/Deaging/psp/encoder4editing",
        model_path="Phase2/Deaging/psp/encoder4editing/pretrained_models/e4e_ffhq_encode.pt",
        age_dir_path="Phase2/Deaging/psp/encoder4editing/editings/interfacegan_directions/age.pt",
        predictor_path="Phase2/Deaging/psp/encoder4editing/shape_predictor_68_face_landmarks.dat"
    ):
        self.image_path = image_path
        self.current_age = current_age
        self.target_age = target_age
        self.output_path = output_path

        self.e4e_path = e4e_path
        self.model_path = model_path
        self.age_dir_path = age_dir_path
        self.predictor_path = predictor_path

        sys.path.append(self.e4e_path)

        from models.psp import pSp
        from utils.alignment import align_face

        self.pSp = pSp
        self.align_face = align_face

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("[INFO] Using device:", self.device)

        self.net = self._load_model()
        self.transform = self._build_transform()

    # ========================================================
    # AGE FACTOR
    # ========================================================
    @staticmethod
    def compute_age_factor(current_age, target_age):
        """
        Exact cubic age-factor mapping
        Anchors satisfied EXACTLY
        """
        current_age = max(1, min(current_age, 80))
        target_age = max(1, min(target_age, 80))

        if target_age >= current_age:
            return 0.0
        if current_age>50:
            current_age=50
        d = current_age - target_age

        x = np.array([0, 16, 35, 50], dtype=np.float32)
        y = np.array([0.0, -2.15, -3.0, -4.5], dtype=np.float32)

        coeffs = np.polyfit(x, y, 3)
        return float(np.polyval(coeffs, d))

    # ========================================================
    # MODEL
    # ========================================================
    def _load_model(self):
        ckpt = torch.load(self.model_path, map_location="cpu")
        opts = Namespace(**ckpt["opts"])

        opts.checkpoint_path = self.model_path
        opts.device = self.device
        opts.is_train = False

        net = self.pSp(opts).to(self.device).eval()
        return net

    # ========================================================
    # PREPROCESS
    # ========================================================
    @staticmethod
    def _build_transform():
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5])
        ])

    # ========================================================
    # ALIGN FACE
    # ========================================================
    def _align_face(self):
        predictor = dlib.shape_predictor(self.predictor_path)
        aligned = self.align_face(self.image_path, predictor)
        # Ensure image is in RGB mode before saving (handles RGBA, etc.)
        if aligned.mode != 'RGB':
            print(f"Converting aligned face from {aligned.mode} to RGB")
            aligned = aligned.convert('RGB')
        aligned.save("input_aligned.jpg")
        print("[INFO] Aligned face saved")
        return aligned

    # ========================================================
    # RUN PIPELINE
    # ========================================================
    def run(self):
        # Age factor
        age_factor = self.compute_age_factor(
            self.current_age,
            self.target_age
        )
        print(f"[INFO] Age factor = {age_factor:.2f}")

        # Align
        aligned = self._align_face()

        # Preprocess
        input_tensor = self.transform(aligned).unsqueeze(0).to(self.device)

        # Encode
        with torch.no_grad():
            _, latent = self.net(input_tensor, return_latents=True)

        # Load age direction
        age_direction = torch.load(self.age_dir_path, map_location=self.device)
        if age_direction.ndim == 2:
            age_direction = age_direction.unsqueeze(0)
        age_direction = age_direction.to(self.device).float()

        # Edit latent
        edited_latent = latent + age_factor * age_direction

        # Decode
        with torch.no_grad():
            result, _ = self.net.decoder(
                [edited_latent],
                input_is_latent=True,
                randomize_noise=False,
                return_latents=False
            )

        # Save
        out = (result[0].permute(1, 2, 0).cpu().numpy() + 1) / 2
        out = np.clip(out, 0, 1)  # Ensure values are in [0, 1] range
        out = (out * 255).astype(np.uint8)

        Image.fromarray(out).save(self.output_path)
        print(f"[DONE] Saved → {self.output_path}")
