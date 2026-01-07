import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from Phase2.Preprocess.model import BiSeNet


class FaceAgeProcessor:
    """Face parsing and age-based resizing processor"""
    
    # Head labels for parsing (REQUIRED)
    HEAD_LABELS = [
        1,   # skin
        2,   # nose
        4, 5,  # eyes
        6, 7,  # eyebrows
        8, 9,  # ears
        10, 11, 12, 13,  # mouth / lips
        14,  # neck
        17   # hair
    ]
    
    def __init__(self, model_path="/content/79999_iter.pth"):
        """
        Initialize the FaceAgeProcessor.
        
        Args:
            model_path (str): Path to the pre-trained model weights
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        
        # Initialize model
        self.net = None
        self.transform = None
        
        self._initialize_model()
        self._initialize_transforms()
    
    def _initialize_model(self):
        """Initialize the BiSeNet model and load weights"""
        self.net = BiSeNet(n_classes=19)
        self.net.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.net.to(self.device)
        self.net.eval()
    
    def _initialize_transforms(self):
        """Initialize image transformation pipeline"""
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    @staticmethod
    def age_to_scale(age):
        """
        Convert age to scale factor for resizing.
        
        Args:
            age (int): Target age (0-100)
            
        Returns:
            float: Scale factor for resizing
        """
        age = max(0, min(age, 100))

        if age <= 12:
            return 0.60 + (age / 12) * 0.12
        elif age <= 26:
            return 0.65 + ((age - 12) / 12) * 0.10
        elif age <= 70:
            return 0.95 - ((age - 26) / 45) * 0.03
        elif age <= 90:
            return 0.93 - ((age - 70) / 20) * 0.10
        else:
            return 0.87
    
    def reinforce_eyebrows(self, mask, parsing):
        """
        Reinforce eyebrow regions in the mask.
        
        Args:
            mask (np.ndarray): Current mask
            parsing (np.ndarray): Parsing result
            
        Returns:
            np.ndarray: Updated mask
        """
        brow = np.zeros_like(mask)
        for lbl in [6, 7]:
            brow[parsing == lbl] = 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        brow = cv2.dilate(brow, kernel, 1)
        mask[brow > 0] = 255
        return mask
    
    def fill_mask_holes(self, mask):
        """
        Fill holes in the mask using flood fill.
        
        Args:
            mask (np.ndarray): Input mask
            
        Returns:
            np.ndarray: Filled mask
        """
        h, w = mask.shape
        flood = mask.copy()
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, flood_mask, (0, 0), 255)
        flood_inv = cv2.bitwise_not(flood)
        return mask | flood_inv
    
    def parse_head(self, image_path):
        """
        Parse head region from image.
        
        Args:
            image_path (str): Path to input image
            
        Returns:
            tuple: (head_image, mask)
        """
        # Load and preprocess image
        img = Image.open(image_path).convert("RGB")
        orig = np.array(img)
        h, w = orig.shape[:2]
        
        # Apply transformations
        inp = self.transform(img).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            out = self.net(inp)[0]
            parsing = out.squeeze(0).argmax(0).cpu().numpy()
        
        # Create initial mask from head labels
        mask = np.zeros_like(parsing, dtype=np.uint8)
        for lbl in self.HEAD_LABELS:
            mask[parsing == lbl] = 255
        
        # Post-process mask
        mask = self.reinforce_eyebrows(mask, parsing)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Improve mask quality with morphological operations
        # Close small gaps
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        
        # Fill holes
        mask = self.fill_mask_holes(mask)
        
        # Dilate slightly to include edge pixels
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.dilate(mask, kernel_dilate, iterations=1)
        
        # Smooth edges with Gaussian blur
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # Threshold to binary
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        # Final cleanup
        mask = (mask > 0).astype(np.uint8) * 255
        
        # Extract head region
        head = cv2.bitwise_and(orig, orig, mask=mask)
        
        return head, mask
    
    def resize_head_by_age(self, head, mask, target_age):
        """
        Resize head region based on target age (skull size varies with age).
        
        Args:
            head (np.ndarray): Head image (RGB)
            mask (np.ndarray): Head mask
            target_age (int): Target age for resizing
            
        Returns:
            np.ndarray: RGBA image with resized head
        """
        # Calculate scale factor based on age
        scale = self.age_to_scale(target_age)
        h, w = head.shape[:2]
        
        # Find bounding box of head
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            # No head found, return original
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[:, :, :3] = head
            rgba[:, :, 3] = mask
            return rgba
            
        y1, y2 = ys.min(), ys.max()
        x1, x2 = xs.min(), xs.max()
        
        # Crop head region
        head_crop = head[y1:y2, x1:x2]
        mask_crop = mask[y1:y2, x1:x2]
        
        orig_h = y2 - y1
        orig_w = x2 - x1
        
        # Calculate new dimensions based on age scale
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        # Ensure minimum size
        new_w = max(10, new_w)
        new_h = max(10, new_h)
        
        # Resize cropped regions
        head_r = cv2.resize(head_crop, (new_w, new_h), cv2.INTER_LINEAR)
        mask_r = cv2.resize(mask_crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Create RGBA canvas
        canvas = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Calculate center position (keep head centered)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        
        # Calculate new position centered
        nx1 = cx - new_w // 2
        ny1 = cy - new_h // 2
        nx2 = nx1 + new_w
        ny2 = ny1 + new_h
        
        # Clamp to canvas bounds and get valid region
        clamp_x1 = max(0, nx1)
        clamp_y1 = max(0, ny1)
        clamp_x2 = min(w, nx2)
        clamp_y2 = min(h, ny2)
        
        # Calculate corresponding source region
        src_x1 = clamp_x1 - nx1
        src_y1 = clamp_y1 - ny1
        src_x2 = src_x1 + (clamp_x2 - clamp_x1)
        src_y2 = src_y1 + (clamp_y2 - clamp_y1)
        
        # Place resized head on canvas
        if clamp_x1 < clamp_x2 and clamp_y1 < clamp_y2 and src_x1 < src_x2 and src_y1 < src_y2:
            canvas[clamp_y1:clamp_y2, clamp_x1:clamp_x2, :3] = head_r[src_y1:src_y2, src_x1:src_x2]
            canvas[clamp_y1:clamp_y2, clamp_x1:clamp_x2, 3] = mask_r[src_y1:src_y2, src_x1:src_x2]
        
        return canvas
    
    def process_image(self, image_path, target_age):
        """
        Process an image: parse head and resize based on age.
        
        Args:
            image_path (str): Path to input image
            target_age (int): Target age for resizing
            
        Returns:
            tuple: (rgba_image, mask)
        """
        # Parse head from image
        head, mask = self.parse_head(image_path)
        
        # Resize head based on age
        rgba = self.resize_head_by_age(head, mask, target_age)
        
        return rgba, mask


