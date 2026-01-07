import cv2

from Phase2.Preprocess.sharpen import ImageSharpening
img = cv2.imread("images/Mohanlal-Biography.jpg")

sharpener = ImageSharpening(strength=1.2)
sharp_img = sharpener.sharpen(img)

cv2.imwrite("sharpened.jpg", sharp_img)
