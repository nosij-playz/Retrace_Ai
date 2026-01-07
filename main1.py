from Phase1.pipeline import FaceVerifier

img_child = r"C:\Users\lenovo\Desktop\Retrace Ai\images\WE Unseen\SreeLakshmi Chettan\WhatsApp Image 2025-09-11 at 19.05.29_eb147fd2.jpg"
img_adult = r"C:\Users\lenovo\Desktop\Retrace Ai\images\WE Unseen\SreeLakshmi\WhatsApp Image 2025-09-01 at 00.12.09_c9934291.jpg"

verifier = FaceVerifier()

verifier.verify(img_child, img_adult)
