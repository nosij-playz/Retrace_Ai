from Preprocess.rembg import BackgroundRemover



remover = BackgroundRemover()

input_image = r"C:\Users\lenovo\Desktop\Retrace Ai\images\128px-Alia_Bhatt_at_Berlinale_2022_Ausschnitt.jpg"
output_file = remover.remove_background(input_image, "removed.png")

print(f"Background removed successfully!\nSaved at: {output_file}")
