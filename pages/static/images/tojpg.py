import glob
import os

from PIL import Image

directory = "."

for filename in glob.glob(os.path.join(directory, "*.png")):
    png_path = os.path.join(directory, filename)
    jpg_path = os.path.join(directory, os.path.splitext(filename)[0] + ".jpg")

    with Image.open(png_path) as img:
        # Convert to RGB (PNG might have alpha channel)
        rgb_img = img.convert("RGB")
        # Save as high-quality JPG
        rgb_img.save(jpg_path, "JPEG", quality=95, optimize=True)

    print(f"Converted {filename} -> {os.path.basename(jpg_path)}")

print("Conversion complete.")
