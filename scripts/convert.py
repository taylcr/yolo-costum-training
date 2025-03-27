import os
from PIL import Image
import pillow_heif

# Register the HEIF opener with Pillow
pillow_heif.register_heif_opener()

def convert_heic_to_jpg(heic_path, jpg_path):
    # Open the HEIC image using Pillow (with pillow-heif handling HEIC files)
    image = Image.open(heic_path)
    # Save the image as JPEG
    image.save(jpg_path, "JPEG")

def main():
    folder = os.getcwd()
    for filename in os.listdir(folder):
        if filename.lower().endswith(".heic"):
            heic_path = os.path.join(folder, filename)
            base, _ = os.path.splitext(filename)
            jpg_filename = base + ".jpg"
            jpg_path = os.path.join(folder, jpg_filename)
            print(f"Converting {filename} to {jpg_filename}...")
            try:
                convert_heic_to_jpg(heic_path, jpg_path)
            except Exception as e:
                print(f"Failed to convert {filename}: {e}")
    print("Conversion complete.")

if __name__ == "__main__":
    main()
