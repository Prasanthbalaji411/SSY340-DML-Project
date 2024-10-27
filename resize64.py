import os
from PIL import Image

# Define the input and output folders
for i in range(10):
    folder = 'cars_split/test/'+str(i)

    # Resize each image
    for filename in os.listdir(folder):
        input_path = os.path.join(folder, filename)
        output_path = os.path.join(folder, filename)

        # Check if file is an image
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            with Image.open(input_path) as img:
                # Resize image to 64x64
                img_resized = img.resize((64, 64), Image.LANCZOS)
                img_resized.save(output_path)
                print(f"Resized and saved {filename} to {folder}")
