import os
from PIL import Image
import torch

'''for i in range(49):
    image_folder = 'carsNotSplit/'+str(i)

    # Loop through all images
    for filename in os.listdir(image_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')): 
            img_path = os.path.join(image_folder, filename)
            img = Image.open(img_path)
            
            # Convert to RGB if not already
            if img.mode != 'RGB':
                img = img.convert('RGB')
                img.save(img_path)  # Overwrite the original image or save as a new file
                print(f'Converted {filename} to RGB format.')

    print('All images processed.', i)'''

print(torch.__version__)  # Print the version of PyTorch
print(torch.cuda.is_available())  # Check if CUDA is available


