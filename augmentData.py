import torch
from torchvision import transforms
from PIL import Image
import os
import shutil
import numpy as np

augmentations = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),        # horizontal flip
    transforms.RandomRotation(30),                 # rotate images up to 30 degrees
    transforms.ColorJitter(brightness=0.2,         # brightness
                           contrast=0.2,
                           saturation=0.2,
                           hue=0.1),
    transforms.RandomResizedCrop(size=(64, 64),    # Randomly crop and resize to 64x64
                                 scale=(0.8, 1.0)),
    transforms.RandomAffine(degrees=0,             # Small affine transformations
                            translate=(0.1, 0.1),
                            scale=(0.9, 1.1),
                            shear=10),
])



image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif'}

maxImages = 0
nImageslist = []
for i in range(10):
    #folderPath = 'cars_split/train/'+str(i)
    folderPath = 'cars_aug/train/'+str(i)
    filenames = [f for f in os.listdir(folderPath) if os.path.isfile(os.path.join(folderPath, f)) and any(f.lower().endswith(ext) for ext in image_extensions)]   
    nImages = len(filenames)
    nImageslist.append(nImages)
    if nImages > maxImages:
        maxImages = nImages

print(nImageslist)
print('maxImages:', maxImages)



'''for i in range(10):
    folderPath = 'cars_split/train/'+str(i)
    destPath = 'cars_aug/train/'+str(i)

    filenames = [f for f in os.listdir(folderPath) if os.path.isfile(os.path.join(folderPath, f)) and any(f.lower().endswith(ext) for ext in image_extensions)]   
    toAdd = maxImages - len(filenames)
    print('toAdd', toAdd)

    os.makedirs(destPath, exist_ok=True)

    for fileName in filenames: # copy original images first
        imagePath = folderPath + '/'+fileName
        shutil.copy2(imagePath, destPath+'/'+fileName)  

    i=0
    while True: # add as many augmented images as needed
        for fileName in filenames:
            imagePath = folderPath +'/'+fileName
            image = Image.open(imagePath).convert("RGB")  
            augmentedImage = augmentations(image)
            augmentedImage.save(destPath+'/aug{}.png'.format(i))
            i += 1
            if i >= toAdd: 
                break
        if i >= toAdd: 
            break'''

