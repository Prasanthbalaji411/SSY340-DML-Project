from scipy.io import loadmat
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import shutil
import pandas as pd
import numpy as np

# 1 make class = brand
# 2 extract brand from annotations Done
# 3 make df with a brand per file

# how handle same file names for train & test?
# Load labels, annotations etc
cars_data_path = '/Users/sonjakanerot/Lokalt/cars_dataset'

test_ann_df = pd.read_csv('/Users/sonjakanerot/Lokalt/cars_dataset/anno_test.csv', header=None)
train_ann_df = pd.read_csv('/Users/sonjakanerot/Lokalt/cars_dataset/anno_train.csv', header=None)
names_df = pd.read_csv('/Users/sonjakanerot/Lokalt/cars_dataset/names.csv', header = None)

# Get unique brands
names = []
for i in range(196):
    names.append(str(names_df.iloc[i]).split(' ')[4])
brands = set(names)

brandsDict = {}
for i, brand in enumerate(brands):
    brandsDict[brand]=i

train_labels_df = pd.DataFrame()
nTrainSamples = train_ann_df.shape[0]

test_labels_df = pd.DataFrame()
nTestSamples = test_ann_df.shape[0]

for i in range(nTrainSamples):
    newRow = {'filename':train_ann_df.iloc[i,0], 'class':brandsDict[str(names_df.iloc[train_ann_df.iloc[i,5]-1]).split('\n')[0].split(' ')[4]]}
    train_labels_df = train_labels_df._append(newRow, ignore_index=True)

"""for i in range(nTrainSamples):
    newRow = {0:train_ann_df.iloc[i,0], 1:brandsDict[str(names_df.iloc[train_ann_df.iloc[i,5]-1]).split('\n')[0].split(' ')[4]]}
    train_labels_df = train_labels_df._append(newRow, ignore_index=True)
"""

class FolderStructure():
    # Load labels, annotations etc
    cars_data_path = '/Users/sonjakanerot/Lokalt/cars_dataset'
    destination = '/Users/sonjakanerot/Lokalt/cars_sorted'


    # Loop through each row in the CSV
    for _, row in train_labels_df.iterrows():
        img_filename = row['filename']
        img_class = row['class']
        
        # Source file path
        src_path = os.path.join(cars_data_path, img_filename)
        
        # Destination directory for this class
        class_dir = os.path.join(destination, img_class)
        os.makedirs(class_dir, exist_ok=True)
        
        # Destination file path
        dest_path = os.path.join(class_dir, img_filename)
        
        # Move the image
        if os.path.exists(src_path):
            shutil.move(src_path, dest_path)
        else:
            print(f"File {src_path} not found.")

    print("Images organized by class")


# NOW WE CAN USE:
#dataset = datasets.ImageFolder(root=cars_data_path+'/cars_test/cars_test', transform=transform)





imageSize=30
# Transformations for input images
transform = transforms.Compose([
    transforms.Resize(imageSize),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])