from scipy.io import loadmat
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import shutil
import pandas as pd
import numpy as np


# Load labels, annotations etc
test_ann_df = pd.read_csv('anno_test.csv', header=None)
train_ann_df = pd.read_csv('anno_train.csv', header=None)
names_df = pd.read_csv('names.csv', header = None)


# Get unique brands
names = []
for i in range(196):
   names.append(str(names_df.iloc[i]).split(' ')[4])
brands = set(names)
brands=list(brands)
brands.sort()


brandsDict = {}
for i, brand in enumerate(brands):
   brandsDict[brand]=i

def imagesToFolders(trainOrTest, sourcePath, destinationPath):
   # Moves all images in train or test to a selected destinationPath, where folders for each brand is made.
   # This structure is needed later when importing the data to a torch dataset using datasets.ImageFolder


   # {'AM': 0, 'Acura': 1, 'Aston': 2, 'Audi': 3, 'BMW': 4, 'Bentley': 5, 'Bugatti': 6, 'Buick': 7,
   # 'Cadillac': 8, 'Chevrolet': 9, 'Chrysler': 10, 'Daewoo': 11, 'Dodge': 12, 'Eagle': 13, 'FIAT': 14,
   # 'Ferrari': 15, 'Fisker': 16, 'Ford': 17, 'GMC': 18, 'Geo': 19, 'HUMMER': 20, 'Honda': 21, 'Hyundai': 22,
   # 'Infiniti': 23, 'Isuzu': 24, 'Jaguar': 25, 'Jeep': 26, 'Lamborghini': 27, 'Land': 28, 'Lincoln': 29,
   # 'MINI': 30, 'Maybach': 31, 'Mazda': 32, 'McLaren': 33, 'Mercedes-Benz': 34, 'Mitsubishi': 35, 'Nissan': 36,
   # 'Plymouth': 37, 'Porsche': 38, 'Ram': 39, 'Rolls-Royce': 40, 'Scion': 41, 'Spyker': 42, 'Suzuki': 43, 'Tesla': 44,
   # 'Toyota': 45, 'Volkswagen': 46, 'Volvo': 47, 'smart': 48}


   for _, row in test_labels_df.iterrows():
       img_filename = row['filename']
       img_class = str(row['class'])


       src_path = os.path.join(sourcePath, img_filename)
      
       class_dir = os.path.join(destinationPath, img_class)
       os.makedirs(class_dir, exist_ok=True)
      
       # if test: change image filenames so they are not the same as train. (00001.jpg --> 08145.jpg)
       if trainOrTest == 'test':
           oldNum, restOfName = int(img_filename[:5]), img_filename[5:]
           newNum = str(oldNum + nTrainSamples)
           if len(newNum)<5:
               newNum = '0'+newNum
              
           img_filename = newNum+restOfName
      
       dest_path = os.path.join(class_dir, img_filename)
      
       # Move the image
       if os.path.exists(src_path):
           shutil.move(src_path, dest_path)
       else:
           print(f"File {src_path} not found.")


   print("Images organized by class")


# Make dataframes that translate from    classes = 196 models -->  classes = 49 brands
train_labels_df, test_labels_df = pd.DataFrame(), pd.DataFrame()
nTrainSamples, nTestSamples = train_ann_df.shape[0], test_ann_df.shape[0]

# Make train dataframe
for i in range(nTrainSamples):
   model_label = train_ann_df.iloc[i, 5] - 1
   model_name = names_df.iloc[model_label]
   brand_name = str(model_name).split('\n')[0].split(' ')[4]

   newRow = {'filename':train_ann_df.iloc[i,0], 'class':brandsDict[brand_name]}
   train_labels_df = train_labels_df._append(newRow, ignore_index=True)

# Same but for test
for i in range(nTestSamples):
   model_label = test_ann_df.iloc[i,5]-1
   model_name = names_df.iloc[model_label]
   brand_name = str(model_name).split('\n')[0].split(' ')[4]

   newRow = {'filename':test_ann_df.iloc[i,0], 'class':brandsDict[brand_name]}
   test_labels_df = test_labels_df._append(newRow, ignore_index=True)


train_images_path = 'C:/deep-machine-learning/home-assignments/Project/cars_dataset/cars_train/cars_train'
test_images_path = 'C:/deep-machine-learning/home-assignments/Project/cars_dataset/cars_test/cars_test'
destination_path = 'C:/deep-machine-learning/home-assignments/Project/cars_sorted'

# FUNCTION CALL (ONLY NEEDED ONCE FOR EACH DATASET)
imagesToFolders('test', test_images_path, destination_path )


imageSize=30
# Transformations for input images
transform = transforms.Compose([
       transforms.Resize(imageSize),
       transforms.ToTensor(),
       transforms.Normalize([0.5], [0.5])
   ])

# NOW WE CAN USE in the other files:
train_dataset = datasets.ImageFolder(root=destination_path+'/train', transform=transform)
test_dataset = datasets.ImageFolder(root=destination_path+'/test', transform=transform)

class Preprocess():
   # TO BE COMPLETED
   pass






class Augment(): # Use later for evalutation
   pass