import os
import shutil
from pathlib import Path

className = '9'
destinationPath = 'cars_split/train/'+className+'/'
sourcePath = 'cars_sorted/test/'+className+'/'

print('class=', className)
filenames =  [f for f in os.listdir(sourcePath) if os.path.isfile(os.path.join(sourcePath, f))]
nToTrain = int(0.5*len(filenames))+1
print('files:', len(filenames))
trainFiles = filenames[:nToTrain]
testFiles = filenames[nToTrain:]

for file in trainFiles:
    filename=file.strip(' ')
    src_path = sourcePath+filename
    

    os.makedirs(destinationPath, exist_ok=True)    
    
    dest_path = os.path.join(destinationPath, filename)
    
    # Move the image
    if os.path.exists(src_path):
        shutil.move(src_path, dest_path)
    else:
        print(f"File {src_path} not found.")

print(nToTrain, 'files moved to train')
destinationPath = 'cars_split/test/'+className+'/'

for file in testFiles:
    filename=file.strip(' ')
    src_path = sourcePath+filename
    

    os.makedirs(destinationPath, exist_ok=True)    
    
    dest_path = os.path.join(destinationPath, filename)
    
    # Move the image
    if os.path.exists(src_path):
        shutil.move(src_path, dest_path)
    else:
        print(f"File {src_path} not found.")
print(len(filenames)-nToTrain, 'files moved to test')