from Preprocessing import *
from GAN import *
from CarClassifier import *
from evaluation import *

train_dataset = datasets.ImageFolder(root=destination_path+'/train', transform=transform)
test_dataset = datasets.ImageFolder(root=destination_path+'/test', transform=transform)

