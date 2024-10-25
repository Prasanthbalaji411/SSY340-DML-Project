import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# load ResNet
model = models.resnet50(pretrained=True)

num_classes = 49 # 49 classes = car brands
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# freeze all layers except last
for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True

transform = transforms.Compose([
    transforms.Resize((224, 224)),      # match ResNet input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

# load data
trainDataset = datasets.ImageFolder(root='cars_sorted/train', transform=transform)
trainLoader = DataLoader(trainDataset, batch_size=32, shuffle=True)

nTestImages = 1000 
testDataset = datasets.ImageFolder(root='cars_sorted/test', transform=transform)
testDataset = [testDataset[i] for i in range(nTestImages)]
testLoader = DataLoader(testDataset, batch_size=32, shuffle=True)

    


# Training
nEpochs = 10
learningRate = 0.001
optimizer = torch.optim.Adam(model.parameters(),lr=learningRate)
criterion = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(device)

losses = []
f1Micro = []
f1Macro = []
f1Weighted = []

for epoch in range(nEpochs):
    model = model.to(device)
    model.train()
    running_loss = 0.0

    for inputs, labels in trainLoader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{nEpochs}, Loss: {running_loss / len(trainLoader)}")
    losses.append(running_loss)

    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in testLoader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        f1mic = f1_score(all_labels, all_preds, average='micro')
        f1mac = f1_score(all_labels, all_preds, average='macro')
        f1w = f1_score(all_labels, all_preds, average='weighted')

    print("micro F1:", f1mic)
    print("macro F1:", f1mac)
    print("weighted F1:", f1w)
    f1Micro.append(f1mic)
    f1Macro.append(f1mac)
    f1Weighted.append(f1w)

print(f1Micro)
print()
print(f1Macro)
print()
print(f1Weighted)

plt.figure(figsize=(8,8))
plt.title('F1-scores Original data')
plt.plot(range(nEpochs), f1Micro, label = 'F1-score micro')
plt.plot(range(nEpochs), f1Macro, label = 'F1-score macro')
plt.plot(range(nEpochs), f1Weighted, label = 'F1-score weighted')
plt.xlabel('Epochs')
plt.legend()

plt.show()




