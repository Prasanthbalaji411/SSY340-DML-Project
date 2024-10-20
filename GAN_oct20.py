from Preprocessing_oct16 import *
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np


imageSize = 32
batchSize = 10  #images per batch
nEpochs = 50
learningRate = 10**-3 
latentSize = 100
nGenFilters= 77  # Number of generator filters
nDiscFilters = 44  # Number of discriminator filters
nImages = 30 # For now it must be divisible by batch size



############################################# IMPORT DATASET ##########################################
images_path = 'cars_sorted'


# Transformations for input images
transform = transforms.Compose([
       transforms.Resize((imageSize,imageSize)),
       transforms.ToTensor(),
       transforms.Normalize([0.5], [0.5])
   ])

# Subset of test dataset
dataset = datasets.ImageFolder(root=images_path+'/train', transform=transform)
dataset = [dataset[i] for i in range(nImages)]

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=False) #True

#######################################################################################################
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layers = nn.Sequential(
            # Input is (latentSize, 1, 1)
            # upsample to (nGenFilters * 8, 4, 4)
            #                 (in_channels, out_channels,    kernel_size,  stride, padding, output_padding, bias)
            nn.ConvTranspose2d(latentSize, nGenFilters * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(nGenFilters * 8),
            nn.ReLU(True),

            # Upsample to (nGenFilters * 4, 8, 8)
            nn.ConvTranspose2d(nGenFilters * 8, nGenFilters * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nGenFilters * 4),
            nn.ReLU(True),

            # Upsample to (nGenFilters * 2, 16, 16)
            nn.ConvTranspose2d(nGenFilters * 4, nGenFilters*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nGenFilters*2),
            nn.ReLU(True),
            

            # Upsample to (nGenFilters, 32, 32)
            nn.ConvTranspose2d(nGenFilters * 2, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(True),

            # Upsample to (3, 64, 64)  
            #nn.ConvTranspose2d(nGenFilters, 3, kernel_size=4, stride=2, padding=1, bias=False),
            #nn.BatchNorm2d(3),
            #nn.ReLU(True)

            # Last out_channels must be 3
            nn.Tanh()
            )



    def forward(self, x):
        return self.layers(x)
    


#######################################################################################################

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.convLayers = nn.Sequential(
            nn.Conv2d(3, nDiscFilters,2, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
            #nn.Conv2d(nDiscFilters, nDiscFilters * 2, 2, 1, 1, bias=False),
            #nn.BatchNorm2d(nDiscFilters * 2),
            #nn.LeakyReLU(0.2, inplace=True),
            #nn.Conv2d(nDiscFilters * 2, nDiscFilters * 4,2, 1, 0, bias=False),
            #nn.BatchNorm2d(nDiscFilters * 4),
            #nn.LeakyReLU(0.2, inplace=True),
            #nn.Conv2d(nDiscFilters * 4, nDiscFilters * 2, 2), #1, 1, bias=False),
            #nn.BatchNorm2d(nDiscFilters * 2),
            #nn.LeakyReLU(0.2, inplace=True),
            #nn.Conv2d(nDiscFilters * 2, nDiscFilters, 1, 1, 0, bias=False),

        )
        self.convOutputSize = self.get_conv_output_size()

        self.linearLayers = nn.Sequential(
            nn.Linear(self.convOutputSize, 1000), 
            nn.Linear(1000, batchSize),
            nn.Sigmoid())
        


    def forward(self, x):
        x = self.convLayers(x)
        x = torch.flatten(x)
        x = self.linearLayers(x)
        return x
    

    def get_conv_output_size(self):
        x = torch.zeros(batchSize, 3, imageSize, imageSize)
        x = self.convLayers(x)
        print('\nnumel=',x.numel())
        return x.numel() 
    
#######################################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device is', device)

# Initialize generator and discriminator
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Loss and optimizers
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=learningRate)#, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=learningRate)#, betas=(0.5, 0.999))


# TODO Make training loop function
for epoch in range(nEpochs):
    for i, (images, _) in enumerate(dataloader):
        # Real images
        images = images.to(device)
        print()
        print('shape batch',images.shape)
        labelsReal = torch.ones(batchSize, 1, device=device)
        labelsFake = torch.zeros(batchSize, 1, device=device)

        # Train Discriminator
        optimizer_d.zero_grad()
        outputs = discriminator(images).view(-1, 1)
        print('disc pred - real images, (all should be 1)',outputs)
        lossReal = criterion(outputs, labelsReal)
        lossReal.backward()

        # Generate fake images
        z = torch.randn(batchSize, latentSize, 1, 1, device=device)
        imagesFake = generator(z)
        print('generated images')
        print(imagesFake.detach().shape)
        outputs = discriminator(imagesFake.detach()).view(-1, 1)
        lossFake = criterion(outputs, labelsFake)
        lossFake.backward()
        optimizer_d.step()

        # Train Generator
        optimizer_g.zero_grad()
        outputs = discriminator(imagesFake).view(-1, 1)
        print('disc pred - fake images, (all should be 0)',outputs)
        loss_g = criterion(outputs, labelsReal)
        loss_g.backward()
        optimizer_g.step()

    # Print losses
    print(f"Epoch [{epoch + 1}/{nEpochs}] - D Loss: {lossReal.item() + lossFake.item():.4f}, G Loss: {loss_g.item():.4f}")


def generate_images(num_images=16):
    with torch.no_grad():
        z = torch.randn(num_images, latentSize, 1, 1, device=device)
        imagesFake = generator(z)
        return imagesFake


#######################################################################################################

# Generate and display images
generated_images = generate_images(num_images=16)
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Generated Images")
plt.imshow(np.transpose(vutils.make_grid(generated_images.cpu(), padding=2, normalize=True), (1, 2, 0)))
plt.show() 

images = []

for image, _ in dataloader:

    images.append(image)
    break
print()

for i in range(np.shape(images)[1]):
    plt.figure(figsize=(8, 8))
    plt.title("Real Images")
    plt.imshow(np.transpose(images[0][i]))
    plt.show() 
