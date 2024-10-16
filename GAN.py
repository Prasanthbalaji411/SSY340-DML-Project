import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np


imageSize = 128
batchSize = 32  #images per batch
nEpochs = 1  
learningRate = 10**-3 
latentSize = 100
nGenFilters= 32  # Number of generator filters
nDiscFilters = 32  # Number of discriminator filters
nImages = 10


transform = transforms.ToTensor() #Remove

############################################# IMPORT DATASET ##########################################


# subset of test dataset
#dataset = datasets.CIFAR10(root="./data", download=False, transform=transform)
#dataset = [dataset[i] for i in range(nImages)]


dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize, shuffle=True)


#######################################################################################################
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latentSize, nGenFilters * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(nGenFilters * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(nGenFilters * 8, nGenFilters * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nGenFilters * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(nGenFilters * 4, nGenFilters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nGenFilters * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(nGenFilters * 2, nGenFilters, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nGenFilters),
            nn.ReLU(True),
            nn.ConvTranspose2d(nGenFilters, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)
    


#######################################################################################################

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, nDiscFilters,2, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nDiscFilters, nDiscFilters * 2, 2, 1, 1, bias=False),
            nn.BatchNorm2d(nDiscFilters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nDiscFilters * 2, nDiscFilters * 4,2, 1, 0, bias=False),
            nn.BatchNorm2d(nDiscFilters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nDiscFilters * 4, nDiscFilters * 2, 2), #1, 1, bias=False),
            #nn.BatchNorm2d(nDiscFilters * 2),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nDiscFilters * 2, nDiscFilters, 1, 1, 0, bias=False),
            nn.Linear(nDiscFilters, 1),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        return self.main(x)
    
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
        labelsReal = torch.ones(batchSize, 1, device=device)
        labelsFake = torch.zeros(batchSize, 1, device=device)

        # Train Discriminator
        optimizer_d.zero_grad()
        outputs = discriminator(images).view(-1, 1)
        print(outputs.shape)
        lossReal = criterion(outputs, labelsReal)
        lossReal.backward()

        # Generate fake images
        z = torch.randn(batchSize, latentSize, 1, 1, device=device)
        imagesFake = generator(z)
        outputs = discriminator(imagesFake.detach()).view(-1, 1)
        lossFake = criterion(outputs, labelsFake)
        lossFake.backward()
        optimizer_d.step()

        # Train Generator
        optimizer_g.zero_grad()
        outputs = discriminator(imagesFake).view(-1, 1)
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
generated_images = generate_images()
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Generated Images")
plt.imshow(np.transpose(vutils.make_grid(generated_images.cpu(), padding=2, normalize=True), (1, 2, 0)))
plt.show()

