import os
from PIL import Image

toAdd = [1285, 990, 1117, 473, 559, 997, 1238, 1116, 1159, 0]

def divideMeshPNG(meshPath, destPath, rows, cols, imgsToAdd):
    mesh_image = Image.open(meshPath)
    mesh_width, mesh_height = mesh_image.size

    image_width = mesh_width // cols
    image_height = mesh_height // rows

    os.makedirs(destPath, exist_ok=True)

    i = 0
    for row in range(rows):
        for col in range(cols):
            left = col * image_width
            upper = row * image_height
            right = left + image_width
            lower = upper + image_height
            bbox = (left, upper, right, lower)

            image = mesh_image.crop(bbox)

            imageName = "gan{}{}.png".format(row,col)
            imagePath = destPath+'/'+imageName
            image.save(imagePath)
            i+=1
            if i >= imgsToAdd:
                break
        if i >=imgsToAdd:
            break

    print(f"Saved {i} images to '{destPath}'.")

clss = 0
imgsToAdd = toAdd[clss]
meshPath = 'fakes0.png'
destPath = 'cars_gan/train/'+str(clss)
rows, cols = 32,32
divideMeshPNG(meshPath, destPath, rows, cols, imgsToAdd)