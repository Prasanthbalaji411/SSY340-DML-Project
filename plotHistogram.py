import os
import matplotlib.pyplot as plt

nTrainImages = []
for cls in range(10):
    dir = 'cars_split/train/'+str(cls)
    print('class=', str(cls))
    files =  [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    print('files:', len(files))
    nTrainImages.append(len(files))


plt.figure(figsize=(8,8))
plt.title('Class distribution')
plt.bar(range(10),nTrainImages)
plt.xticks(ticks=range(10), labels= ['AM','Acura', 'Aston', 'Audi', 'BMW', 'Bentley', 'Bugatti', 'Buick','Cadillac', 'Chevrolet'], rotation=90)
plt.show()
