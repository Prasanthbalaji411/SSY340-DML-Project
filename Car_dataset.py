import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Load the .mat file containing annotations and metadata
cars_annos_path = 'D:/Academic Documents/Courses/Deep Machine Leaning/cars_annos.mat'
car_data = scipy.io.loadmat(cars_annos_path)

# Extract annotations and meta (class names)
annotations = car_data['annotations'][0]
meta = car_data['class_names'][0]  # 'class_names' contains car make, model, year

# Extract car class IDs from annotations
car_class_ids = [anno[5][0][0] for anno in annotations]  # [5] corresponds to class ID (1-based)

# Extract the car company name from the class names (meta)
# The company name is the first word of the car name (e.g., 'Chevrolet' from 'Chevrolet Malibu Wagon 2012')
car_companies = [meta[car_id - 1][0].split()[0] for car_id in car_class_ids]

# Count the occurrences of each car company
company_counts = Counter(car_companies)

# Plot the histogram of car companies
plt.figure(figsize=(12, 8))
plt.bar(company_counts.keys(), company_counts.values(), color='skyblue')
plt.xlabel('Car Company')
plt.ylabel('Number of Images')
plt.xticks(rotation=90)  # Rotate x-axis labels for readability
plt.title('Car Company Distribution in Stanford Cars Dataset')
plt.tight_layout()
plt.show()
