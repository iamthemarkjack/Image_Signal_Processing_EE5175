import numpy as np
import matplotlib.pyplot as plt

from kmeans import KMeans

# read in the images
car = plt.imread(r'car.png')
flower = plt.imread(r'flower.png')

# convert [0, 1] to [0, 255]
car = (car * 255).astype(np.uint8)
flower = (flower * 255).astype(np.uint8)

# given initial cluster means
init_means = [[255, 0, 0],
              [0, 0, 0],
              [255, 255, 255]]

# kmeans for car with 5 iterations
car_flatten = car.reshape(-1, 3)
car_kmeans = KMeans(car_flatten, 3, init_means, 5)
car_cluster_ids = car_kmeans.predict(car_flatten)
colored_car = car_kmeans.means[car_cluster_ids] 
colored_car = colored_car.reshape(car.shape) 
colored_car = colored_car / 255.0

# kmeans for flower with 5 iterations
flower_flatten = flower.reshape(-1, 3)
flower_kmeans = KMeans(flower_flatten, 3, init_means, 5)
flower_cluster_ids = flower_kmeans.predict(flower_flatten)
colored_flower = flower_kmeans.means[flower_cluster_ids] 
colored_flower = colored_flower.reshape(flower.shape) 
colored_flower = colored_flower / 255.0

# saving the outputs
plt.imsave('car_known_init.png', colored_car)
plt.imsave('flower_known_init.png', colored_flower)