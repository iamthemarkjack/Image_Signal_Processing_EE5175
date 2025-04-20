import numpy as np
import matplotlib.pyplot as plt

from kmeans import KMeans

# read in the images
car = plt.imread(r'car.png')
flower = plt.imread(r'flower.png')

# convert [0, 1] to [0, 255]
car = (car * 255).astype(np.uint8)
flower = (flower * 255).astype(np.uint8)

N = 30

# running KMeans with 5 iterations for 30 times for car
car_costs, car_outputs = [], []
for n in range(N):
    car_flatten = car.reshape(-1, 3)
    car_kmeans = KMeans(car_flatten,k=3,max_iters=5)
    cost = np.sum(np.min(np.linalg.norm(car_flatten[:, np.newaxis] - car_kmeans.means, axis=2), axis=1))
    car_cluster_ids = car_kmeans.predict(car_flatten)
    colored_car = car_kmeans.means[car_cluster_ids] 
    car_outputs.append(colored_car)
    car_costs.append(cost)

# car with minimum cost
car_min = car_outputs[np.argmin(car_costs)]
car_min = car_min.reshape(car.shape) / 255.0

# car with maximum cost
car_max = car_outputs[np.argmax(car_costs)]
car_max = car_max.reshape(car.shape) / 255.0

# running KMeans with 5 iterations for 30 times for flower
flower_costs, flower_outputs = [], []
for n in range(N):
    flower_flatten = flower.reshape(-1, 3)
    flower_kmeans = KMeans(flower_flatten,k=3,max_iters=5)
    cost = np.sum(np.min(np.linalg.norm(flower_flatten[:, np.newaxis] - flower_kmeans.means, axis=2), axis=1))
    flower_cluster_ids = flower_kmeans.predict(flower_flatten)
    colored_flower = flower_kmeans.means[flower_cluster_ids] 
    flower_outputs.append(colored_flower)
    flower_costs.append(cost)

# flower with minimum cost
flower_min = flower_outputs[np.argmin(flower_costs)]
flower_min = flower_min.reshape(flower.shape) / 255.0

# flower with maximum cost
flower_max = flower_outputs[np.argmax(flower_costs)]
flower_max = flower_max.reshape(flower.shape) / 255.0

# saving the outputs
plt.imsave('car_random_init_min.png', car_min)
plt.imsave('car_random_init_max.png', car_max)
plt.imsave('flower_random_init_min.png', flower_min)
plt.imsave('flower_random_init_max.png', flower_max)