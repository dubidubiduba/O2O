import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from matplotlib.animation import FuncAnimation

# 生成样本数据
data, y = make_blobs(n_samples=80, n_features=2, centers=3)

fig, ax = plt.subplots()
data_scatter = ax.scatter(data[:,0], data[:,1], c=y, s=50)
centers_scatter = ax.scatter([], [], s=200, marker='*')

def kmeans_step():
    k = 3
    centers = data[np.random.choice(data.shape[0], size=k)]
    for i in range(30):
        distances = np.sqrt(((data[:, np.newaxis] - centers[None, :])**2).sum(axis=2))
        labels = np.argmin(distances, axis=1)
        new_centers = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        centers = new_centers
        yield data_scatter, centers_scatter, labels, new_centers

def update_plot(i, data_scatter, centers_scatter, *arg):
    labels, new_centers = arg[2:]
    colors = np.array(['red', 'green','blue'])
    data_scatter.set_facecolor(colors[labels])
    centers_scatter.set_offsets(new_centers)
    return data_scatter, centers_scatter

anim = FuncAnimation(fig, func=lambda i: update_plot(i, data_scatter, centers_scatter, *kmeans_step().__next__()),
                     frames=30, interval=50)

plt.show()
