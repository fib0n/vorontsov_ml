import math
import numpy as np
from skimage import img_as_float
from skimage.io import imread
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

image = imread('week6/parrots.jpg')
image = img_as_float(image)

n, m, d = tuple(image.shape)
X = np.reshape(image, (n * m, d))

for n_clusters in range(20, 0, -1):
    clf = KMeans(n_clusters=n_clusters, init='k-means++', random_state=241)
    clf.fit(X)

    X_new = np.empty(X.shape)
    for cluster in range(0, len(clf.cluster_centers_)):
        X_new[clf.labels_ == cluster] = clf.cluster_centers_[cluster]

    MSE = mean_squared_error(X, X_new)
    PSNR = 10.0 * math.log10(1.0 / MSE)
    print n_clusters, PSNR
    if PSNR < 20:
        break
