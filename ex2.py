from __future__ import print_function

import time
import os
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from scipy import misc
from struct import unpack

from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def plot_mean_image(X, log):
    meanrow = X.mean(0)
    # present the row vector as an image
    plt.figure(figsize=(3,3))
    plt.imshow(np.reshape(meanrow,(28,28)), cmap=plt.cm.binary)
    plt.title('Mean image of ' + log)
    plt.show()

def get_labeled_data(imagefile, labelfile):
    """
    Read input-vector (image) and target class (label, 0-9) and return it as list of tuples.
    Adapted from: https://martin-thoma.com/classify-mnist-with-pybrain/
    """
    # Open the images with gzip in read binary mode
    images = open(imagefile, 'rb')
    labels = open(labelfile, 'rb')

    # Read the binary data
    # We have to get big endian unsigned int. So we need '>I'

    # Get metadata for images
    images.read(4)  # skip the magic_number
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    # Get metadata for labels
    labels.read(4)  # skip the magic_number
    N = labels.read(4)
    N = unpack('>I', N)[0]

    if number_of_images != N:
        raise Exception('number of labels did not match the number of images')

    # Get the data
    X = np.zeros((N, rows * cols), dtype=np.uint8)  # Initialize numpy array
    y = np.zeros(N, dtype=np.uint8)  # Initialize numpy array
    for i in range(N):
        for id in range(rows * cols):
            tmp_pixel = images.read(1)  # Just a single byte
            tmp_pixel = unpack('>B', tmp_pixel)[0]
            X[i][id] = tmp_pixel
        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0]
    return (X, y)

def my_clustering(X, y, n_clusters, pca):
    # =======================================
    # Complete the code here.
    # return scores like this: return [score, score, score, score]
    # clustering
    centersOld = np.zeros([n_clusters, int(X.shape[1])])
    centersNew = np.zeros([n_clusters, int(X.shape[1])])
    for i in range(n_clusters):
        centersNew[i] = X[round(int(X.shape[0] - 1) * np.random.rand())]
    yNew = np.zeros(int(X.shape[0]))
    distance = np.zeros([int(X.shape[0]), n_clusters])

    while((centersNew != centersOld).any()):
        centersOld = centersNew
        centersOldMatrix = np.ones([int(X.shape[0]) ,int(X.shape[1]), n_clusters])
        distance = np.zeros([int(X.shape[0]), n_clusters])
        sum = np.zeros([n_clusters, int(X.shape[1])])
        num = np.zeros(n_clusters)
        # create matrix of centersOld
        for i in range(n_clusters):
            centersOldMatrix[:, :, i] = centersOldMatrix[:, :, i] * centersOld[i]
        for i in range(n_clusters):
            distance[:, i] = np.linalg.norm(centersOldMatrix[:, :, i] - X, axis=1)
        for i in range(distance.shape[0]):
            distanceList = list(distance[i])
            yNew[i] = distanceList.index(min(distanceList))
        for i in range(int(X.shape[0])):
            for j in range(n_clusters):
                if yNew[i] == j:
                    sum[j] = sum[j] + X[i]
                    num[j] = num[j] + 1
                    break
        for j in range(n_clusters):
            centersNew[j] = sum[j] / num[j]
    '''
    while((centersNew != centersOld).any()):
        centersOld = centersNew
        for i in range(int(X.shape[0])):
            for j in range(n_clusters):
                distance[j] = np.linalg.norm(X[i] - centersOld[j])
            distance = list(distance)
            yNew[i] = distance.index(min(distance))

        centersNew = np.zeros([n_clusters, int(X.shape[1])])
        sum = np.zeros([n_clusters, int(X.shape[1])])
        num = np.zeros(n_clusters)
        for i in range(int(X.shape[0])):
            for j in range(n_clusters):
                if yNew[i] == j:
                    sum[j] = sum[j] + X[i]
                    num[j] = num[j] + 1
                    break
        for j in range(n_clusters):
            centersNew[j] = sum[j] / num[j]
    '''
    # ARI
    ari = metrics.adjusted_rand_score(y, yNew)

    # MRI
    mri = metrics.adjusted_mutual_info_score(y, yNew)


    # v-measure
    v_measure = metrics.v_measure_score(y, yNew)

    # silhouette_avg
    silhouette_avg = metrics.silhouette_score(X, yNew, metric='euclidean')

    print("Cost time is " + str(tEnd - tStart))
    # =======================================
    return [ari, mri, v_measure, silhouette_avg]  # You won't need this line when you are done

def main():
    # Load the dataset
    fname_img = 't10k-images.idx3-ubyte'
    fname_lbl = 't10k-labels.idx1-ubyte'
    [X, y]=get_labeled_data(fname_img, fname_lbl)
    print(X.shape)


    # Plot the mean image
    plot_mean_image(X, 'all images')

    # =======================================
    # Complete the code here.
    # Use PCA to reduce the dimension here.
    # You may want to use the following codes. Feel free to write in your own way.
    # - pca = PCA(n_components=...)
    # - pca.fit(X)
    # - print('We need', pca.n_components_, 'dimensions to preserve 0.95 POV')
    file = open('description2.txt', 'w')

    file.write('The number of samples is ')
    file.write(str(X.shape[0]))
    file.write(' and the number of features is ')
    file.write(str(X.shape[1]))
    file.close()

    n_components = 28 * 28
    pca = PCA(n_components = n_components)
    pca.fit(X)
    pov = np.zeros(n_components)
    pov[0] = abs(pca.explained_variance_ratio_[0])
    for i in range(n_components):
        pov[i] = abs(pca.explained_variance_ratio_[i]) + pov[i - 1]
        if(pov[i] >= 0.95):
            break
    print('We need', i + 1, 'dimensions to preserve 0.95 POV')
    X_transform = pca.transform(X)
    X_dimension_reduced = X_transform[:, 0: i + 1]
    X = np.zeros((int(X_dimension_reduced.shape[0]), i))
    X = X_dimension_reduced

    # =======================================

    # Clustering
    range_n_clusters = [8, 9, 10, 11, 12]
    ari_score = [None] * len(range_n_clusters)
    mri_score = [None] * len(range_n_clusters)
    v_measure_score = [None] * len(range_n_clusters)
    silhouette_avg = [None] * len(range_n_clusters)

    for n_clusters in range_n_clusters:
        i = n_clusters - range_n_clusters[0]
        print("Number of clusters is: ", n_clusters)
        tStart = time.time()
        [ari_score[i], mri_score[i], v_measure_score[i], silhouette_avg[i]] = my_clustering(X, y, n_clusters, pca)
        tEnd = time.time()
        print('Cost time is ' + str(tEnd - tStart))
        print('The ARI score is: ', ari_score[i])
        print('The MRI score is: ', mri_score[i])
        print('The v-measure score is: ', v_measure_score[i])
        print('The average silhouette score is: ', silhouette_avg[i])
    # =======================================


    # Complete the code here.
    # Plot scores of all four evaluation metrics as functions of n_clusters.
    plt.figure()
    plt.plot(range_n_clusters, ari_score)
    plt.plot(range_n_clusters, mri_score)
    plt.plot(range_n_clusters, v_measure_score)
    plt.plot(range_n_clusters, silhouette_avg)
    plt.savefig('result.png')
    plt.show()


    # system k-means
    ari_score_system = [None] * len(range_n_clusters)
    mri_score_system = [None] * len(range_n_clusters)
    v_measure_score_system = [None] * len(range_n_clusters)
    silhouette_avg_system = [None] * len(range_n_clusters)
    for n_clusters in range_n_clusters:
        k_means = KMeans(n_clusters=n_clusters)
        k_means.fit_predict(X)
        # ARI
        ari_score_system[n_clusters - 8] = metrics.adjusted_rand_score(y, k_means.labels_)

        # MRI
        mri_score_system[n_clusters - 8] = metrics.adjusted_mutual_info_score(y, k_means.labels_)

        # v-measure
        v_measure_score_system[n_clusters - 8] = metrics.v_measure_score(y, k_means.labels_)

        # silhouette_avg
        silhouette_avg_system[n_clusters - 8] = metrics.silhouette_score(X, k_means.labels_, metric='euclidean')
    plt.figure()
    plt.plot(range_n_clusters, ari_score_system)
    plt.plot(range_n_clusters, mri_score_system)
    plt.plot(range_n_clusters, v_measure_score_system)
    plt.plot(range_n_clusters, silhouette_avg_system)
    plt.savefig('system.png')
    plt.show()
    # =======================================

if __name__ == '__main__':
    main()
