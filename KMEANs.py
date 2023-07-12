import imageio
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix
from scipy.spatial.distance import cdist
import timeit

# np.random.seed(42)


def KMEANs_L2(image, k, image_name, plot=False):
    ''''
    pixels: an RGB image  (L*W*3)
    k : number of clusters

    output:
        lables for each pixel as array
        centroid of each cluster
    '''
    # convert image into matrix (L*W, 3)
    pixels = image.reshape(-1, image.shape[2])

    flag = False
    errors = []
    # Randomly initialize centroids with data points;
    c = pixels[np.random.randint(pixels.shape[0], size=(1, k))[0], :].astype(float)

    iterno = 300
    start = timeit.default_timer()
    for iter in range(0, iterno):
        c_old = c
        # compute the pairewise L2 distnace
        tmpdiff = cdist(pixels, c, 'euclidean')
        # assign the point to a cluster
        labels = np.argmin(tmpdiff, axis=1)

        # parts from code below taken from demo code
        # Update data assignment matrix;
        m = pixels.shape[0]
        P = csc_matrix((np.ones(m), (np.arange(0, m, 1), labels)), shape=(m, k))
        count = P.sum(axis=0)

        # if a cluster has no points, break
        if np.min(count) == 0:
            # print(count)
            flag = True
            break

        # x*P implements summation of data points assigned to a given cluster.
        c = np.array((P.T.dot(pixels)).T / count).T
        # compute how much a centroid changes
        e = np.linalg.norm(c - c_old, ord='fro')
        errors.append(e)
        # if converged then exit
        if e <= 1e-6:
            break

    if flag: # if a cluster has no points, run the algorithm again with lesser k
        return KMEANs_L2(pixels, k - 1, plot=plot)
    else:
        stop = timeit.default_timer()
        print('Time to converge: ', stop - start, 'seconds')
        print('iterations to converge:- ', iter)

    plt.figure(0)
    plt.plot(range(len(errors)), errors)
    plt.xlabel('iteration')
    plt.ylabel('error')
    plt.title('convergence')
    plt.savefig(f'L2_{image_name}_convergence_k_{k}.png')

    if plot:
        plt.show()

    return labels+1, c


def KMEANs_L1(image, k, image_name, plot=False):
    ''''
    pixels: an RGB image (L*W*3)
    k : number of clusters

    output:
        lables for each pixel as array
        centroid of each cluster
    '''
    # convert image into matrix (L*W, 3)
    pixels = image.reshape(-1, image.shape[2])


    errors = []
    # Randomly initialize centroids with data points;
    c = pixels[np.random.randint(pixels.shape[0], size=(1, k))[0], :].astype(float)

    iterno = 300
    start = timeit.default_timer()
    for iter in range(0, iterno):
        c_old = c
        # compute the pairewise L1 distnace
        tmpdiff = cdist(pixels, c, 'cityblock')
        # assign the point to a cluster
        labels = np.argmin(tmpdiff, axis=1)

        # parts from code below taken from demo code
        # Update data assignment matrix;
        m = pixels.shape[0]
        P = csc_matrix((np.ones(m), (np.arange(0, m, 1), labels)), shape=(m, k)).toarray()

        centroids = []
        for i in range(P.shape[1]):
            # compute the median of poiints as new centroid
            centroids.append(np.median(pixels[P[:, i] == 1], axis=0))

        # create new centoid materix
        c = np.vstack(centroids)

        # compute how much a centroid changes
        e = np.linalg.norm(c - c_old, ord='fro')
        errors.append(e)
        if e <= 1e-6:
            break

    stop = timeit.default_timer()
    print('Time to converge: ', stop - start, 'seconds')
    print('iterations to converge:- ', iter)


    plt.figure(0)
    plt.plot(range(len(errors)), errors)
    plt.xlabel('iteration')
    plt.ylabel('error')
    plt.title('convergence')
    plt.savefig(f'L1_{image_name}_convergence_k_{k}.png')

    if plot:
        plt.show()


    return labels+1, c

def main():
    image_name = 'kids'
    k = 8

    path = os.getcwd() + f'\\data\\{image_name}.bmp'
    image = imageio.v2.imread(path)

    classes, centroids = KMEANs_L2(image, k, image_name=image_name, plot=True)
    print('num clusters:-', np.max(classes))

    # prepare the compressed_image for plotting (coloring it)
    compressed_image = []
    for i in classes:
        compressed_image.append(centroids[i-1])

    compressed_image = np.array(compressed_image)
    compressed_image = compressed_image.reshape(image.shape[0], image.shape[1], 3).astype(int)
    plt.figure(1)
    plt.imshow(compressed_image)
    plt.title('Compressed image')
    plt.savefig(f'L2_{image_name}_compressed_k_{k}.png')
    plt.show()

if __name__ == "__main__":
    main()

