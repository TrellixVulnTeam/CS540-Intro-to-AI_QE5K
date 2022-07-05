from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    """load the dataset from a provided .npy file,
    re-center it around the origin and return it as a NumPy array of floats
    :param filename: the filename
    :return: a NumPy array of floats
    """

    x = np.reshape(np.load(filename), (2000, 784))
    return x - np.mean(x, axis=0)


def get_covariance(dataset):
    """calculate and return the covariance matrix
    of the dataset as a NumPy matrix (d x d array)
    :param dataset: the dataset
    :return: a NumPy matrix
    """

    return np.dot(np.transpose(dataset), dataset) / (len(dataset) - 1)


def get_eig(S, m):
    """perform eigen decomposition on the covariance
    matrix S and return a diagonal matrix (NumPy array)
    with the largest m eigenvalues on the diagonal, and
    a matrix (NumPy array) with the corresponding eigenvectors as columns
    :param S: covariance matrix S
    :param m: largest m eigenvalues
    :return: a diagonal matrix
    """

    n = len(S)
    w, v = eigh(S)
    return np.diag(w[-m:][::-1]), v[:, -m:][:, ::-1]


def get_eig_perc(S, perc):
    """similar to get_eig, but instead of returning the first m,
    return all eigenvectors that explains more than perc % of variance
    :param S: covariance matrix S
    :param perc: perc % of variance
    :return: eigenvectors that explains more than perc % of variance
    """

    w, v = eigh(S)
    count = 0
    total = np.sum(w)
    for i in range(len(w) - 1, -1, -1):
        if w[i] / total <= perc:
            break
        count += 1
    return np.diag(w[-count:][::-1]), v[:, -count:][:, ::-1]


def project_image(image, U):
    """project each image into your m-dimensional space and
    return the new representation as a d x 1 NumPy array
    :param image: image
    :param U: eigenvectors
    :return: a d x 1 NumPy array
    """

    return np.dot(np.dot(U, np.transpose(U)), image)


def display_image(orig, proj):
    """use matplotlib to display a visual representation
    of the original image and the projected image side-by-side
    :param orig: original image
    :param proj: projected image
    :return: None
    """

    orig = np.reshape(orig, (28, 28))
    proj = np.reshape(proj, (28, 28))
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(9, 3))
    ax1.set_title("Original")
    ax2.set_title("Projection")
    pos1 = ax1.imshow(orig, aspect="equal", cmap="gray")
    pos2 = ax2.imshow(proj, aspect="equal", cmap="gray")
    fig.colorbar(pos1, ax=ax1)
    fig.colorbar(pos2, ax=ax2)
    plt.show()

x = load_and_center_dataset('mnist.npy')
S = get_covariance(x)
Lambda, U = get_eig(S, 2)
print(np.sum(U))