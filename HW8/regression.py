import numpy as np
import csv
import random


# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_dataset(filename):
    """takes a filename and returns the data as described below in an n-by-(m+1) array
    INPUT:
        filename - a string representing the path to the csv file.

    RETURNS:
        An n by m+1 array, where n is # data points and m is # features.
        The labels y should be in the first column.
    """
    with open(filename) as f:
        return np.array([list(map(float, col)) for col in list(csv.reader(f))[1:]])[:, 1:]


def print_stats(dataset, col):
    """takes the dataset as produced by the previous function and prints several statistics
    about a column of the dataset; does not return anything
    INPUT:
        dataset - the body fat n by m+1 array
        col     - the index of feature to summarize on. 
                  For example, 1 refers to density.

    RETURNS:
        None
    """
    print(len(dataset))
    feature_col = dataset[:, col]
    print(round(np.mean(feature_col), 2))
    print(round(np.std(feature_col), 2))


def regression(dataset, cols, betas):
    """calculates and returns the mean squared error on the dataset given fixed betas
    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        mse of the regression model
    """
    total = 0
    n = len(dataset)
    len_cols = len(cols)
    for i in range(n):
        partial_mse = 0
        for j in range(len_cols):
            partial_mse += betas[j + 1] * dataset[i][cols[j]]
        partial_mse += betas[0]
        partial_mse -= dataset[i][0]
        total += pow(partial_mse, 2)
    return total / n


def gradient_descent(dataset, cols, betas):
    """performs a single step of gradient descent on the MSE and returns the derivative
    values as an 1D array
    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]

    RETURNS:
        An 1D array of gradients
    """
    n = len(dataset)
    len_cols = len(cols)
    grad = []
    for idx in range(len(betas)):
        total = 0
        for i in range(n):
            partial_mse = 0
            for j in range(len_cols):
                partial_mse += betas[j + 1] * dataset[i][cols[j]]
            partial_mse += betas[0]
            partial_mse -= dataset[i][0]
            total += partial_mse * (1 if idx == 0 else dataset[i][cols[idx - 1]])
        grad.append(2 * total / n)
    return np.array(grad)


def iterate_gradient(dataset, cols, betas, T, eta):
    """performs T iterations of gradient descent starting at the given betas and prints the
    results; does not return anything
    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
        T       - # iterations to run
        eta     - learning rate

    RETURNS:
        None
    """
    new_betas = betas
    n = len(new_betas)
    for i in range(1, T + 1):
        gd = gradient_descent(dataset, cols, new_betas)
        new_betas = [(new_betas[j] - eta * gd[j]) for j in range(n)]
        print(i, "{:.2f}".format(round(regression(dataset, cols, new_betas), 2)), end=" ")
        print(" ".join(["{:.2f}".format(round(beta, 2)) for beta in new_betas]))


def compute_betas(dataset, cols):
    """using the closed-form solution, calculates and returns the values of betas and the
    corresponding MSE as a tuple
    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.

    RETURNS:
        A tuple containing corresponding mse and several learned betas
    """
    xs = np.c_[np.ones(len(dataset)), dataset[:, cols]]
    ys = dataset[:, 0]
    betas = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(xs), xs)), np.transpose(xs)), ys)
    mse = regression(dataset, cols, betas)
    return mse, *betas


def predict(dataset, cols, features):
    """using the closed-form solution betas, return the predicted body fat percentage of the give
    features.
    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        features- a list of observed values

    RETURNS:
        The predicted body fat percentage value
    """
    return np.dot(compute_betas(dataset, cols)[1:], [1, *features])


def random_index_generator(min_val, max_val, seed=42):
    """
    DO NOT MODIFY THIS FUNCTION.
    DO NOT CHANGE THE SEED.
    This generator picks a random value between min_val and max_val,
    seeded by 42.
    """
    random.seed(seed)
    while True:
        yield random.randrange(min_val, max_val)


def sgd(dataset, cols, betas, T, eta):
    """performs stochastic gradient descent, prints results as in function 5
    You must use random_index_generator() to select individual data points.
    INPUT:
        dataset - the body fat n by m+1 array
        cols    - a list of feature indices to learn.
                  For example, [1,8] refers to density and abdomen.
        betas   - a list of elements chosen from [beta0, beta1, ..., betam]
        T       - # iterations to run
        eta     - learning rate

    RETURNS:
        None
    """
    new_betas = betas
    len_betas = len(new_betas)
    len_dataset = len(dataset)
    len_cols = len(cols)
    generator = random_index_generator(0, len_dataset)
    for i in range(1, T + 1):
        chosen_idx = next(generator)
        gd = []
        for idx in range(len_betas):
            total = 0
            for j in range(len_cols):
                total += new_betas[j + 1] * dataset[chosen_idx][cols[j]]
            total += new_betas[0]
            total -= dataset[chosen_idx][0]
            total *= 1 if idx == 0 else dataset[chosen_idx][cols[idx - 1]]
            gd.append(2 * total)
        new_betas = [(new_betas[j] - eta * gd[j]) for j in range(len_betas)]
        print(i, "{:.2f}".format(round(regression(dataset, cols, new_betas), 2)), end=" ")
        print(" ".join(["{:.2f}".format(round(beta, 2)) for beta in new_betas]))


if __name__ == '__main__':
    pass
