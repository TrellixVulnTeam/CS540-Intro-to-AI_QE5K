import csv
import numpy as np
import scipy.cluster.hierarchy


def load_data(filepath):
    """takes in a string with a path to a CSV file formatted as in the link above,
    and returns the first 20 data points (without the Generation and Legendary
    columns but retaining all other columns) in a single structure.
    :param filepath: the path to the file
    :return: the first 20 data points
    """

    with open(filepath, encoding="utf-8") as csv_file:
        res = list(csv.DictReader(csv_file))[0:20]
        non_convert = ("Name", "Type 1", "Type 2")
        for item in res:
            del item["Generation"]
            del item["Legendary"]
            for key in item:
                if key not in non_convert:
                    item[key] = int(item[key])
        return res


def calculate_x_y(stats):
    """takes in one row from the data loaded from the previous function, calculates
    the corresponding x, y values for that Pokemon as specified above, and returns
    them in a single structure
    :param stats: a pokemon dict
    :return: feature values as a tuple
    """

    x = stats["Attack"] + stats["Sp. Atk"] + stats["Speed"]
    y = stats["Defense"] + stats["Sp. Def"] + stats["HP"]
    return x, y


def hac(dataset):
    """performs single linkage hierarchical agglomerative clustering on the Pokemon with
    the (x,y) feature representation, and returns a data structure representing the clustering.
    :param dataset: pokemon dataset
    :return: the clustering
    """

    z = []
    clusters = []
    m = len(dataset)
    for i in range(m):
        clusters.append([i, [dataset[i]]])
    iteration = 0
    while iteration < m - 1:
        distance = None
        target1 = None
        target2 = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distance_list = []
                for point1 in clusters[i][1]:
                    for point2 in clusters[j][1]:
                        distance_list.append(cal_dis(point1, point2))
                single = min(distance_list)
                if distance is None or single < distance:
                    target1 = clusters[i]
                    target2 = clusters[j]
                    distance = single
        new_cluster = []
        new_cluster.extend(target1[1])
        new_cluster.extend(target2[1])
        clusters.remove(target1)
        clusters.remove(target2)
        clusters.append([m + iteration, new_cluster])
        z.append([target1[0], target2[0], distance, len(new_cluster)])
        iteration += 1
    return np.array(z).reshape(m - 1, 4)


def cal_dis(point1, point2):
    """calculates the distance between two clusters
    :param point1: point1
    :param point2: point2
    :return: single-linkage
    """

    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

res1 = hac(list(map(calculate_x_y, load_data("Pokemon.csv"))))
res2 = scipy.cluster.hierarchy.linkage(list(map(calculate_x_y, load_data("Pokemon.csv"))))
print(res1)
print(res2)