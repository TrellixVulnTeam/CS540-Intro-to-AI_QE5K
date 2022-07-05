def manhattan_distance(data_point1, data_point2):
    """return the Manhattan distance between two
    dictionary data points from the data set
    :param data_point1: point 1
    :param data_point2: point 2
    :return: manhattan_distance
    """

    # Calculate the manhattan_distance
    return abs(data_point1['PRCP'] - data_point2['PRCP']) + abs(data_point1['TMAX'] - data_point2['TMAX']) + abs(
        data_point1['TMIN'] - data_point2['TMIN'])


def read_dataset(filename):
    """ return a list of data point dictionaries
    read from the specified file
    :param filename: the name of the file
    :return: a list of data dictionaries
    """

    res = []
    with open(filename) as file_object:
        # Read each line
        for line in file_object:
            dic = {}
            # Split by space
            words = line.split(' ')
            dic['DATE'] = words[0]
            dic['PRCP'] = float(words[1])
            dic['TMAX'] = float(words[2])
            dic['TMIN'] = float(words[3])
            dic['RAIN'] = words[4].strip()
            res.append(dic)
    return res


def majority_vote(nearest_neighbors):
    """return a prediction of whether it is raining
    or not based on a majority vote of the list of neighbors.
    :param nearest_neighbors: list of nearest neighbors
    :return: the majority vote
    """

    res = 0
    # Calculate the sum of TRUE and FALSE
    for neighbors in nearest_neighbors:
        if neighbors['RAIN'] == 'TRUE':
            res += 1
        else:
            res -= 1
    return 'TRUE' if res >= 0 else 'FALSE'


def k_nearest_neighbors(filename, test_point, k, year_interval):
    """using the above functions, return the majority vote prediction
    for whether it's raining or not on the provided test point.
    :param filename: the name of the file
    :param test_point: test point
    :param k: num of accepted neighbors
    :param year_interval: accepted year interval
    :return: the majority vote
    """

    # Retrieve the current year
    current_date = int(test_point['DATE'].split('-')[0])
    # Get the test_point that lies within year interval
    dataset = [dic for dic in read_dataset(filename) if
               abs(int(dic['DATE'].split('-')[0]) - current_date) < year_interval]

    # find the closest k valid neighbors and return the result
    return majority_vote(sorted(dataset,
                                key=lambda dic: manhattan_distance(dic, test_point))[:k])


def main():
    name1 = k_nearest_neighbors("rain.txt", {'DATE': '1948-01-01', 'TMAX': 51.0, 'PRCP': 0.47, 'TMIN': 42.0}, 2, 10)
    name2 = k_nearest_neighbors("rain.txt", {'DATE': '1998-01-01', 'TMAX': 55.0, 'PRCP': 0.47, 'TMIN': 34.0}, 2, 10)
    name3 = k_nearest_neighbors("rain.txt", {'DATE': '2004-01-01', 'TMAX': 34.0, 'PRCP': 0.23, 'TMIN': 13.0}, 5, 5)
    name4 = k_nearest_neighbors("rain.txt", {'DATE': '1944-01-01', 'TMAX': 38.0, 'PRCP': 0.55, 'TMIN': 24.0}, 10, 10)
    name5 = k_nearest_neighbors("rain.txt", {'DATE': '1987-02-14', 'TMAX': 45.0, 'PRCP': 0.76, 'TMIN': 34.0}, 3, 10)
    name6 = k_nearest_neighbors("rain.txt", {'DATE': '1987-05-01', 'TMAX': 53.0, 'PRCP': 0.45, 'TMIN': 32.0}, 10, 15)
    print(name1)
    print(name2)
    print(name3)
    print(name4)
    print(name5)
    print(name6)


main()
