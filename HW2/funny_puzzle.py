import numpy as np
import heapq


def manhattan(state):
    """returns the manhattan distance of the current state
    :param state: the current state
    :return: the manhattan distance
    """

    # A mapper for the coordinates
    mapper = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]

    distance = 0
    for i in range(9):
        current = state[i]
        if current != 0 and current != i + 1:
            origin = mapper[current - 1]
            target = mapper[i]
            # Calculate the distance from the target point
            distance += abs(origin[0] - target[0]) + abs(origin[1] - target[1])
    return distance


def successors(state):
    """returns the successors of the current state
    :param state: the current state
    :return: a list of successors
    """

    res = []
    a = np.array(state).reshape(3, 3)
    # Find the coordinate of 0
    i, j = np.where(a == 0)
    i = i[0]
    j = j[0]
    index_of_zero = i * 3 + j
    # Able to move left ?
    if j > 0:
        successor = state[:]
        index_of_target = i * 3 + j - 1
        swap(successor, index_of_zero, index_of_target)
        res.append(successor)
    # Able to move up ?
    if i > 0:
        successor = state[:]
        index_of_target = (i - 1) * 3 + j
        swap(successor, index_of_zero, index_of_target)
        res.append(successor)
    # Able to move right ?
    if j < 2:
        successor = state[:]
        index_of_target = i * 3 + j + 1
        swap(successor, index_of_zero, index_of_target)
        res.append(successor)
    # Able to move down ?
    if i < 2:
        successor = state[:]
        index_of_target = (i + 1) * 3 + j
        swap(successor, index_of_zero, index_of_target)
        res.append(successor)
    res.sort()
    return res


def swap(state, src, dest):
    """swap the content of src and dest
    :param state: the current state
    :param src: the src index
    :param dest: the dest index
    :return: None
    """

    state[src], state[dest] = state[dest], state[src]


def print_succ(state):
    """given a state of the puzzle, represented as a single list
    of integers with a 0 in the empty space, print to the console
    all of the possible successor states
    :param state: the current state
    :return: None
    """

    res = successors(state)
    for successor in res:
        print(successor, end="")
        print(" h=" + str(manhattan(successor)))


def solve(state):
    """given a state of the puzzle, perform the A* search algorithm
    and print the path from the current state to the goal state
    :param state: the start state
    :return: None
    """

    open = []
    closed = []
    visited = []
    g = 0
    h = manhattan(state)
    heapq.heappush(open, (g + h, state, (g, h, -1)))
    find = False
    while open:
        temp = heapq.heappop(open)
        closed.append(temp)
        visited.append(temp[1])

        # Checking the goal state
        if manhattan(temp[1]) == 0:
            find = True
            break

        # Calculate the g
        g = temp[2][0] + 1

        for successor in successors(temp[1]):

            h = manhattan(successor)
            # If not in visited, push to open
            if successor not in visited:
                heapq.heappush(open, (g + h, successor, (g, h, len(closed) - 1)))

    # if find, print the result by parent index
    if find:
        backtrack = [len(closed) - 1]
        parent_index = closed[-1][2][2]
        while parent_index != -1:
            backtrack.append(parent_index)
            parent_index = closed[parent_index][2][2]
        backtrack.reverse()
        move = 0
        for i in backtrack:
            print(closed[i][1], end="")
            print(" h=" + str(closed[i][2][1]) + " moves: " + str(move))
            move += 1
