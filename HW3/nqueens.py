import random


def succ(state, static_x, static_y):
    """given a state of the board, return a list of all
    valid successor states
    :param state: the current state
    :param static_x: the static x-position
    :param static_y: the static y-position
    :return: a list of valid successors
    """

    res = []
    # Check whether there is a queen on the static point
    if state[static_x] != static_y:
        return res

    offset = (-1, 1)
    size = len(state)
    # Iterate through the queens
    for i in range(size):
        if i == static_x:
            continue
        for j in offset:
            if -1 < state[i] + j < size:
                su = state[:]
                su[i] = state[i] + j
                res.append(su)
    return sorted(res)


def f(state):
    """given a state of the board, return an integer score such
    that the goal state scores 0
    :param state: the current state
    :return: the score of the current state
    """

    score = 0
    size = len(state)
    for x in range(size):
        i = 1
        while x - i > -1 or x + i < size:
            candidate = (state[x] + i, state[x] - i, state[x])
            if (x - i > -1 and state[x - i] in candidate) or (x + i < size and state[x + i] in candidate):
                score += 1
                break
            i += 1
    return score


def choose_next(curr, static_x, static_y):
    """given the current state, use succ() to generate the
    successors and return the selected next state
    :param curr: the current state
    :param static_x: the static x-position
    :param static_y: the static y-position
    :return: the selected next state
    """

    # Generate successors
    successors = succ(curr, static_x, static_y)
    if not successors:
        return None
    successors.append(curr)
    for i in range(len(successors)):
        successors[i] = (f(successors[i]), successors[i])
    # Choose the lowest
    return sorted(successors)[0][1]


def n_queens(initial_state, static_x, static_y, print_path=True):
    """run the hill-climbing algorithm from a given initial
    state, return the convergence state
    :param initial_state: the initial state
    :param static_x: the static x-position
    :param static_y: the static y-position
    :param print_path: whether wants to print the path
    :return: None
    """

    current = initial_state
    next_score = f(current)
    while True:
        current_score = next_score
        if print_path:
            print(str(current), "-", "f=" + str(current_score))
        # Ends the execution when finds reaches the target
        if current_score == 0:
            return current
        chose_next = choose_next(current, static_x, static_y)
        next_score = f(chose_next)
        if next_score == current_score:
            if print_path:
                print(str(chose_next), "-", "f=" + str(next_score))
            return chose_next
        current = chose_next


def n_queens_restart(n, k, static_x, static_y):
    """ run the hill-climbing algorithm on an n*n board with
    random restarts
    :param n: board size n
    :param k: runs how many restarts
    :param static_x: the static x-position
    :param static_y: the static y-position
    :return: None
    """

    random.seed(1)

    res_list = []
    times = 0
    while times < k:
        state = []
        for i in range(n):
            if i == static_x:
                state.append(static_y)
                continue
            state.append(random.randint(0, n - 1))
        res = n_queens(state, static_x, static_y, False)
        res_score = f(res)
        if res_score == 0:
            print(str(res), "-", "f=" + str(res_score))
            return
        if not res_list or res_list[0][0] > res_score:
            res_list = [(res_score, res)]
        elif res_list[0][0] == res_score:
            res_list.append((res_score, res))
        times += 1
    for li in sorted(res_list):
        print(str(li[1]), "-", "f=" + str(li[0]))
