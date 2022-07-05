import heapq
import numpy as np
import copy


def manhattan(state):
    """returns the manhattan distance of the current state
    :param state: the current state
    :return: the manhattan distance
    """

    # A mapper for the coordinates
    mapper = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]]

    distance = 0
    for i in range(0, 9):
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

            # If not in both or in closed and lower but not in open, push to open
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


def tile_manhattan(cur_index_array, goal_index_array):
    '''
    given two 2d index array, one represent cur state index and other
    one represent goal index array for certain tile, return the
    manhattan distance for such tile.
    '''
    return abs(cur_index_array[0][0] - goal_index_array[0][0]) + abs(cur_index_array[0][1] - goal_index_array[0][1])


def total_manhattan(state):
    '''
    given a state, return the sum of Manhattan distance of each tile to
    its goal position as our heuristic function.
    '''
    total_value = 0
    goal_state = np.array([1, 2, 3, 4, 5, 6, 7, 8, 0]).reshape(3, 3)
    cur_state = np.array(state).reshape(3, 3)
    for x in range(1, 9):
        cur_index_array = np.argwhere(cur_state == x)
        goal_index_array = np.argwhere(goal_state == x)
        total_value += tile_manhattan(cur_index_array, goal_index_array)
    return (total_value)


def puzzle_move(target_puzzle, cur_state):
    '''
    given puzzle to be move and cur_state in 2d numpy array.
    return a list represent a state after moved given puzzle
    '''
    temp_state = copy.deepcopy(cur_state)
    target_index = np.argwhere(temp_state == target_puzzle)
    zero_index = np.argwhere(temp_state == 0)
    temp_state[target_index[0][0]][target_index[0][1]] = 0
    temp_state[zero_index[0][0]][zero_index[0][1]] = target_puzzle
    return (temp_state.flatten().tolist())


def print_succ1(state):
    '''
    given a state of the puzzle, represented as a single list of integers with a 0
    in the empty space, print to the console all of the possible successor states
    '''
    list_of_successors = successor_helper(state)
    for successors in list_of_successors:
        print(successors, end=(" h=" + str(total_manhattan(successors)) + "\n"))


def successor_helper(state):
    state_2d = np.array(state).reshape(3, 3)
    moveable_puzzle = []
    list_of_successors = []
    zero_index = np.argwhere(state_2d == 0)

    # test if puzzle around 0 is moveable
    if zero_index[0][0] + 1 < 3:
        moveable_puzzle.append(state_2d[zero_index[0][0] + 1][zero_index[0][1]])
    if zero_index[0][1] + 1 < 3:
        moveable_puzzle.append(state_2d[zero_index[0][0]][zero_index[0][1] + 1])
    if zero_index[0][0] - 1 >= 0:
        moveable_puzzle.append(state_2d[zero_index[0][0] - 1][zero_index[0][1]])
    if zero_index[0][1] - 1 >= 0:
        moveable_puzzle.append(state_2d[zero_index[0][0]][zero_index[0][1] - 1])

    for puzzles in moveable_puzzle:
        list_of_successors.append(puzzle_move(puzzles, state_2d))
    list_of_successors = sorted(list_of_successors)
    return list_of_successors


def solve1(state):
    '''
    given a state of the puzzle, perform the A* search algorithm and print the
    path from the current state to the goal state
    '''
    pq_open = []
    pq_close = []
    moves = 0
    par_index = -1
    done = False

    heapq.heappush(pq_open, (total_manhattan(state) + moves, state, (moves, total_manhattan(state), par_index)))

    while done != True:

        if len(pq_open) == 0:
            return
        cur = heapq.heappop(pq_open)
        pq_close.append(cur)

        # if done
        if total_manhattan(cur[1]) == 0:
            done = True
            result_list = []
            result_list.insert(0, cur)

            if cur[2][2] != -1:
                temp = pq_close[cur[2][2]]

                while temp[2][2] != -1:
                    result_list.insert(0, temp)
                    temp = pq_close[temp[2][2]]
                result_list.insert(0, temp)

            for steps in result_list:
                print(str(steps[1]) + " h=" + str(steps[2][1]) + " moves: " + str(steps[2][0]), end="\n")

        # if not done
        list_of_successors = successor_helper(cur[1])
        moves = int(cur[2][0] + 1)
        par_index = len(pq_close) - 1
        for successor in list_of_successors:
            presents = False
            # check open
            for item in pq_open:
                if item[1] == successor:
                    presents = True
                    if moves < item[2][0]:
                        heapq.heappush(pq_open, (
                            total_manhattan(successor) + moves, successor,
                            (moves, total_manhattan(successor), par_index)))
                        pq_open.remove(item)

            # check close
            if presents == False:
                for item in pq_close:
                    if item[1] == successor:
                        presents = True
                        if moves < item[2][0]:
                            heapq.heappush(pq_open, (total_manhattan(successor) + moves, successor,
                                                     (moves, total_manhattan(successor), par_index)))
                            pq_close.remove(item)

            # successor is new
            if presents == False:
                heapq.heappush(pq_open, (
                    total_manhattan(successor) + moves, successor, (moves, total_manhattan(successor), par_index)))


def main():
    # [1, 2, 5, 7, 6, 4, 0, 3, 8]
    # solve([8, 6, 7, 2, 5, 4, 3, 0, 1])
    # for i in range(0, 32):
    #     file_name = "puzzle3x3-" + str(i).zfill(2) + ".txt"
    #     state = []
    #     with open(file_name) as file_object:
    #         lines = file_object.readlines()
    #         temp = lines[1].split(" ")
    #         state.append(int(temp[1]))
    #         state.append(int(temp[3]))
    #         state.append(int(temp[5]))
    #         temp = lines[2].split(" ")
    #         state.append(int(temp[1]))
    #         state.append(int(temp[3]))
    #         state.append(int(temp[5]))
    #         temp = lines[3].split(" ")
    #         state.append(int(temp[1]))
    #         state.append(int(temp[3]))
    #         state.append(int(temp[5]))
    #         print(state)
    #     solve(state)
    #     print("-------------------------------------------------------")
    # with open("8.txt") as file_object:
    #     for line in file_object:
    #         state = []
    #         for i in range(9):
    #             state.append(int(line[i]))
    #         solve(state)
    #         print("-------------------------------------------------------")
    list = [1, 2, 5, 7, 6, 4, 0, 3, 8]
    solve(list)
    solve1(list)


main()
