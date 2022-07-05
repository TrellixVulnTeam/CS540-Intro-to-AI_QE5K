import numpy as np


def fill(state, max, which):
    """returns a copy of state which fills the jug
    corresponding to the index in which (0 or 1)
    to its maximum capacity. Do not modify state
    :param state: current state
    :param max: maximum capacity for each pug
    :param which: which one to fill
    :return: a new state after modifying
    """

    # Copy the state and fill the specific pug
    new_state = state[:]
    new_state[which] = max[which]
    return new_state


def empty(state, max, which):
    """returns a copy of state which empties the jug
    corresponding to the index in which (0 or 1).
    Do not modify state
    :param state: current state
    :param max: maximum capacity for each pug
    :param which: which one to clear
    :return: a new state after modifying
    """

    # Copy the state and empty the specific pug
    new_state = state[:]
    new_state[which] = 0
    return new_state


def xfer(state, max, source, dest):
    """returns a copy of state which pours the contents
    of the jug at index source into the jug at index dest,
    until source is empty or dest is full.
    Do not modify state
    :param state: current state
    :param max: maximum capacity for each pug
    :param source: which to retrieve
    :param dest: which to pour
    :return: a new state after modifying
    """

    new_state = state[:]
    # If dest still have enough space, clear clear source and pour into dest
    # Otherwise make dest full and subtract from source by previous empty space in dest
    if max[dest] - state[dest] > state[source]:
        new_state[dest] += new_state[source]
        new_state[source] = 0
    else:
        new_state[source] -= max[dest] - new_state[dest]
        new_state[dest] = max[dest]
    return new_state


def succ(state, max):
    """prints the list of unique successor states of the
    current state in any order. This function will generate
    the unique successor states of the current state by
    applying fill, empty, xfer operations on the current state.
    :param state: current state
    :param max:maximum capacity for each pug
    """

    # Create the succ state list by calling functions under every circumstance
    res = np.array([fill(state, max, 0), fill(state, max, 1), empty(state, max, 0), empty(state, max, 1),
                    xfer(state, max, 0, 1), xfer(state, max, 1, 0)])
    # Print the list after deleting the duplications
    print(np.unique(res, axis=0).tolist())
