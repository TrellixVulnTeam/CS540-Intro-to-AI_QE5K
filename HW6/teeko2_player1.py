import random
import copy


class Teeko2Player:
    """ An object representation for an AI game player for the game Teeko2.
    """

    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a Teeko2Player object by randomly selecting red or black as its
        piece color.
        """
        self.board = [[' ' for j in range(5)] for i in range(5)]
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this Teeko2Player object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """
        drop_phase = self.is_drop_phrase(state)
        move = []
        successors = self.succ(state, self.my_piece)
        score = None
        selected = []
        for successor in successors:
            temp_score, depth = self.Min_Value(successor, 0, (float("-inf"), float("inf")), (float("inf"), float("inf")))
            if not score or temp_score > score:
                score = temp_score
                selected = [(depth, successor)]
            elif temp_score == score:
                selected.append((depth, successor))
        selected = sorted(selected)[0][1]

        if not drop_phase:
            # choose a piece to move and remove it from the board
            # (You may move this condition anywhere, just be sure to handle it)
            #
            # Until this part is implemented and the move list is updated
            # accordingly, the AI will not follow the rules after the drop phase!
            move.append((selected[1], selected[2]))
            move.append((selected[3], selected[4]))

        else:
            # select an unoccupied space randomly
            move.append((selected[1], selected[2]))

        # ensure the destination (row,col) tuple is at the beginning of the move list
        return move

    def Max_Value(self, state, depth, a, b):
        """ Max function of Minimax algorithm
        :param state: the current state
        :param depth: the current depth
        :param a: a value
        :param b: b value
        :return: score
        """
        value = self.game_value(state[0])
        if value == 1 or value == -1:
            return value, depth
        if depth == 2:
            return self.heuristic_game_value(state[0]), depth
        for successor in self.succ(state[0], self.my_piece):
            a = max(a, self.Min_Value(successor, depth + 1, a, b))
            if a >= b:
                return b
        return a

    def Min_Value(self, state, depth, a, b):
        """ Min function of Minimax algorithm
        :param state: the current state
        :param depth: the current depth
        :param a: a value
        :param b: b value
        :return: score
        """
        value = self.game_value(state[0])
        if value == 1 or value == -1:
            return value, depth
        if depth == 2:
            return self.heuristic_game_value(state[0]), depth
        for successor in self.succ(state[0], self.opp):
            b = min(b, self.Max_Value(successor, depth + 1, a, b))
            if a >= b:
                return a
        return b

    def is_drop_phrase(self, state):
        """ Detect whether this state is drop phrase
        :param state: current state
        :return: True is is drop phrase, False otherwise
        """
        return sum([i.count(self.my_piece) for i in state]) < 4

    def succ(self, state, mark):
        """ A successor function (e.g. succ(state)) that takes in a board state
        and returns a list of the legal successors. During the drop phase, this
        simply means adding a new piece of the current player's type to the board;
        during continued gameplay, this means moving any one of the current player's
        pieces to an unoccupied location on the board, adjacent to that piece.
        :return: a list of legal successors
        """
        res = []
        drop_phase = self.is_drop_phrase(state)
        if drop_phase:
            for i in range(5):
                for j in range(5):
                    if state[i][j] == " ":
                        temp = copy.deepcopy(state)
                        temp[i][j] = mark
                        res.append((temp, i, j))
        else:
            offset = [-1, 0, 1]
            for i in range(5):
                for j in range(5):
                    if state[i][j] == mark:
                        for x in offset:
                            for y in offset:
                                if -1 < i + x < 5 and -1 < j + y < 5 and state[i + x][j + y] == " ":
                                    temp = copy.deepcopy(state)
                                    temp[i + x][j + y] = mark
                                    temp[i][j] = " "
                                    res.append((temp, i + x, j + y, i, j))
        return res

    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row is not None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row) + ": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this Teeko2Player object, or a generated successor state.

        Returns:
            int: 1 if this Teeko2Player wins, -1 if the opponent wins, 0 if no winner

        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i + 1] == row[i + 2] == row[i + 3]:
                    return 1 if row[i] == self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i + 1][col] == state[i + 2][col] == state[i + 3][col]:
                    return 1 if state[i][col] == self.my_piece else -1

        # check \ diagonal wins
        for row in range(2):
            for col in range(2):
                if state[row][col] != " " and state[row][col] == state[row + 1][col + 1] == state[row + 2][col + 2] == state[row + 3][col + 3]:
                    return 1 if state[row][col] == self.my_piece else -1

        # check / diagonal wins
        for row in range(2):
            for col in range(3, 5):
                if state[row][col] != " " and state[row][col] == state[row + 1][col - 1] == state[row + 2][col - 2] == state[row + 3][col - 3]:
                    return 1 if state[row][col] == self.my_piece else -1

        # check diamond wins
        for row in range(3):
            for col in range(1, 4):
                if state[row][col] != " " and state[row + 1][col] == " ":
                    if state[row][col] == state[row + 1][col - 1] == state[row + 1][col + 1] == state[row + 2][col]:
                        return 1 if state[row][col] == self.my_piece else -1

        return 0  # no winner yet

    def heuristic_game_value(self, state):
        """ This function should return some float value between 1 and -1
        :return: heuristic value
        """
        return 0.2 * self.heuristic_score(state, self.my_piece) - 0.8 * self.heuristic_score(state, self.opp)

    def heuristic_score(self, state, mark):
        """ Return the estimated heuristic for specific index
        :param state: the current state
        :param mark: mark of current player
        :return: the estimated score
        """
        res = [0.0]

        # check horizontal score
        for row in range(5):
            for col in range(2):
                score = 0
                for offset in range(0, 4):
                    if state[row][col + offset] == mark:
                        score += 0.0002
                if score == 0.0004:
                    score += 0.0004
                if score == 0.0006:
                    score += 0.0006
                res.append(score)

        # check vertical score
        for col in range(5):
            for row in range(2):
                score = 0
                for offset in range(0, 4):
                    if state[row + offset][col] == mark:
                        score += 0.0002
                if score == 0.0004:
                    score += 0.0004
                if score == 0.0006:
                    score += 0.0006
                res.append(score)

        # check \ diagonal score
        for row in range(2):
            for col in range(2):
                score = 0
                for offset in range(0, 4):
                    if state[row + offset][col + offset] == mark:
                        score += 0.0002
                if score == 0.0004:
                    score += 0.0004
                if score == 0.0006:
                    score += 0.0006
                res.append(score)

        # check / diagonal score
        for row in range(2):
            for col in range(3, 5):
                score = 0
                for offset in range(0, 4):
                    if state[row + offset][col - offset] == mark:
                        score += 0.0002
                if score == 0.0004:
                    score += 0.0004
                if score == 0.0006:
                    score += 0.0006
                res.append(score)

        # check diamond score
        for row in range(3):
            for col in range(1, 4):
                score = 0
                if state[row][col] == mark:
                    score += 0.0002
                if state[row + 1][col - 1] == mark:
                    score += 0.0002
                if state[row + 1][col + 1] == mark:
                    score += 0.0002
                if state[row + 2][col] == mark:
                    score += 0.0002
                if score == 0.0004:
                    score += 0.0004
                if score == 0.0006:
                    score += 0.0006
                res.append(score)

        return sum(res)


############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################

def main():
    ai = Teeko2Player()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:
        mapper = "ABCDE"
        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
        else:
            move_made = False
            ai.print_board()
            while not move_made:
                player_move = "ABCDE"[random.randint(0, 4)] + str(random.randint(0, 4))
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0]) - ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
        else:
            move_made = False
            res = []
            offset = [-1, 0, 1]
            for i in range(5):
                for j in range(5):
                    if ai.board[i][j] == ai.opp:
                        for x in offset:
                            for y in offset:
                                if -1 < i + x < 5 and -1 < j + y < 5 and ai.board[i + x][j + y] == " ":
                                    temp = copy.deepcopy(ai.board)
                                    temp[i + x][j + y] = ai.opp
                                    temp[i][j] = " "
                                    res.append((temp, i + x, j + y, i, j))
            selected = random.choice(res)
            while not move_made:
                move_from = mapper[selected[4]] + str(selected[3])
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = mapper[selected[2]] + str(selected[1])
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0]) - ord("A")),
                                      (int(move_from[1]), ord(move_from[0]) - ord("A"))])
                    move_made = True
                except Exception as e:
                    pass

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        return 1, 0
    else:
        return 0, 1


if __name__ == "__main__":
    ai = 0
    opp = 0
    i = 0
    while i < 3:
        s, q = main()
        ai += s
        opp += q
        i += 1
    print(ai, opp)