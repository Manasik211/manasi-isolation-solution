"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
	"""
	Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

	return heuristic5(game, player)

def heuristic1(game, player):
    """With `heuristic1()`, the more available moves `player` has available from the evaluated position, the better.
    This function simply returns the difference in number of legal moves left between the players.
    It `player` and its opponent have the same number of moves, then the returned value is zero.
    If the returned value is positive (negative), then `player` is doing better (worse) than its opponent.
    If the returned value is "inf" ("-inf"), then `player` has won (lost) the game.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.

   """

    # Are we winner?
    if game.is_winner(player):
        return float("inf")

    # Are we losing?
    if game.is_loser(player):
        return float("-inf")

    # We still have moves to play. How many more than our opponent?
    player_moves_left = len(game.get_legal_moves(player))
    opponent_moves_left = len(game.get_legal_moves(game.get_opponent(player)))
    return float(player_moves_left - opponent_moves_left)

def heuristic2(game, player):
    """With this heuristic, the more moves `player` has available from the evaluated position, the better, but not all
    starting positions are equal. If a player's position is closer to the center of the board, it is more probable that
    this player can do better than a player whose remaining moves are near the edge of the board (where they will have
    less options to move down the line).
    It the players have the same number of moves and are at the same distance from the center, then returned value is 0.
    If the returned value is positive (negative), then `player` is doing better (worse) than its opponent.
    If the returned value is "inf" ("-inf"), then `player` has won (lost) the game.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.

    
    """

    # Are we winning?
    if game.is_winner(player):
        return float("inf")

    # Are we losing?
    if game.is_loser(player):
        return float("-inf")

    # We still have moves to play. How many more than our opponent?
    player_moves_left = len(game.get_legal_moves(player))
    opponent_moves_left = len(game.get_legal_moves(game.get_opponent(player)))

    if player_moves_left != opponent_moves_left:
        return float(player_moves_left - opponent_moves_left)

    else:
        """ If we have the same number of moves available, look for a positional advantage and Use the 
		Manhattan distance to the center of the board to assess positional advantage."""
        center_y_pos, center_x_pos = int(game.height / 2), int(game.width / 2)
        player_y_pos, player_x_pos = game.get_player_location(player)
        opponent_y_pos, opponent_x_pos = game.get_player_location(game.get_opponent(player))
        player_distance = abs(player_y_pos - center_y_pos) + abs(player_x_pos - center_x_pos)
        opponent_distance = abs(opponent_y_pos - center_y_pos) + abs(opponent_x_pos - center_x_pos)
        # All we need now is to take the difference between the two distances to evaluate positional advantage.
        # Scale this number between 0 and +-1.
        # In the best possible case, our opponent's distance is 6 from the center (for a 7x7 grid) and we're at pos 0,0 -> return 0.6
        # In the worst possible case, our opponent's distance is 0 from the center (for a 7x7 grid) and we're in a corner -> return -0.6
        # If both players are at the same distance from the center -> return 0.
        return float(opponent_distance - player_distance) / 10.

def heuristic3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    Modify the Open moves score heuristic provided to us by weighting each
    open move by a weight that depends on the position the move leads to
    on the game board.
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """

    # Are we winning?
    
    if game.is_winner(player):
        return float("inf")
    
	# Are we losing?
    if game.is_loser(player):
        return float("-inf")

    # We still have moves to play.

    h, w = game.height, game.width
    score = 0
    player_moves = game.get_legal_moves(player)
    for move in player_moves:
        if move in [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)]:
            score += 2
        elif move in [(0, 1), (0, w-2), (1, 0), (1, w-1), (h-2, 0), (h-2, w-1), (h-1, 1), (h-1, w-2)]:
            score += 3
        elif ((move[0] == 0 or move[0] == h-1) and move[1] >= 2 and move[1] <= w-3) or ((move[1] == 0 or move[1] == w-1) and move[0] >= 2 and move[0] <= h-3):
            score += 4
        elif move in [(1, 1), (1, w-2), (h-2, 1), (h-2, w-2)]:
            score += 4
        elif ((move[0] == 1 or move[0] == h-2) and move[1] >= 2 and move[1] <= w-3) or ((move[1] == 1 or move[1] == w-2) and move[0] >= 2 and move[0] <= h-3):
            score += 6
        else:
            score += 8

    return float(score)

def heuristic4(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    Calculate the difference in the weighted open moves scores between the
    current player and its opponent and use that as the score of the
    current game state.
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """
	# Are we winning?
    if game.is_winner(player):
        return float("inf")

    # Are we losing?
    if game.is_loser(player):
        return float("-inf")

    # We still have moves to play.

    h, w = game.height, game.width
    player_score = 0
    opponent_score = 0
    player_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    for move in player_moves:
        if move in [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)]:
            player_score += 2
        elif move in [(0, 1), (0, w-2), (1, 0), (1, w-1), (h-2, 0), (h-2, w-1), (h-1, 1), (h-1, w-2)]:
            player_score += 3
        elif ((move[0] == 0 or move[0] == h-1) and move[1] >= 2 and move[1] <= w-3) or ((move[1] == 0 or move[1] == w-1) and move[0] >= 2 and move[0] <= h-3):
            player_score += 4
        elif move in [(1, 1), (1, w-2), (h-2, 1), (h-2, w-2)]:
            player_score += 4
        elif ((move[0] == 1 or move[0] == h-2) and move[1] >= 2 and move[1] <= w-3) or ((move[1] == 1 or move[1] == w-2) and move[0] >= 2 and move[0] <= h-3):
            player_score += 6
        else:
            player_score += 8

    for move in opponent_moves:
        if move in [(0, 0), (0, w-1), (h-1, 0), (h-1, w-1)]:
            opponent_score += 2
        elif move in [(0, 1), (0, w-2), (1, 0), (1, w-1), (h-2, 0), (h-2, w-1), (h-1, 1), (h-1, w-2)]:
            opponent_score += 3
        elif ((move[0] == 0 or move[0] == h-1) and move[1] >= 2 and move[1] <= w-3) or ((move[1] == 0 or move[1] == w-1) and move[0] >= 2 and move[0] <= h-3):
            opponent_score += 4
        elif move in [(1, 1), (1, w-2), (h-2, 1), (h-2, w-2)]:
            opponent_score += 4
        elif ((move[0] == 1 or move[0] == h-2) and move[1] >= 2 and move[1] <= w-3) or ((move[1] == 1 or move[1] == w-2) and move[0] >= 2 and move[0] <= h-3):
            opponent_score += 6
        else:
            opponent_score += 8

    #print(player_score, opponent_score, player_score - opponent_score)
    return float(player_score - opponent_score)

def heuristic5(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    The difference in the number of available moves between the current
    player and its opponent one ply ahead in the future is used as the
    score of the current game state.
    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
    player : objects
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)
    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """
	#Are we winning?
    if game.is_winner(player):
        return float("inf")
    # Are we losing?
    if game.is_loser(player):
        return float("-inf")

    
	# We still have moves to play.
    h, w = game.height, game.width
    player_score = 0
    opponent_score = 0
    player_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    for move in player_moves:
        player_score += len(game.__get_moves__(move))

    for move in opponent_moves:
        opponent_score += len(game.__get_moves__(move))

    #print(player_score, opponent_score, player_score - opponent_score)
    return float(player_score - opponent_score)
	

class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        # Are there any legal moves left for us to play? If not, we stop playing!
        if not legal_moves:
            return (-1, -1)

        # pick the center position if we are starting the game.
        if game.move_count == 0:
            return(int(game.height/2), int(game.width/2))

        # Let's search for a good move!
        best_move_so_far = (-1, -1)

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if self.iterative == True:
                iterative_search_depth = 1
                if self.method == 'minimax':
                    while True:
                        best_score_so_far, best_move_so_far = self.minimax(game, iterative_search_depth)
                        if best_score_so_far == float("inf") or best_score_so_far == float("-inf"):
                            break
                        iterative_search_depth += 1
                elif self.method == 'alphabeta':
                    while True:
                        best_score_so_far, best_move_so_far = self.alphabeta(game, iterative_search_depth)
                        if best_score_so_far == float("inf") or best_score_so_far == float("-inf"):
                            break
                        iterative_search_depth += 1
                else:
                    raise ValueError('ERR in CustomPlayer.get_move() - invalid param')
            else:
                if self.method == 'minimax':
                    _, best_move_so_far = self.minimax(game, self.search_depth)
                elif self.method == 'alphabeta':
                    _, best_move_so_far = self.alphabeta(game, self.search_depth)
                else:
                    raise ValueError('ERR in CustomPlayer.get_move() - invalid param')

        except Timeout:
            # Handle any actions required at timeout, if necessary
            pass

        # Return the best move from the last completed search iteration
        return best_move_so_far

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # Are there any legal moves left for us to play? If not, then we lost!
        # The maximizing (minimizing) player returns the lowest (highest) possible score.
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            if maximizing_player == True:
                return float("-inf"), (-1, -1)
            else:
                return float("inf"), (-1, -1)

        
        # If we have reached the target search depth, return the best possible move at this level.
        # For the maximizing (minimizing) player, that would be the move with the highest (lowest) score.
        lowest_score_so_far, highest_score_so_far = float("inf"), float("-inf")
        best_move_so_far = (-1, -1)
        if depth == 1:
            if maximizing_player == True:
                for move in legal_moves:                
                    score = self.score(game.forecast_move(move), self)
                    # If this is a winning move, no need to search further. Otherwise, remember the best move.
                    if score == float("inf"):
                        return score, move
                    if score > highest_score_so_far:
                        highest_score_so_far, best_move_so_far = score, move
                return highest_score_so_far, best_move_so_far
            else:
                for move in legal_moves:                    
                    score = self.score(game.forecast_move(move), self)
                    # If this is a winning move, no need to search further. Otherwise, remember the best move.
                    if score == float("-inf"):
                        return score, move
                    if score < lowest_score_so_far:
                        lowest_score_so_far, best_move_so_far = score, move
                return lowest_score_so_far, best_move_so_far

        # There are still some legal moves and we are not at target search depth.
        # Go down search branches one after the other, and return the best possible branch at this level.     
        if maximizing_player == True:
            for move in legal_moves:                
                score, _ = self.minimax(game.forecast_move(move), depth-1, maximizing_player = False)
                # If this branch ensures a win, no need to search further. Else,, remember the best move.
                if score == float("inf"):
                    return score, move
                if score > highest_score_so_far:
                    highest_score_so_far, best_move_so_far = score, move
            return highest_score_so_far, best_move_so_far
        else:
            for move in legal_moves:                
                score, _ = self.minimax(game.forecast_move(move), depth-1, maximizing_player=True)
                # If this branch ensures a sure win, no need to search further. Else, remember the best move.
                if score == float("-inf"):
                    return score, move
                if score < lowest_score_so_far:
                    lowest_score_so_far, best_move_so_far = score, move
            return lowest_score_so_far, best_move_so_far

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # If there are no legal moves left to play, we lose.
        # The maximizing (minimizing) player returns the lowest (highest) possible score.
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            if maximizing_player == True:
                return float("-inf"), (-1, -1)
            else:
                return float("inf"), (-1, -1)

				
        lowest_score_so_far, highest_score_so_far = float("inf"), float("-inf")
        best_move_so_far = (-1, -1)
        if depth == 1:
            if maximizing_player == True:
                for move in legal_moves:                    
                    score = self.score(game.forecast_move(move), self)
                    # If this is a score better than beta, no need to search further. Otherwise, remember the best move.
                    if score >= beta:
                        return score, move
                    if score > highest_score_so_far:
                        highest_score_so_far, best_move_so_far = score, move
                return highest_score_so_far, best_move_so_far
            else:
                for move in legal_moves:                   
                    score = self.score(game.forecast_move(move), self)
                    # If this is a score worse than alpha, no need to search further. Otherwise, remember the best move.
                    if score <= alpha:
                        return score, move
                    if score < lowest_score_so_far:
                        lowest_score_so_far, best_move_so_far = score, move
                return lowest_score_so_far, best_move_so_far


        if maximizing_player == True:
            for move in legal_moves:                
                score, _ = self.alphabeta(game.forecast_move(move), depth-1, alpha, beta, maximizing_player = False)
                # If this branch yields a score better than beta, no need to search further.
                if score >= beta:
                    return score, move
                # Else, remember the best move and update alpha.
                if score > highest_score_so_far:
                    highest_score_so_far, best_move_so_far = score, move
                alpha = max(alpha, highest_score_so_far)
            return highest_score_so_far, best_move_so_far
        else:
            for move in legal_moves:
                # Evaluate this move in depth.
                score, _ = self.alphabeta(game.forecast_move(move), depth-1, alpha, beta, maximizing_player=True)
                # If this branch yields a score worse than alpha, no need to search further.
                if score <= alpha:
                    return score, move
                # Else, remember the best move and update beta.
                if score < lowest_score_so_far:
                    lowest_score_so_far, best_move_so_far = score, move
                beta = min(beta, lowest_score_so_far)
            return lowest_score_so_far, best_move_so_far
