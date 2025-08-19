"""
Heuristic agent for Gomoku.
"""
import random


class HeuristicAgent:
    """
    An agent that uses basic pattern recognition and heuristics for move selection.
    
    Priority order:
    1. Immediate win - if we can win in one move, take it
    2. Immediate block - if opponent can win in one move, block it  
    3. Pattern scoring - score moves based on creating/blocking patterns
    4. Center proximity - prefer center moves as tie-breaker
    """
    
    def __init__(self, seed=None):
        """
        Initialize the heuristic agent.
        
        Args:
            seed (int, optional): Random seed for tie-breaking reproducibility
        """
        self.rng = random.Random(seed)
        
    def select_action(self, game):
        """
        Select the best move using heuristic evaluation.
        
        Args:
            game: Game instance with current board state
            
        Returns:
            tuple: (row, col) coordinates of selected move, or None if no legal moves
        """
        legal_moves = game.board.get_legal_moves()
        
        if not legal_moves:
            return None
            
        # Check for immediate win
        win_move = self._find_immediate_win(game, game.current_player)
        if win_move:
            return win_move
            
        # Check for immediate block (opponent win)
        opponent = -game.current_player
        block_move = self._find_immediate_win(game, opponent)
        if block_move:
            return block_move
            
        # Score all moves and pick the best
        best_moves = []
        best_score = float('-inf')
        
        for row, col in legal_moves:
            score = self._score_move(game, row, col, game.current_player)
            
            if score > best_score:
                best_score = score
                best_moves = [(row, col)]
            elif score == best_score:
                best_moves.append((row, col))
                
        # Use center proximity for tie-breaking
        if len(best_moves) > 1:
            best_moves.sort(key=lambda move: self._center_distance(move))
            
        return self.rng.choice(best_moves[:min(3, len(best_moves))])  # Pick from top 3 center-closest
        
    def _find_immediate_win(self, game, player):
        """
        Find a move that creates an immediate win for the specified player.
        Uses the board's proper win detection logic which handles overlines correctly.
        
        Args:
            game: Current game state
            player: Player to check for wins (1 or -1)
            
        Returns:
            tuple or None: (row, col) of winning move, or None if none exists
        """
        legal_moves = game.board.get_legal_moves()
        
        for row, col in legal_moves:
            # Temporarily place the stone
            game.board.state[row, col] = player
            
            # Check if this creates a valid win (board handles overline detection)
            winner = game.board.check_winner(row, col)
            
            # Remove the temporary stone
            game.board.state[row, col] = 0
            
            if winner == player:
                return (row, col)
                
        return None
        
    def _score_move(self, game, row, col, player):
        """
        Score a move based on pattern analysis.
        
        Args:
            game: Current game state
            row, col: Move coordinates
            player: Player making the move
            
        Returns:
            float: Score for this move (higher is better)
        """
        # First check if this move creates an overline (illegal in Standard Gomoku)
        game.board.state[row, col] = player
        winner = game.board.check_winner(row, col)
        game.board.state[row, col] = 0
        
        # If this move would create an overline, heavily penalize it
        if winner is None and self._would_create_overline(game.board, row, col, player):
            return -10000  # Heavily penalize overlines
            
        score = 0.0
        opponent = -player
        
        # Analyze patterns in all 4 directions
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            # Score patterns created by our move
            our_pattern = self._analyze_line_pattern(game.board, row, col, player, dr, dc)
            score += self._pattern_score(our_pattern, is_our_pattern=True)
            
            # Score patterns we disrupt for opponent
            opp_pattern = self._analyze_line_pattern(game.board, row, col, opponent, dr, dc)
            score += self._pattern_score(opp_pattern, is_our_pattern=False) * 0.8  # Slightly less weight
            
        return score
        
    def _would_create_overline(self, board, row, col, player):
        """
        Check if placing a stone at (row, col) would create an overline (6+ in a row).
        
        Args:
            board: Board instance
            row, col: Position to check
            player: Player whose stone we're placing
            
        Returns:
            bool: True if this would create an overline
        """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        
        for dr, dc in directions:
            # Count contiguous stones in negative direction
            count_neg = 0
            r, c = row - dr, col - dc
            while (0 <= r < board.size and 0 <= c < board.size and 
                   board.state[r, c] == player):
                count_neg += 1
                r, c = r - dr, c - dc
                
            # Count contiguous stones in positive direction  
            count_pos = 0
            r, c = row + dr, col + dc
            while (0 <= r < board.size and 0 <= c < board.size and
                   board.state[r, c] == player):
                count_pos += 1
                r, c = r + dr, c + dc
                
            # Total length including the new stone
            total_length = count_neg + 1 + count_pos
            
            # If we have 6 or more stones in a row, it's an overline
            if total_length >= 6:
                return True
                
        return False
        
    def _analyze_line_pattern(self, board, row, col, player, dr, dc):
        """
        Analyze the pattern that would be created by placing a stone at (row, col).
        
        Args:
            board: Board instance
            row, col: Position to analyze
            player: Player whose pattern we're analyzing
            dr, dc: Direction to analyze
            
        Returns:
            dict: Pattern analysis with 'length', 'open_ends', 'blocked_ends'
        """
        # Count contiguous stones in negative direction
        count_neg = 0
        r, c = row - dr, col - dc
        while (0 <= r < board.size and 0 <= c < board.size and 
               board.state[r, c] == player):
            count_neg += 1
            r, c = r - dr, c - dc
            
        # Count contiguous stones in positive direction  
        count_pos = 0
        r, c = row + dr, col + dc
        while (0 <= r < board.size and 0 <= c < board.size and
               board.state[r, c] == player):
            count_pos += 1
            r, c = r + dr, c + dc
            
        # Total length including the new stone
        total_length = count_neg + 1 + count_pos
        
        # Check if ends are open (empty) or blocked
        open_ends = 0
        
        # Check negative end
        neg_end_r, neg_end_c = row - dr * count_neg - dr, col - dc * count_neg - dc
        if (0 <= neg_end_r < board.size and 0 <= neg_end_c < board.size and
            board.state[neg_end_r, neg_end_c] == 0):
            open_ends += 1
            
        # Check positive end
        pos_end_r, pos_end_c = row + dr * count_pos + dr, col + dc * count_pos + dc
        if (0 <= pos_end_r < board.size and 0 <= pos_end_c < board.size and
            board.state[pos_end_r, pos_end_c] == 0):
            open_ends += 1
            
        return {
            'length': total_length,
            'open_ends': open_ends,
            'blocked_ends': 2 - open_ends
        }
        
    def _pattern_score(self, pattern, is_our_pattern):
        """
        Score a pattern based on its characteristics.
        
        Args:
            pattern: Pattern dict from _analyze_line_pattern
            is_our_pattern: True if this is our pattern, False if opponent's
            
        Returns:
            float: Score for this pattern
        """
        length = pattern['length']
        open_ends = pattern['open_ends']
        
        # Base scoring - longer sequences are exponentially more valuable
        if length >= 5:
            # This shouldn't happen in legal moves, but just in case
            base_score = 10000
        elif length == 4:
            base_score = 1000 if open_ends >= 1 else 50  # 4-open vs 4-blocked
        elif length == 3:
            base_score = 100 if open_ends == 2 else (20 if open_ends == 1 else 2)  # 3-open vs 3-semi vs 3-blocked
        elif length == 2:
            base_score = 10 if open_ends == 2 else (3 if open_ends == 1 else 1)  # 2-open vs 2-semi vs 2-blocked
        else:
            base_score = 1  # Single stone
            
        # Multiply by sign depending on whose pattern it is
        return base_score if is_our_pattern else -base_score * 0.9  # Slightly less weight to blocking
        
    def _center_distance(self, move):
        """
        Calculate distance from center for tie-breaking.
        
        Args:
            move: (row, col) tuple
            
        Returns:
            float: Distance from center (7, 7)
        """
        row, col = move
        center = 7  # Center of 15x15 board
        return ((row - center) ** 2 + (col - center) ** 2) ** 0.5