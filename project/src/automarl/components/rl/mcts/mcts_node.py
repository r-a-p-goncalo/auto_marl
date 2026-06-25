

class MCTSNode:
    def __init__(self, state, parent=None, prior=0.0, player_to_move=None):
        self.state = state
        self.parent = parent
        self.prior = prior
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.player_to_move = player_to_move

    @property
    def q_value(self):
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count