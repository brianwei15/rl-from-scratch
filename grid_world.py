from enum import IntEnum

class Action(IntEnum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    RIGHT = 3

ACTION_DELTAS = {
    Action.RIGHT: (1, 0),
    Action.UP: (0, 1),
    Action.LEFT: (-1, 0),
    Action.DOWN: (0, -1),
}

class GridWorld():
    def __init__(self):
        self.pos: tuple[int, int] = (0, 0) # (x, y)
        self.goal_pos: tuple[int, int] = [5, 5]
        self.mines: set[tuple[int, int]] = {(3, 3)}
        self.x_bound = 5
        self.y_bound = 5

    def reset(self) -> tuple[int, int]:
        self.pos = (0, 0)
        return self.pos

    def step_forward(self, action: int):
        direction: Action = Action[action]
        delta_x, delta_y = ACTION_DELTAS[direction]
        new_pos: tuple[int, int] = (self.pos[0] + delta_x, self.pos[1] + delta_y)
        if (not self.in_bounds(new_pos)):
            return self.pos, 0, False

        # move to valid new pos and update reward and termination status accordingly
        self.pos = new_pos
        if (self.pos == self.goal_pos):
            reward = 1
            terminate = True
        elif (self.pos in self.mines):
            reward = -1
            terminate = True
        else:
            reward = 0
            terminate = False

        return self.pos, reward, terminate

    def in_bounds(self, pos: tuple[int, int]):
        x, y = pos
        return x >= 0 and x <= self.x_bound and y >= 0 and y <= self.y_bound

    