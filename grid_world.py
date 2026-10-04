from enum import IntEnum
import random

class Action(IntEnum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3

ACTION_DELTAS = {
    Action.RIGHT: (1, 0),
    Action.UP: (0, 1),
    Action.LEFT: (-1, 0),
    Action.DOWN: (0, -1),
}

class GridWorld():
    def __init__(self, start_pos=None, goal_pos=(10, 10), mines={(3, 3)}, x_bound=10, y_bound=10, seed=1):
        self.goal_pos: tuple[int, int] = goal_pos
        self.mines: set[tuple[int, int]] = mines
        self.x_bound = x_bound
        self.y_bound = y_bound

        self.rng = random.Random(seed)
        self.start_positions = [
            (x, y)
            for x in range(self.x_bound + 1)
            for y in range(self.y_bound + 1)
            if (x, y) != self.goal_pos
            and (x, y) not in self.mines
        ]

        self.reset(start_pos)

    def reset(self, start=None) -> tuple[int, int]:
        if start is None:
            start = self.rng.choice(self.start_positions)
        elif start not in self.start_positions:
            raise ValueError(f"Invalid starting position: {start}")

        self.pos = start
        return self.pos

    def step_forward(self, action: int):
        direction: Action = Action(action)
        delta_x, delta_y = ACTION_DELTAS[direction]
        new_pos: tuple[int, int] = (self.pos[0] + delta_x, self.pos[1] + delta_y)
        if (not self.in_bounds(new_pos)):
            return self.pos, -0.1, False

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

    def at_goal(self) -> bool:
        return self.pos == self.goal_pos

    def in_bounds(self, pos: tuple[int, int]):
        x, y = pos
        return x >= 0 and x <= self.x_bound and y >= 0 and y <= self.y_bound

    