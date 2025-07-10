import numpy as np
from numba import njit

@njit
def move(position, facing_dir, left_speed, right_speed):
    lv = min(max(abs(left_speed) + abs(right_speed), 75), 150) / 7.0
    facing_dir = (facing_dir + (right_speed - left_speed) / 10.0 + np.pi) % (2 * np.pi) - np.pi

    position[0] = max(0.0, min(1600.0, position[0] + lv * np.cos(facing_dir)))
    position[1] = max(0.0, min(1200.0, position[1] + lv * np.sin(facing_dir)))

    return position, facing_dir


class Worm:
    """
    A very light actor: position, heading, ‘sees_food’ flag.
    The flag is **set externally** by the environment; we do not
    pass a stimulus into the numba kernel any more.
    """
    def __init__(self, position, facing_dir=0.0):
        self.position     = np.asarray(position, dtype=np.float32)
        self.facing_dir   = float(facing_dir)
        self.sees_food    = False

    def kinematic_step(self, left_speed, right_speed):
        self.position, self.facing_dir = move(
            self.position, self.facing_dir, left_speed, right_speed
        )
