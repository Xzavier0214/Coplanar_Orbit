import gym
import numpy as np


class NoneSpace(gym.spaces.Space):
    def __init__(self):
        super(NoneSpace, self).__init__(None, None)

    def sample(self):
        return None

    def seed(self, seed=None):
        pass

    def contains(self, x):
        return x is None


class Union(gym.spaces.Space):
    def __init__(self, spaces):
        assert spaces is not None
        self.spaces = spaces
        for space in spaces:
            assert isinstance(space, gym.spaces.Space)
        super(Union, self).__init__(None, None)

    def sample(self):
        return self.spaces[0].sample()

    def seed(self, seed=None):
        [space.seed(seed) for space in self.spaces]

    def contains(self, x):
        return any(space.contains(x) for space in self.spaces)


if __name__ == "__main__":
    union = Union((NoneSpace(), gym.spaces.Box(
        low=np.array([1]), high=np.array([3]))))

    print(union.sample())
    print(union.contains(None))
    print(union.contains(np.array([2])))
    print(union.contains(np.array([4])))
