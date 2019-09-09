import gym
from action import Action, NoneAction, ImpulseAction, LowThrustAction


class ActionSpace(gym.spaces.Space):
    def __init__(self):
        super(ActionSpace, self).__init__(None, None)

    def sample(self):
        return NoneAction()

    def contains(self, x):
        return isinstance(x, Action)


if __name__ == "__main__":
    action_space = ActionSpace()

    print(action_space.sample())
    print(action_space.contains(NoneAction()))
    print(action_space.contains(ImpulseAction(100, 0, None)))
    print(action_space.contains(LowThrustAction(100, 0, None)))
    print(action_space.contains([2, 3]))
