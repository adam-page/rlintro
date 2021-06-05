from abc import ABC

import matplotlib.pyplot as plt
import numpy as np


class Bandit:
    """A simple k-armed bandit."""

    def __init__(
        self,
        arms: int = 10,
        qs: np.ndarray = None,
        walk: float = None,
    ):
        """Create the simple k-armed bandit.

        Args:
            arms (int, optional): Number of arms. Defaults to 10.
            qs (np.ndarray, optional): Initial q values-random if not set. Must be equal in length to arms. Defaults to None.
            walk (float, optional): Standard deviation for walk. No walking behavior if None. Defaults to None.

        Raises:
            IndexError: Raised when qs is specified but its length != arms.
        """
        if qs is None:
            self._qs: np.ndarray = np.random.normal(size=(arms, 1))
        elif len(qs) == arms:
            self._qs = qs
        else:
            raise IndexError
        self._walk = walk

    def step(self, arm: int) -> int:
        """Run a step by pulling one of the arms and getting a reward.

        Args:
            arm (int): Which arm to pull.

        Raises:
            IndexError: Raised when the arm value is greater than the number of arms for this bandit.

        Returns:
            int: The reward for this step.
        """
        if arm > len(self._qs):
            raise IndexError
        reward = np.random.normal(loc=self._qs[arm])

        if self._walk is not None:
            self._qs += np.random.normal(scale=self._walk, size=self._qs.shape)

        return reward


class BanditAgent(ABC):
    """Base class for all bandit agents."""

    def action(self, prev_reward: int = None) -> int:
        pass


class BanditExperiment:
    """Handles a bandit experiment with an agent and reward averaging."""

    def __init__(
        self, bandit: Bandit, agent: BanditAgent, steps: int, reward_over: int = None
    ):
        """Create the experiment.

        Args:
            bandit (Bandit): The bandit for this experiment.
            agent (BanditAgent): The agent for this experiment.
            steps (int): The total number of steps.
            reward_over (int, optional): The number of steps (from the end) to average reward over. Defaults to None.
        """
        self._bandit = bandit
        self._agent = agent
        self._steps = steps
        self._r_over = self._steps - reward_over if reward_over else 0
        self._avg_r = 0.0
        self._nr = 0
        self._current_step = 0

    def run(self) -> float:
        """Run the experiment over all steps.

        Returns:
            float: The average reward over the last reward_over steps or all steps if reward_over is not set.
        """
        reward = None
        for i in range(self._steps):
            r = self._bandit.step(self._agent.action(reward))
            if i >= self._r_over:
                self._avg_r = (r + (self._nr * self._avg_r)) / float(self._nr + 1)
                self._nr += 1
        return self._avg_r


class RandomLeverAgent(BanditAgent):
    """A simple BanditAgent which just randomly chooses always."""

    def action(self, prev_reward: int) -> int:
        """Get the next action based on previous reward.

        Args:
            prev_reward (int, optional): The previous reward. Defaults to None.

        Returns:
            int: The action selected by the agent.
        """
        return np.random.randint(10)


experiment_parameters = [
    float(1 / 128),
    float(1 / 64),
    float(1 / 32),
    float(1 / 16),
    float(1 / 8),
    float(1 / 4),
    float(1 / 2),
    1,
    2,
    4,
]


if __name__ == "__main__":
    print(experiment_parameters)
    exp = BanditExperiment(Bandit(), RandomLeverAgent(), 2000, reward_over=1000)
    print(exp.run())
