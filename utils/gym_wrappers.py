import gymnasium as gym
from collections import deque
import numpy as np


def space_stack(space: gym.Space, repeat: int):
    """
    Creates new Gym space that represents the original observation/action space
    repeated `repeat` times.
    """

    if isinstance(space, gym.spaces.Box):
        return gym.spaces.Box(
            low=np.repeat(space.low[None], repeat, axis=0),
            high=np.repeat(space.high[None], repeat, axis=0),
            dtype=space.dtype,
        )
    elif isinstance(space, gym.spaces.Discrete):
        return gym.spaces.MultiDiscrete([space.n] * repeat)
    elif isinstance(space, gym.spaces.Dict):
        return gym.spaces.Dict(
            {k: space_stack(v, repeat) for k, v in space.spaces.items()}
        )
    else:
        raise ValueError(f"Space {space} is not supported by Octo Gym wrappers.")


class TemporalEnsembleWrapper(gym.Wrapper):
    """
    Performs temporal ensembling from https://arxiv.org/abs/2304.13705
    At every timestep we execute an exponential weighted average of the last
    `pred_horizon` predictions for that timestep.
    """

    def __init__(self, env: gym.Env, pred_horizon: int, exp_weight: int = 0):
        super().__init__(env)
        self.pred_horizon = pred_horizon
        self.exp_weight = exp_weight

        self.act_history = deque(maxlen=self.pred_horizon)

        self.action_space = space_stack(self.env.action_space, self.pred_horizon)

    def step(self, actions):
        assert len(actions) >= self.pred_horizon

        self.act_history.append(actions[: self.pred_horizon])
        num_actions = len(self.act_history)

        # select the predicted action for the current step from the history of action chunk predictions
        curr_act_preds = np.stack(
            [
                pred_actions[i]
                for (i, pred_actions) in zip(
                    range(num_actions - 1, -1, -1), self.act_history
                )
            ]
        )

        # more recent predictions get exponentially *less* weight than older predictions
        weights = np.exp(-self.exp_weight * np.arange(num_actions))
        weights = weights / weights.sum()
        # compute the weighted average across all predictions for this timestep
        action = np.sum(weights[:, None] * curr_act_preds, axis=0)

        return self.env.step(action)

    def reset(self, **kwargs):
        self.act_history = deque(maxlen=self.pred_horizon)
        return self.env.reset(**kwargs)
