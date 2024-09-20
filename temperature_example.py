from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Union
import gymnasium as gym
import numpy as np
import random
from stable_baselines3.common.env_checker import check_env

STEPS = 60
GymObs = Union[Tuple, Dict, np.ndarray, int]

class CustomEnv(gym.Env):

    def __init__(self):
        super(CustomEnv, self).__init__()

        self.observation_space = gym.spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)
        self.current_temp = 36
        self.current_step = 0

    def reset(self, seed: int = None) -> Tuple[GymObs, Dict]:
        super().reset(seed=seed)
        self.current_step = 0
        obs = self.observation_space.sample()
        info = {}
        return obs, info

    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        temp_variation = 1
        reward = 0

        if action == 0:  # Aumentar temperatura
            self.current_temp += temp_variation
        elif action == 1:  # Disminuir temperatura
            self.current_temp -= temp_variation

        if 36 <= self.current_temp <= 38:
            reward = 1 
        else:
            reward = -1

        # Actualiza la observación para reflejar el estado actual
        obs = np.array([self.current_temp], dtype=np.float32)

        # Incrementa el contador de pasos y verifica si el episodio debe terminar
        self.current_step += 1
        done = self.current_step >= STEPS

        truncated = False 
        info = {}

        print(str(self.current_step) + ". Temperatura: " + str(obs[0]) + " Recompensa: " + str(reward)) 

        return obs, reward, done, truncated, info

env = CustomEnv()

check_env(env)

obs, info = env.reset()
print("Primera observación:", obs)

for _ in range(STEPS):
    obs, reward, done, truncated, info = env.step(env.action_space.sample())
    if(done):
        break
print("Nueva observación:", obs, "Recompensa:", reward, "Episodio terminado:", done)
