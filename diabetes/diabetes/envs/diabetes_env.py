#import libraries
import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np

import scipy
from scipy import stats

from typing import Optional

import time
import sys

# Diabetes Env
class DiabetesEnv(gym.Env):

    def __init__(self, age=None, bmi=None, glucose=None):
        """
        Define the initial starting patient

        Action Space: [low_insulin, high_insulin, metformin, glimepiride, None] --> [0, 1, 2, 3, 4]
        """
        self.action_space = spaces.Discrete(5)

        self.STEP_LIMIT = 365 # 30 days
        self.sleep = 0

        self.w1 = 0.1
        self.w2 = 0.2
        self.w3 = [-7, -12, -4.5, -2.5, 0.4]
        self.w4 = 1/0.99
        self.normalization_factor = 3.0

        self.meds = 0
        self.meds_list = []
        self.score = 0

        self.meds_map = {0: 'low_insulin',
                         1: 'high_insulin',
                         2: 'metformin',
                         3: 'glimepiride',
                         4: 'None'}

        self.age_init = age
        self.bmi_init = bmi
        self.glucose_init = glucose

    def reset(self):
        """
        Reset the game
        """
        if self.age_init == None:
            self.age = stats.norm(60, 10).rvs()
        else:
            self.age = self.age_init
        if self.bmi_init == None:
            self.bmi = stats.norm(30, 5).rvs()
        else:
            self.bmi = self.bmi_init
        if self.glucose_init == None:
            self.glucose = stats.norm(100, 10).rvs()
        else:
            self.glucose = self.glucose_init

        self.steps = 0

        self.obs = np.array([self.age, self.bmi, self.glucose])

        return self.obs

    def step(self, action):

        self.meds = action
        self.meds_list.append(self.meds_map[action])
        self.update_game_state()

        reward = self.breach_norm()

        #
        done = self.game_over()

        obs = np.array([self.age, self.bmi, self.glucose])
        #
        info = {'score': self.score}
        self.steps += 1
        time.sleep(self.sleep)
        #
        return (obs, reward, done, info)

    def breach_norm(self):
        if 140 > self.glucose > 80:
            self.score += 1
            reward = +1
        elif self.glucose < 60:
            reward = -4
        elif self.glucose > 300:
            reward = -3
        elif self.glucose > 450:
            reward = -4
        else:
            reward = -1
        return reward

    def calc_next_glucose(self):
        glucose_det = self.w1*self.age + self.w2*self.bmi + self.w3[self.meds] * self.normalization_factor +  self.w4*self.w3[self.meds]/(self.age * self.bmi)
        self.glucose = stats.norm(self.glucose + 0.2*glucose_det, 5).rvs()
        self.glucose = np.max(self.glucose, 40)
        self.glucose = np.min(self.glucose, 500)
        return self.glucose

    def update_game_state(self):
        self.calc_next_glucose()

        self.age = self.age
        self.bmi = self.bmi
        self.glucose = self.glucose

    def game_over(self):
        if self.steps >= self.STEP_LIMIT:
            return True
        else:
            return False

    def render(self):
        sys.stdout.write(f'\rMeds Sequence are: {self.meds_list}')
        sys.stdout.flush()

    def close(self):
        pass
