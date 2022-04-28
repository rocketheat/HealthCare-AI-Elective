from gym.envs.registration import register

register(id='diabetes-v0', entry_point='diabetes.envs:DiabetesEnv',)
