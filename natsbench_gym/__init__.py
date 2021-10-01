from gym.envs.registration import register


register(id='Natsbench-v0',
         entry_point='natsbench_gym.envs:NatsbenchEnv')
