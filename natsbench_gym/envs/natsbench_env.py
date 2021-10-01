import gym
import numpy as np
from gym import spaces
from natsbench_gym.envs.nasbench_helper import AccuracyScorer
import warnings


class NatsbenchEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,api=None,dataset='cifar10',scorer=None):
        self.observation = np.array(5* [0,0,0,0,0,0,0,0])
        self.api = api
        self.dataset = dataset
        if scorer is None:
            self.scorer = AccuracyScorer(api=api, dataset=dataset)
        else:
            self.scorer = scorer
        if api is None:
            warnings.warn("Warning...........\n Randomized scores will be created.")
        self.observation_space = spaces.Box(0,7,shape=(40,),dtype=np.int16)
        self.action_space = spaces.Discrete(8)
        self.counter = 0
        self.done = False
        self.reward = 0.0
        self.actions = []
        self.possible_actions = ['8','16','24','32','40','48','56','64']
        self.info = {
            'scorer_history': [],
            'metrics_history' : [],
            'idx_history' : []
        }


    def step(self, action):
        """

        Parameters
        ----------
        action : [0,1,2,3,4,5,6,7]

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        self.actions.append(action)
        if len(self.actions) == 5:
            self.done = True

        self.observation[(len(self.actions)-1)*8+action] = 1

        if self.done:
            action_strings = [self.possible_actions[action] for action in self.actions]
            arch = ':'.join(action_strings)
            if self.api is None:
                idx = 10
                reward = np.random.random()
            else:
                idx = self.api.query_index_by_arch(arch)
                reward = self.scorer.score(idx)
                self.info['scorer_history'].append(reward)
                a = self.api.get_more_info(index=idx, dataset=self.dataset)
                b = self.api.get_cost_info(index=idx, dataset=self.dataset)
                a.update(b)
                self.info['metrics_history'].append(a)
                self.info['idx_history'].append(idx)

            return self.observation, reward, self.done, self.info
        else:
            reward = 0.0
            return self.observation, reward, self.done, self.info

    def reset(self):
        self.counter = 0
        self.done = False
        self.reward = 0.0
        self.actions = []
        self.observation_space = spaces.Box(0,7,shape=(40,),dtype=np.int16)
        self.observation = np.array(5* [0,0,0,0,0,0,0,0])
        return self.observation

    def render(self, mode='human', close=False):
        pass
