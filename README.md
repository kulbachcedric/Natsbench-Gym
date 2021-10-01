# Natsbench-gym 
A Gym Environment to apply RL Agents on the Nats Bench Dataset

## Installation
Download the natsbench benchmark file [NATS-sss-v1_0-50262.pickle.pbz2](https://drive.google.com/file/d/1IabIvzWeDdDAWICBzFtTCMXxYWPIOIOX/view?usp=sharing) 
```shell
cd data
gdown --id 1IabIvzWeDdDAWICBzFtTCMXxYWPIOIOX
```
Install the Gym Environment:
```shell
pip install nats-bench==1.1
pip install -e .
```
## Dummy Example:
```python
import gym
from nats_bench import create

dataset = 'cifar10'
api = create('./NATS-sss-v1_0-50262.pickle.pbz2', 'sss',fast_mode=False, verbose=False)

env = gym.make('Natsbench-v0',api=api, dataset=dataset)
for _ in range(1000):
    env.render()
    action = env.action_space.sample() # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
    if done:
        observation = env.reset()
env.close()
```
