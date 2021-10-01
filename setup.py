from setuptools import setup

setup(
    name='natsbench-gym',
    version='0.1',
    packages=['natsbench_gym', 'natsbench_gym.envs'],
    url='',
    license='',
    author='cedrickulbach',
    author_email='kulbach@fzi.de',
    description='A openAI gym environment for natsbench',
    install_requires=[
        'gym',
        'numpy',
        'nas-bench-201',
        'torch'
    ]

)
