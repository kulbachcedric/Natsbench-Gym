import numpy as np

def get_metrics(api, index:int, dataset:str):
    res = []
    a = api.get_more_info(index=index, dataset=dataset)
    b = api.get_cost_info(index=index, dataset=dataset)
    a.update(b)
    res.append(a['flops'])
    res.append(a['params'])
    res.append(a['latency'])
    res.append(a['train-accuracy'])
    res.append(a['test-accuracy'])
    return np.array(res)

class AccuracyScorer():
    def __init__(self, api, dataset):
        self.api = api
        self.dataset = dataset

    def score(self,idx):
        X = get_metrics(api=self.api, index=idx, dataset=self.dataset)
        return X[4]
