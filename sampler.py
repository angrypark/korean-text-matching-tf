import numpy as np
from abc import *

class BaseSampler(ABC):
    def __init__(self, config):
        pass

    def fit(self, sentences):
        pass

    def state_dict(self):
        pass

    def load_state_dict(self):
        pass

    @abstractmethod
    def sample(self, queries, replies, num_samples=9):
        pass


class RandomSampler(BaseSampler):
    def __init__(self, config):
        self.batch_size = config.batch_size

    def fit(self, sentences):
        self.candidates = sentences

    def sample(self, queries, replies, num_samples=9, add_echo=True):
        sampled_queries = list()
        sampled_replies = list()
        labels = list()
        for query, reply in zip(queries, replies):
            sampled_queries += [query]*(num_samples+1)
            if add_echo:
                sampled_replies.append(reply)
                sampled_replies.append(query)
                sampled_replies += np.random.choice(self.candidates, size=num_samples-1, replace=False).tolist()
                labels += [1] + [0]*num_samples
            else:
                sampled_replies.append(reply)
                sampled_replies += np.random.choice(self.candidates, size=num_samples, replace=False).tolist()
                labels += [1] + [0] * num_samples
        return sampled_queries, sampled_replies, labels

