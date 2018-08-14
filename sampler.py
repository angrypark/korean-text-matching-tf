import numpy as np
from abc import *

class BaseSampler(ABC):
    def __init__(self, config):
        pass

    @abstractmethod
    def sample(self, queries, replies, num_samples=9):
        pass


class RandomSampler(BaseSampler):
    def __init__(self, config):
        self.batch_size = config.batch_size

    def sample(self, queries, replies, num_samples=9, add_echo=True):
        """
        get queries, replies and returns index of sampled queries, index of samples replies and labels
        :param queries: queries batch
        :param replies: replies batch
        :param num_samples: if 9, get 9 negative samples per 1 positive sample
        :param add_echo: if True, add input query as negative sample of replies
        :return:
        """
        sampled_queries_indices = list()
        sampled_replies_indices = list()
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

