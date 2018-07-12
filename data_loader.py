import numpy as np

class DataGenerator:
    def __init__(self, preprocessor, config):
        self.config = config
        train_set, val_set = load_data(config.train_dir, config.val_dir, small=(config.mode=="small"))
        self.train_size = len(train_set)
        if config.shuffle:
            np.random.shuffle(train_set)
            
        # split data into lines and lables
        train_queries, train_replies = split_data(train_set)
        val_queries, val_replies = split_data(val_set)
        
        # build preprocessor
        self.preprocessor = preprocessor
        self.preprocessor.build_preprocessor(train_lines)

        # preprocess line and make it to a list of word indices
        train_data = np.array([self.preprocessor.preprocess(sentence) for sentence in train_lines])
        val_data = np.array([self.preprocessor.preprocess(sentence) for sentence in val_lines])
        
        # merge train data and val data
        data = dict()
        data['train_data'], data['train_labels'] = train_data, train_labels
        data['val_data'], data['val_labels'] = val_data, val_labels
        self.data = data

    def next_batch(self, batch_size):
        num_batches_per_epoch = (self.train_size-1)//batch_size + 1
        for epoch in range(self.config.num_epochs):
            for batch_num in range(num_batches_per_epoch):
                start_idx = batch_num*batch_size
                end_idx = min((batch_num+1)*batch_size, self.train_size)
                yield self.data['train_data'][start_idx:end_idx], self.data['train_labels'][start_idx:end_idx]

    def load_val_data(self):
        return self.data['val_data'], self.data['val_labels']


def load_data(train_dir, val_dir, small=False):
    with open(train_dir, 'r') as f:
        train_data = [line.strip() for line in f.readlines()]
        if small:
            train_data = train_data[:500]
    with open(val_dir, 'r') as f:
        val_data = [line.strip() for line in f.readlines()]
        if small:
            val_data = val_data[:50]
    return train_data, val_data

def split_data(data):
    queries, replies = zip(*[line.split('\t') for line in data])
    return queries, replies
