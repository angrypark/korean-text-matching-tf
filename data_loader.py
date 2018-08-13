import numpy as np

class DataGenerator:
    def __init__(self, preprocessor, config):
        self.config = config
        train_set, val_set = load_data(config.train_dir, config.val_dir, small=(config.mode=="debug"))
        self.train_size = len(train_set)
        if config.shuffle:
            np.random.shuffle(train_set)
            
        # split data into lines and lables
        train_queries, train_replies, train_labels = split_data(train_set)
        val_queries, val_replies, val_labels = split_data(val_set)
        test_queries, test_replies, test_labels = load_test_data()
        
        # build preprocessor
        self.preprocessor = preprocessor
        self.preprocessor.build_preprocessor(train_queries + train_replies)

        # preprocess line and make it to a list of word indices
        train_queries, train_queries_lengths = zip(*[self.preprocessor.preprocess(query) for query in train_queries])
        train_replies, train_replies_lengths = zip(*[self.preprocessor.preprocess(reply) for reply in train_replies])

        val_queries, val_queries_lengths = zip(*[self.preprocessor.preprocess(query) for query in val_queries])
        val_replies, val_replies_lengths = zip(*[self.preprocessor.preprocess(reply) for reply in val_replies])

        test_queries, test_queries_lengths = zip(*[self.preprocessor.preprocess(query) for query in test_queries])
        test_replies, test_replies_lengths = zip(*[self.preprocessor.preprocess(reply) for reply in test_replies])

        train_queries, train_replies = np.array(train_queries), np.array(train_replies)
        val_queries, val_replies = np.array(val_queries), np.array(val_replies)
        
        # merge train data and val data
        data = dict()
        data['train_queries'], data['train_replies'], data['train_labels'] = train_queries, train_replies, train_labels
        data['train_queries_lengths'], data['train_replies_lengths'] = train_queries_lengths, train_replies_lengths
        
        data['val_queries'], data['val_replies'], data['val_labels'] = val_queries, val_replies, val_labels
        data['val_queries_lengths'], data['val_replies_lengths'] = val_queries_lengths, val_replies_lengths


        self.data = data

    def next_batch(self, batch_size):
        num_batches_per_epoch = (self.train_size-1)//batch_size + 1
        for epoch in range(self.config.num_epochs):
            for batch_num in range(num_batches_per_epoch):
                start_idx = batch_num*batch_size
                end_idx = min((batch_num+1)*batch_size, self.train_size)
                yield self.data['train_queries'][start_idx:end_idx], self.data['train_replies'][start_idx:end_idx], \
                self.data['train_labels'][start_idx:end_idx], self.data['train_queries_lengths'][start_idx:end_idx], \
                self.data['train_replies_lengths'][start_idx:end_idx]

    def load_val_data(self):
        return self.data['val_queries'], self.data['val_replies'], self.data['val_labels'], self.data['val_queries_lengths'], self.data['val_replies_lengths']


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
    queries, replies, labels = zip(*[line.split('\t') for line in data])
    return queries, replies, labels

def load_test_data():
    base_dir = "/home/angrypark/reply_matching_model/data/"
    with open(os.path.join(base_dir, "test_queries.txt"), "r") as f:
        test_queries = [line.strip() for line in f]
    with open(os.path.join(base_dir, "test_replies.txt"), "r") as f:
        test_replies = [line.strip().split("\t") for line in f]
    with open(os.path.join(base_dir, "test_labels.txt"), "r") as f:
        test_labels = [[int(y) for y in line.strip().split("\t")] for line in f]
    return test_queries, test_replies, test_labels