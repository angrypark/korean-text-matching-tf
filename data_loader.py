import numpy as np
import tensorflow as tf
import os

class DataGenerator:
    def __init__(self, preprocessor, config):
        self.config = config
        self.preprocessor = preprocessor
        test_queries, test_replies, test_labels = load_test_data()
        data = {"test_queries": test_queries, "test_replies": test_replies, "test_labels": test_labels}
        self.data = data
        train_size = 0
        for fname in os.listdir(config.train_dir):
            with open(os.path.join(config.train_dir, fname), "r") as f:
                length = sum([1 for line in f])
                train_size += length
        self.train_size = train_size
        with open(config.val_dir, "r") as f:
            self.val_size = sum([1 for line in f])
        self.shuffle = config.shuffle
            
    def get_train_iterator(self, batch_size):
        while True:
            for fname in sorted(os.listdir(self.config.train_dir)):
                file_dir = os.path.join(self.config.train_dir, fname)

                # get number of lines of a single file
                with open(file_dir, "r") as f:
                    file_length = sum([1 for line in f])

                # get data and preprocess them
                with open(file_dir, "r") as f:
                    train_data = [line.strip() for line in f]
                    if self.shuffle:
                        np.random.shuffle(train_data)
                        
                    train_queries, train_replies = split_data(train_data)

                    # preprocess
                    train_queries, train_queries_lengths = zip(*[self.preprocessor.preprocess(query)
                                                                 for query in train_queries])
                    train_replies, train_replies_lengths = zip(*[self.preprocessor.preprocess(reply)
                                                                 for reply in train_replies])

                num_batches_per_file = (file_length-1)//batch_size + 1
                for batch_num in range(num_batches_per_file):
                    start = batch_num*batch_size
                    end = min((batch_num+1)*batch_size, file_length)
                    yield train_queries[start:end], train_replies[start:end], \
                          train_queries_lengths[start:end], train_replies_lengths[start:end]

    def get_val_iterator(self, batch_size):
        with open(self.config.val_dir, "r") as f:
            val_data = [line.strip() for line in f]
            val_size = len(val_data)
        val_queries, val_replies = split_data(val_data)

        # preprocess
        val_queries, val_queries_lengths = zip(*[self.preprocessor.preprocess(query)
                                                     for query in val_queries])
        val_replies, val_replies_lengths = zip(*[self.preprocessor.preprocess(reply)
                                                     for reply in val_replies])
        num_batches = (val_size-1)//batch_size + 1
        for batch_num in range(num_batches):
            start = batch_num*batch_size
            end = min((batch_num+1)*batch_size, val_size)
            yield val_queries[start:end], val_replies[start:end], \
                  val_queries_lengths[start:end], val_replies_lengths[start:end]
            
            
def split_data(data):
    _, queries, replies = zip(*[line.split('\t') for line in data])
    return queries, replies

def load_test_data():
    base_dir = "/home/angrypark/reply_matching_model/data/"
    with open(os.path.join(base_dir, "test_queries.txt"), "r") as f:
        test_queries = [line.strip() for line in f]
    with open(os.path.join(base_dir, "test_replies.txt"), "r") as f:
        test_replies = [line.strip().split("\t") for line in f]
    with open(os.path.join(base_dir, "test_labels.txt"), "r") as f:
        test_labels = [[int(y) for y in line.strip().split("\t")] for line in f]
    return test_queries, test_replies, test_labels
