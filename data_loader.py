import numpy as np
import tensorflow as tf
import os

class DataGenerator:
    def __init__(self, preprocessor, config):
        # get size of train and validataion set
        self.train_size = 298554955
        with open(config.val_dir, "r") as f:
            self.val_size = sum([1 for line in f])
            
        # data config
        self.train_dir = config.train_dir
        self.val_dir = config.val_dir
        self.max_length = config.max_length
        self.batch_size = config.batch_size
        self.shuffle = config.shuffle
        self.num_epochs = config.num_epochs
            
    def get_train_iterator(self, index_table):
        train_files = [os.path.join(self.train_dir, fname) 
                       for fname in sorted(os.listdir(self.train_dir)) 
                       if "validation" not in fname]
        
        train_set = tf.data.TextLineDataset(train_files)
        train_set = train_set.map(lambda line: parse_single_line(line, index_table, self.max_length),
                                  num_parallel_calls=8)
        train_set = train_set.shuffle(buffer_size=10000)
        train_set = train_set.batch(self.batch_size)
        train_set = train_set.repeat(self.num_epochs)
        
        train_iterator = train_set.make_initializable_iterator()
        return train_iterator
        
    def get_val_iterator(self, index_table):
        val_set = tf.data.TextLineDataset(self.val_dir)
        val_set = val_set.map(lambda line: parse_single_line(line, index_table, self.max_length),
                              num_parallel_calls=2)
        val_set = val_set.shuffle(buffer_size=1000)
        val_set = val_set.batch(self.batch_size)

        val_iterator = val_set.make_initializable_iterator()
        return val_iterator
            
    def load_test_data(self):
        base_dir = "/home/angrypark/reply_matching_model/data/"
        with open(os.path.join(base_dir, "test_queries.txt"), "r") as f:
            test_queries = [line.strip() for line in f]
        with open(os.path.join(base_dir, "test_replies.txt"), "r") as f:
            replies_set = [line.strip().split("\t") for line in f]
        with open(os.path.join(base_dir, "test_labels.txt"), "r") as f:
            test_labels = [[int(y) for y in line.strip().split("\t")] for line in f]

        test_queries, test_queries_lengths = zip(*[self.preprocessor.preprocess(query)
                                                         for query in test_queries])
        test_replies = list()
        test_replies_lengths = list()
        for replies in replies_set:
            r, l = zip(*[self.preprocessor.preprocess(reply) for reply in replies])
            test_replies.append(r)
            test_replies_lengths.append(l)
        return test_queries, test_replies, test_queries_lengths, test_replies_lengths, test_labels

def split_data(data):
    _, queries, replies = zip(*[line.split('\t') for line in data])
    return queries, replies

def parse_single_line(line, index_table, max_length):
    """get single line from train set, and returns after padding and indexing
    :param line: corpus id \t query \t reply
    """
    splited = tf.string_split([line], delimiter="\t")
    query = tf.concat([["<SOS>"], tf.string_split([splited.values[1]], delimiter=" ").values, ["<EOS>"]], axis=0)
    reply = tf.concat([["<SOS>"], tf.string_split([splited.values[2]], delimiter=" ").values, ["<EOS>"]], axis=0)
    
    paddings = tf.constant([[0, 0],[0, max_length]])
    padded_query = tf.slice(tf.pad([query], paddings, constant_values="<PAD>"), [0, 0], [-1, max_length])
    padded_reply = tf.slice(tf.pad([reply], paddings, constant_values="<PAD>"), [0, 0], [-1, max_length])
    
    indexed_query = tf.squeeze(index_table.lookup(padded_query))
    indexed_reply = tf.squeeze(index_table.lookup(padded_reply))
    
    return indexed_query, indexed_reply, tf.shape(query)[0], tf.shape(reply)[0]