import os
import sys
import tensorflow as tf
import numpy as np
import argparse
from datetime import datetime
from tqdm import tqdm

sys.path.append("/home/angrypark/korean-text-matching-tf/")

from data_loader import DataGenerator
from trainer import MatchingModelTrainer
from preprocessor import DynamicPreprocessor
from utils.dirs import create_dirs
from utils.logger import SummaryWriter
from utils.config import load_config, save_config
from models.base import get_model
from utils.utils import JamoProcessor


PROCESS_ID = 2
NAME = "delstm_1024_nsrandom4_lr1e-3"
TOKENIZER = "SentencePieceTokenizer"

if __name__ == "__main__":
    base_dir = "/media/scatter/scatterdisk/reply_matching_model/runs/{}/".format(NAME)
    config_dir = base_dir + "config.json"
    best_model_dir = base_dir + "best_loss/best_loss.ckpt"
    model_config = load_config(config_dir)
    preprocessor = DynamicPreprocessor(model_config)
    preprocessor.build_preprocessor()

    infer_config = load_config(config_dir)
    setattr(infer_config, "tokenizer", TOKENIZER)
    setattr(infer_config, "soynlp_scores", "/media/scatter/scatterdisk/tokenizer/soynlp_scores.sol.100M.txt")
    infer_preprocessor = DynamicPreprocessor(infer_config)
    infer_preprocessor.build_preprocessor()

    model_config.add_echo = False

    graph = tf.Graph()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    with graph.as_default():
        Model = get_model(model_config.model)
        data = DataGenerator(preprocessor, model_config)
        infer_model = Model(data, model_config)
        infer_sess = tf.Session(config=tf_config, graph=graph)
        infer_sess.run(tf.global_variables_initializer())
        infer_sess.run(tf.local_variables_initializer())

        infer_model.load(infer_sess, model_dir=best_model_dir)

    with open("../reply_matching_model/data/reply_set_new.txt", "r") as f:
        reply_set = [line.strip() for line in f if line]
    indexed_reply_set, reply_set_lengths = zip(*[infer_preprocessor.preprocess(r) for r in reply_set])

    def get_result(query, reply):
        preprocessed_query, query_length = infer_preprocessor.preprocess(query)
        preprocessed_reply, reply_length = infer_preprocessor.preprocess(reply)

        input_queries, query_lengths = [preprocessed_query]*(len(indexed_reply_set)+1), [query_length]*(len(indexed_reply_set)+1)
        input_replies, reply_lengths = list(indexed_reply_set)+[preprocessed_reply], list(reply_set_lengths)+[reply_length]

        feed_dict = {infer_model.input_queries: input_queries, 
                     infer_model.input_replies: input_replies, 
                     infer_model.queries_lengths: query_lengths, 
                     infer_model.replies_lengths: reply_lengths, 
                     infer_model.dropout_keep_prob: 1}
        probs = infer_model.infer(infer_sess, feed_dict=feed_dict)

        real_score = probs[-1][0]
        result = sorted([(reply, "{:.4f}".format(score[0])) 
                         for reply, score in zip(reply_set, probs[:-1])], 
                        key=lambda x: x[1], reverse=True)[:10]

        return "{}\t{}\t{:.4f}\t{}".format(query, 
                                           reply, 
                                           real_score, 
                                           "\t".join(["\t".join(item) for item in result]))

    data_dir = "/media/scatter/scatterdisk/reply_matching_model/normalized/sol.raw_{}.txt"
    target_dir = "/media/scatter/scatterdisk/reply_matching_model/crowdworks/sol.{}.txt"
    with open(data_dir.format(PROCESS_ID+1), "r") as f1, open(target_dir.format(PROCESS_ID+1), "a") as f2:
        for i in tqdm(range(100000)):
            line = f1.readline()
            query, reply = line.strip().split("\t")[1:3]
            f2.write(get_result(query, reply) + "\n")
            