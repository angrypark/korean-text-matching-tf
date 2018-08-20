import tensorflow as tf
import numpy as np
import argparse
from datetime import datetime

from data_loader import DataGenerator

from trainer import MatchingModelTrainer
from preprocessor import DynamicPreprocessor
from utils.dirs import create_dirs
from utils.logger import SummaryWriter
from utils.config import load_config, save_config
from models.base import get_model
from utils.utils import JamoProcessor

now = datetime.now()

# Parameters
# ==================================================

# Task specification
args = argparse.ArgumentParser()
args.add_argument("--mode", type=str, default="train", choices=["train", "debug", "infer"])
args.add_argument("--name", type=str, default="test")
args.add_argument("--config", type=str, default="")

# Data loading and saving parameters
args.add_argument("--train_dir", type=str, default="/media/scatter/scatterdisk/reply_matching_model/debug/")
args.add_argument("--val_dir", type=str, default="/media/scatter/scatterdisk/reply_matching_model/debug/sol.small.txt")
args.add_argument("--pretrained_embed_dir", type=str, default="/media/scatter/scatterdisk/pretrained_embedding/fasttext.sent_piece_50K.256D")
args.add_argument("--checkpoint_dir", type=str, default="/media/scatter/scatterdisk/reply_matching_model/runs/") 

# Model specification
args.add_argument("--model", type=str, default="DualEncoderLSTM")
args.add_argument("--sent_piece_model", type=str, default="/media/scatter/scatterdisk/tokenizer/sent_piece.50K.model")
args.add_argument("--normalizer", type=str, default="DummyNormalizer")
args.add_argument("--tokenizer", type=str, default="DummyTokenizer")
args.add_argument("--vocab_size", type=int, default=100000)
args.add_argument("--vocab_list", type=str, default="/media/scatter/scatterdisk/pretrained_embedding/vocab_list.sent_piece_50K.txt")

# Model hyperparameters
args.add_argument("--embed_dim", type=int, default=256)
args.add_argument("--learning_rate", type=float, default=1e-2)
args.add_argument("--min_length", type=int, default=1)
args.add_argument("--max_length", type=int, default=20)
args.add_argument("--lstm_dim", type=int, default=512)

# Training parameters
args.add_argument("--batch_size", type=int, default=256)
args.add_argument("--num_epochs", type=int, default=20)
args.add_argument("--evaluate_every", type=int, default=50000)
args.add_argument("--save_every", type=int, default=50000)
args.add_argument("--max_to_keep", type=int, default=5)
args.add_argument("--shuffle", type=bool, default=True)

# Sampling parameters
args.add_argument("--negative_sampling", type=str, default="random", choices=["random", "hard", "weighted"])
args.add_argument("--num_negative_samples", type=int, default=9)

# Misc parameters
args.add_argument("--gpu", type=str, default="a")

def main():
    config = args.parse_args()

    # Load pre-defined config if possible
    if config.config:
        config = load_config(config.config)

    config_str = " | ".join(["{}={}".format(attr.upper(), value) for attr, value in vars(config).items()])
    print(config_str)

    # create the experiments dirs
    config = create_dirs(config)
    print(config.checkpoint_dir)
    # create tensorflow session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))

    # build preprocessor
    preprocessor = DynamicPreprocessor(config)

    # load data, preprocess and generate data
    data = DataGenerator(preprocessor, config)

    # create tensorboard summary writer
    summary_writer = SummaryWriter(sess, config)

    # create trainer and pass all the previous components to it
    trainer = MatchingModelTrainer(sess, preprocessor, data, config, summary_writer)

    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()
