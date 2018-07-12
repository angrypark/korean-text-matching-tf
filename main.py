import tensorflow as tf
import argparse
from datetime import datetime

from data_loader import DataGenerator

from trainer import Trainer
from preprocessor import Preprocessor
from utils.dirs import create_dirs
from utils.logger import Logger

now = datetime.now()

# Parameters
# ==================================================

# Task specification
args = argparse.ArgumentParser()
args.add_argument("--mode", type=str, default="train", choices=["train", "debug", "infer"])
args.add_argument("--config", type=str, default="")

# Data loading and saving parameters
args.add_argument("--train_dir", type=str, default="data/train.txt")
args.add_argument("--val_dir", type=str, default="data/test.txt")
args.add_argument("--pretrained_embed_dir", type=str, default="/home/shuuki4/sandbox_project/fasttext/results/ft_256.bin")

# Model specification
args.add_argument("--model", type=str, default="DualEncoderLSTM")
args.add_argument("--normalizer", type=str, default="BaseNormalizer")
args.add_argument("--tokenizer", type=str, default="BaseTokenizer")
args.add_argument("--vocab_size", type=int, default=200000)

# Model hyperparameters
args.add_argument("--embed_dim", type=int, default=256)
args.add_argument("--learning_rate", type=float, default=1e-1)
args.add_argument("--min_length", type=int, default=1)
args.add_argument("--max_length", type=int, default=20)
args.add_argument("--lstm_dim", type=int, default=512)

# Training parameters
args.add_argument("--batch_size", type=int, default=256)
args.add_argument("--num_epochs", type=int, default=200)
args.add_argument("--evaluate_every", type=int, default=1)
args.add_argument("--checkpoint_every", type=int, default=1)
args.add_argument("--max_to_keep", type=int, default=10)
args.add_argument("--shuffle", type=bool, default=True)

# Misc parameters
args.add_argument("--gpu", type=str, default="a")

config = args.parse_args()
config_str = " | ".join(["{}={}".format(attr.upper(), value) for attr, value in config.__dict__.items()])
print(config_str)


def main():
    # create the experiments dirs
    create_dirs(config)

    # create tensorflow session
    sess = tf.Session()

    # build preprocessor
    preprocessor = Preprocessor(config)

    # load data, preprocess and generate data
    data = DataGenerator(preprocessor, config)

    # create an instance of the model you want
    model = TextCNN.TextCNN(preprocessor, config)

    # create tensorboard logger
    logger = Logger(sess, config)

    # create trainer and pass all the previous components to it
    trainer = Trainer(sess, model, data, config, logger)

    # load model if exists
    model.load(sess)

    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()
