import os
import json
import tensorflow as tf

def load_config(path):
    with open(os.path.join(path, "config.json"), "r", encoding="utf-8") as f:
        config_dict = json.loads(json.loads(f.readline()))

    config = tf.contrib.training.HParams(**config_dict)
    return config

def save_config(path, config):
    with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config.to_json(), f)
