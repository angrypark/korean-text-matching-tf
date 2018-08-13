import tensorflow as tf
from utils.logger import setup_logger

class BaseModel:
    def __init__(self, preprocessor, config):
        self.preprocessor = preprocessor
        self.config = config
        self.init_global_step()
        self.init_cur_epoch()
        self.init_saver()
        self.logger = setup_logger()

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, sess, save_dir):
        self.saver.save(sess, save_dir, self.global_step_tensor)
        print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, sess):
        latest_checkpoint = tf.train.latest_checkpoint(self.config.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {}_{} ...\n".format(self.config.name, latest_checkpoint))
            self.saver.restore(sess, latest_checkpoint)
            print("Model loaded")
        else:
            self.logger.error("No checkpoint found in {}".format(self.config.checkpoint_dir))

    # just initialize a tensorflow variable to use it as epoch counter
    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    # just initialize a tensorflow variable to use it as global step counter
    def init_global_step(self):
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
            self.increment_global_step_tensor = tf.assign(self.global_step_tensor, self.global_step_tensor + 1)

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def build_model(self):
        raise NotImplementedError
