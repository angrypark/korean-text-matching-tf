from tqdm import tqdm
import numpy as np
import tensorflow as tf

from models.base import get_model
from utils.logger import setup_logger
from utils.config import save_config


class BaseTrainer:
    def __init__(self, sess, preprocessor, data, config, summary_writer):
        self.sess = sess
        self.preprocessor = preprocessor
        self.data = data
        self.config = config
        self.summary_writer = summary_writer

        self.logger = setup_logger()
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    def train(self):
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
            self.train_epoch(models, sess)
            self.sess.run(self.model.increment_cur_epoch_tensor)

    def train_epoch(self, model, sess):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self, model, sess):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError

    def build_graph(self, name="train"):
        graph = tf.Graph()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        # train graph
        with graph.as_default():
            self.logger.info("Building {} graph...".format(name))
            model = get_model(self.config.model_name)
            sess = tf.Session(config=tf_config, graph=graph)
            if (self.config.checkpoint_path) and (name == "train"):
                self.logger.info('Loading checkpoint from {}'.format(
                    config.checkpoint_dir))
                model.load(train_sess, config.checkpoint_dir)
            else:
                sess.run(self.init)
        return model, sess


class MatchingModelTrainer(BaseTrainer):
    def __init__(self, sess, preprocessor, data, config, summary_writer):
        super(Trainer, self).__init__(sess, preprocessor, data, config, summary_writer)
        # get size of data
        self.train_size = data.train_size
        self.val_size = data.val_size
        self.batch_size = config.batch_size

        # initialize global step, epoch
        self.num_steps_per_epoch = (self.train_size - 1) // self.batch_size + 1
        self.cur_epoch = 0
        self.global_step = 0

        # for summary and logger
        self.summary_dict = dict()
        self.train_summary = "Epoch : {:2d} | Train loss : {:.4f} | Train accuracy : {:.4f} "
        self.val_summary = "| Val loss : {:.4f} | Val accuracy : {:.4f} "


    def train_epoch(self, model, sess):
        """Not used because data size is too big"""
        self.cur_epoch += 1
        loop = tqdm(range(self.num_steps_per_epoch))
        losses = list()
        scores = list()

        for step in loop:
            loss, score = self.train_step(model, sess)
            losses.append(loss)
            scores.append(score)
        train_loss = np.mean(losses)
        train_score = np.mean(scores)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'train_loss': train_loss,
            'train_accuracy': train_score
        }

    def train_step(self, model, sess):
        batch_queries, batch_replies, batch_labels, \
        batch_queries_lengths, batch_replies_lengths = next(self.data.next_batch(self.batch_size))

        feed_dict = {model.input_queries: batch_queries,
                     model.input_replies: batch_replies,
                     model.input_labels: batch_labels,
                     model.queries_lengths: batch_queries_lengths,
                     model.replies_lengths: batch_replies_lengths,
                     model.is_training: True}

        _, loss, score = sess.run([model.train_step, model.loss, model.accuracy],
                                     feed_dict=feed_dict)
        sess.run(model.increment_global_step_tensor)
        self.global_step += 1
        return loss, score

    def val(self, model, sess, global_step):
        # load latest checkpoint
        model.load(sess)

        losses = list()
        scores = list()
        num_instances = 0

        num_batches_per_epoch = (self.data.val_size - 1) // self.batch_size + 1
        val_queries, val_replies, val_queries_lengths, val_replies_lengths = self.data.load_val_data()
        for batch_num in range(num_batches_per_epoch):
            start_idx = batch_num * self.batch_size
            end_idx = min((batch_num + 1) * self.batch_size, self.val_size)
            feed_dict = {model.input_queries: val_queries[start_idx:end_idx],
                         model.input_replies: val_replies[start_idx:end_idx],
                         model.queries_lengths: val_queries_lengths[start_idx:end_idx],
                         model.replies_lengths: val_replies_lengths[start_idx:end_idx],
                         model.is_training: False}
            result = model.val(sess, feed_dict=feed_dict)
            losses.append(result["loss"] * batch_size)
            scores.append(result["score"] * batch_size)
            num_instances += batch_size

        val_loss = np.sum(losses) / num_instances
        val_score = np.sum(scores) / num_instances

        best_loss = getattr(self.config, "best_loss", 1e+5)
        if val_loss < best_loss:
            setattr(config, "best_loss", val_loss)
            model.save(sess,
                       os.path.join(config.checkpoint_dir, "best_loss", "best_loss" + ".ckpt"))
            self.logger.warn("[Step {}] Saved for best loss : {:.5f}".format(global_step, best_loss))
            # if best model, infer for the test set
            save_config(config.checkpoint_dir, config)
        return val_loss, val_score

    def infer(self, model, sess, global_step):
        # load best model
        model.load(sess, tf.train.latest_checkpoint(self.config.checkpoint_dir))
        return None

    def train(self):
        # build train, val, test, infer graph
        train_model, train_sess = self.build_graph(name="train")
        val_model, val_sess = self.build_graph(name="val")
        infer_model, infer_sess = self.build_graph(name="infer")
        summaries_dict = dict()

        for cur_epoch in range(train_model.cur_epoch_tensor.eval(train_sess), self.config.num_epochs + 1, 1):
            train_sess.run(train_model.increment_cur_epoch_tensor)
            losses = list()
            scores = list()
            for step in range(1, self.num_steps_per_epoch+1):
                train_score, train_loss = self.train_step(train_model, train_sess)
                losses.append(train_loss)
                scores.append(train_score)

                if self.global_step % self.config.save_every == 0:
                    train_model.save(train_sess, os.path.join(config.checkpoint_dir, "model.ckpt"))

                if self.global_step % self.config.evaluate_every == 0:
                    val_loss, val_score = self.val(val_model, val_sess)
                    self.logger.warn(self.train_summary.format(self.cur_epoch, train_loss, train_score) \
                                     + self.val_summary.format(val_loss, val_score))

            # val step
            val_loss, val_score = self.val(val_model, val_sess, global_step)
            summaries_dict['val_loss'], summaries_dict['val_accuracy'] = val_loss, val_score
            self.logger.warn(self.train_summary.format(self.cur_epoch, train_loss, train_score) \
                                 + self.val_summary.format(val_loss, val_score))

            self.summary_writer.summarize(cur_it, summaries_dict=summaries_dict)
