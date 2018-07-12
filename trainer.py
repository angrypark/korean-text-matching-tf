from base.base_trainer import BaseTrainer
from tqdm import tqdm
import numpy as np
import tensorflow as tf


class ExampleTrainer(BaseTrainer):
    def __init__(self, sess, model, data, config, logger):
        super(ExampleTrainer, self).__init__(sess, model, data, config, logger)

    def train_epoch(self):
        loop = tqdm(range(self.config.num_iter_per_epoch))
        losses = []
        scores = []
        for _ in loop:
            loss, score = self.train_step()
            losses.append(loss)
            scores.append(score)
        loss = np.mean(losses)
        score = np.mean(scores)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'loss': loss,
            'score': score,
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):
        batch_x, batch_y = next(self.data.next_batch(self.config.batch_size))
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True}
        _, loss, score = self.sess.run([self.model.train_step, self.model.loss, self.model.score],
                                     feed_dict=feed_dict)
        return loss, score


class Trainer(BaseTrainer):
    def __init__(self, sess, model, data, config, logger):
        super(Trainer, self).__init__(sess, model, data, config, logger)
        self.train_size = data.train_size
        self.batch_size = config.batch_size
        self.dropout_keep_prob = config.dropout_keep_prob
        self.num_iter_per_epoch = (self.train_size - 1) // self.batch_size + 1
        self.cur_epoch = 0
        self.train_summary = "Epoch : {:2d} | Train loss : {:.4f} | Train accuracy : {:.4f} "
        self.val_summary = "| Val loss : {:.4f} | Val accuracy : {:.4f} "

    def train_epoch(self):
        self.cur_epoch += 1
        loop = tqdm(range(self.num_iter_per_epoch))
        losses = list()
        scores = list()

        for _ in loop:
            loss, score = self.train_step()
            losses.append(loss)
            scores.append(score)
        train_loss = np.mean(losses)
        train_score = np.mean(scores)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'train_loss': train_loss,
            'train_accuracy': train_score
        }

        if self.cur_epoch % self.config.evaluate_every == 0:
            val_loss, val_score = self.val_step()
            summaries_dict['val_loss'], summaries_dict['val_accuracy'] = val_loss, val_score
            print(self.train_summary.format(self.cur_epoch, train_loss, train_score) + \
                  self.val_summary.format(val_loss, val_score))
        else:
            print(self.train_summary.format(self.cur_epoch, train_loss, train_score))

        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):
        batch_x, batch_y = next(self.data.next_batch(self.batch_size))
        feed_dict = {self.model.input_x: batch_x,
                     self.model.input_y: batch_y,
                     self.model.is_training: True,
                     self.model.dropout_keep_prob: self.dropout_keep_prob}

        _, loss, score = self.sess.run([self.model.train_step, self.model.loss, self.model.score],
                                       feed_dict=feed_dict)
        return loss, score

    def val_step(self):
        val_data, val_labels = self.data.load_val_data()
        feed_dict = {self.model.input_x: val_data,
                     self.model.input_y: val_labels,
                     self.model.is_training: False,
                     self.model.dropout_keep_prob: 1}

        loss, score = self.sess.run([self.model.loss, self.model.score], feed_dict=feed_dict)
        return loss, score
