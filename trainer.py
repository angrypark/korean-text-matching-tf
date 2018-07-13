from tqdm import tqdm
import numpy as np
import tensorflow as tf


class BaseTrainer:
    def __init__(self, sess, model, data, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.data = data
        self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(self.init)

    def train(self):
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs+1, 1):
            self.train_epoch()
            self.sess.run(self.model.increment_cur_epoch_tensor)

    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError


class ExampleTrainer(BaseTrainer):
    def __init__(self, sess, model, data, config, logger):
        super(ExampleTrainer, self).__init__(sess, model, data, config, logger)

    def train_epoch(self):
        loop = tqdm(range(self.config.num_iter_per_epoch))
        losses = list()
        scores = list()
        
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
        batch_queries, batch_replies, batch_labels, \
        batch_queries_lengths, batch_replies_lengths = next(self.data.next_batch(self.batch_size))
        
        feed_dict = {self.model.input_queries: batch_queries,
                     self.model.input_replies: batch_replies,
                     self.model.input_labels: batch_labels,
                     self.model.queries_lengths: batch_queries_lengths,
                     self.model.replies_lengths: batch_replies_lengths,
                     self.model.is_training: True}

        _, loss, accuracy = self.sess.run([self.model.train_step, self.model.loss, self.model.accuracy],
                                       feed_dict=feed_dict)
        return loss, accuracy

    def val_step(self):
        val_queries, val_replies, val_labels, \
        val_queries_lengths, val_replies_lengths = self.data.load_val_data()
        
        feed_dict = {self.model.input_queries: val_queries,
                     self.model.input_replies: val_replies,
                     self.model.input_labels: val_labels,
                     self.model.queries_lengths: val_queries_lengths,
                     self.model.replies_lengths: val_replies_lengths,
                     self.model.is_training: False}

        loss, accuracy = self.sess.run([self.model.loss, self.model.accuracy], feed_dict=feed_dict)
        return loss, accuracy
