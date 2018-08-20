import tensorflow as tf
import numpy as np
from gensim.models import FastText

from utils.utils import JamoProcessor
from models.base import BaseModel

def get_embeddings(idx2word, config):
    embedding = np.random.uniform(-1/16, 1/16, [config.vocab_size, config.embed_dim])
    if config.pretrained_embed_dir:
        processor = JamoProcessor()
        ft = FastText.load(config.pretrained_embed_dir)
        num_oov = 0
        for i, vocab in enumerate(idx2word):
            try:
                embedding[i, :] = ft.wv[processor.word_to_jamo(vocab)]
            except:
                num_oov += 1
    else:
        print("No pre-trained embedding found, initialize with random distribution")
    return embedding

def make_negative_mask(distances, method="random", num_negative_samples=2, batch_size=256):
    if method == "random":
        mask = np.zeros([batch_size, batch_size])
        #for i in range(batch_size):
        #    indices = np.random.choice([j for j in range(batch_size) if j != i], size=num_negative_samples, replace=False)
        #    mask[i, indices] = True
        #    mask[i, i] = False
        for i in range(batch_size):
            if i < batch_size - 1:
                mask[i, i+1] = 1.
            else:
                mask[i, 0] = 1.
        mask = tf.convert_to_tensor(mask)
    elif method == "hard":
        top_k = tf.contrib.framework.sort(tf.expand_dims(tf.nn.top_k(-distances, k=num_negative_samples+1).indices, -1), axis=1)
        row_indices = tf.expand_dims(tf.transpose(tf.reshape(tf.tile(tf.range(0, batch_size, 1), [num_negative_samples+1]), [num_negative_samples+1, batch_size])), -1)
        mask_indices = tf.to_int64(tf.squeeze(tf.reshape(tf.concat([row_indices, top_k], 2), [(num_negative_samples+1)*batch_size,1,2])))
        mask_sparse = tf.SparseTensor(mask_indices, [1]*((num_negative_samples+1)*batch_size), [batch_size,batch_size])
        mask = tf.sparse_tensor_to_dense(mask_sparse)
        drop_positive = tf.to_int32(tf.subtract(tf.ones([batch_size, batch_size]), tf.eye(batch_size)))
        mask = tf.multiply(mask, drop_positive)
    elif method == "weighted":
        weight = tf.map_fn(lambda x: get_distance_weight(x, batch_size), tf.to_float(distances))
        mask = weight
#         mask = tf.to_int32(tf.contrib.framework.sort(tf.expand_dims(tf.multinomial(weight, num_negative_samples+1), -1), axis=1))
#         weighted_samples_indices = tf.to_int32(tf.contrib.framework.sort(tf.expand_dims(tf.multinomial(weight, num_negative_samples+1), -1), axis=1))
#         row_indices = tf.expand_dims(tf.transpose(tf.reshape(tf.tile(tf.range(0, batch_size, 1), [num_negative_samples+1]), [num_negative_samples+1, batch_size])), -1)
#         mask_indices = tf.to_int64(tf.squeeze(tf.reshape(tf.concat([row_indices, weighted_samples_indices], 2), [(num_negative_samples+1)*batch_size,1,2])))
#         mask_sparse = tf.SparseTensor(mask_indices, [1]*((num_negative_samples+1)*batch_size), [batch_size,batch_size])
#         mask = tf.sparse_tensor_to_dense(mask_sparse)
#         drop_positive = tf.to_int32(tf.subtract(tf.ones([batch_size, batch_size]), tf.eye(batch_size)))
#         mask = tf.multiply(mask, drop_positive)
    return mask


class DualEncoderLSTM(BaseModel):
    def __init__(self, preprocessor, config):
        super(DualEncoderLSTM, self).__init__(preprocessor, config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        with tf.variable_scope("inputs"):
            # Placeholders for input, output
            self.input_queries = tf.placeholder(tf.int32, [None, self.config.max_length])
            self.input_replies = tf.placeholder(tf.int32, [None, self.config.max_length])

            self.queries_lengths = tf.placeholder(tf.int32, [None])
            self.replies_lengths = tf.placeholder(tf.int32, [None])

        cur_batch_length = tf.shape(self.input_queries)[0]

        # Define optimizer
        self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate)

        # Embedding layer
        # embeddings = tf.Variable(tf.random_uniform([self.config.vocab_size, self.config.embed_dim], -1.0, 1.0), name="embeddings")
        with tf.variable_scope("embedding"):
            embeddings = tf.Variable(get_embeddings(self.preprocessor.vectorizer.idx2word, self.config), trainable=True, name="embeddings")

            queries_embedded = tf.nn.embedding_lookup(embeddings, self.input_queries, name="queries_embedded")
            replies_embedded = tf.nn.embedding_lookup(embeddings, self.input_replies, name="replies_embedded")
            queries_embedded, replies_embedded = tf.cast(queries_embedded, tf.float32), tf.cast(replies_embedded, tf.float32)

        # Build LSTM layer
        with tf.variable_scope("lstm") as vs:
            send_lstm_cell = tf.nn.rnn_cell.LSTMCell(self.config.lstm_dim,
                                                     forget_bias=2.0,
                                                     use_peepholes=True,
                                                     state_is_tuple=True,
                                                     name='send')
            recv_lstm_cell = tf.nn.rnn_cell.LSTMCell(self.config.lstm_dim,
                                                     forget_bias=2.0,
                                                     use_peepholes=True,
                                                     state_is_tuple=True,
                                                     name='recv')

            _, encoding_queries = tf.nn.dynamic_rnn(
                cell=send_lstm_cell,
                inputs=queries_embedded,
                sequence_length=self.queries_lengths,
                dtype=tf.float32,
            )
            _, encoding_replies = tf.nn.dynamic_rnn(
                cell=recv_lstm_cell,
                inputs=replies_embedded,
                sequence_length=self.replies_lengths,
                dtype=tf.float32,
            )
            encoding_queries = encoding_queries.h
            encoding_replies = encoding_replies.h

        # Predict a response
        with tf.variable_scope("prediction") as vs:
            M = tf.get_variable("M",
                                shape=[self.config.lstm_dim, self.config.lstm_dim],
                                initializer=tf.truncated_normal_initializer())
            encoding_queries = tf.matmul(encoding_queries, M)

        with tf.variable_scope("negative_sampling") as vs:
            # distances = pairwise_distances(encoding_queries, encoding_replies, squared=False)
            distances = tf.matmul(encoding_queries, tf.transpose(encoding_replies))
            positive_mask = tf.reshape(tf.eye(cur_batch_length), [-1])
            negative_mask = make_negative_mask(distances,
                                               method=self.config.negative_sampling,
                                               num_negative_samples=self.config.num_negative_samples,
                                               batch_size=self.config.batch_size)
            
            # slice negative mask for when current batch size is smaller than predefined batch size
            negative_mask = tf.slice(negative_mask, [0,0], [cur_batch_length, cur_batch_length])
            negative_mask = tf.reshape(negative_mask, [-1])

        with tf.variable_scope("logits"):
            positive_logits = tf.gather(tf.reshape(distances, [-1]), tf.where(positive_mask), 1)
            self.positive_probs = tf.sigmoid(positive_logits)
            negative_logits = tf.gather(tf.reshape(distances, [-1]), tf.where(negative_mask), 1)
            num_positives = tf.shape(positive_logits)[0]
            num_negatives = tf.shape(negative_logits)[0]
            self.logits = tf.concat([positive_logits, negative_logits], 0)

        # Calculate mean cross-entropy loss
        with tf.variable_scope("loss"):
            self.labels = tf.to_float(tf.concat([tf.ones([num_positives, 1]), tf.zeros([num_negatives, 1])], 0))
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.labels)
            self.losses = losses # DEBUG
            self.loss = tf.reduce_mean(losses)
            self.train_step = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)

        # Calculate accuracy
        with tf.name_scope("score"):
            # Apply sigmoid to convert logits to probabilities
            self.probs = tf.sigmoid(self.logits)
            self.predictions = tf.cast(self.probs > 0.5, dtype=tf.int32)
            correct_predictions = tf.equal(self.predictions, tf.to_int32(self.labels))
            self.score = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def val(self, sess, feed_dict=None):
        loss = sess.run(self.loss, feed_dict=feed_dict)
        score = sess.run(self.score, feed_dict=feed_dict)
        probs = sess.run(self.probs, feed_dict=feed_dict)
        return loss, score, probs

    def infer(self, sess, feed_dict=None):
        return sess.run(self.positive_probs, feed_dict=feed_dict)
