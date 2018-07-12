import tensorflow as tf
import numpy as np
import fastText

from base.base_model import BaseModel
from utils.jamo_fasttext import JamoFastText

def get_embeddings(idx2word, config):
    embedding = np.random.uniform(-1.0, 1.0, [config.vocab_size, config.embed_dim])
    if config.pretrained_embed_dir:
        num_oov = 0
        print("Loading pre-trained embedding from {}...".format(config.pretrained_embed_dir))
        ft = fastText.load_model(config.pretrained_embed_dir)
        jft = JamoFastText()
        for i, vocab in enumerate(idx2word):
            try:
                embedding[i, :] = ft.get_word_vector(jft._convert_word_to_jamo(vocab))
            except:
                num_oov += 1
        print("Embedding loaded, number of OOV : {} / {}".format(num_oov, len(idx2word)))
    else:
        print("No pre-trained embedding found, initialize with random distribution")
    return embedding


class DualEncoderLSTM(BaseModel):
    def __init__(self, preprocessor, config):
        super(DualEncoderLSTM, self).__init__(config)
        self.config = config
        self.preprocessor = preprocessor
        self.build_model()
        self.init_saver()

    def build_model(self):
        # Placeholders for input, output
        self.input_queries = tf.placeholder(tf.int32, [None, self.config.max_length], name="input_queries")
        self.input_replies = tf.placeholder(tf.int32, [None, self.config.max_length], name="input_replies")

        self.query_lengths = tf.placeholder(tf.int32, [None], name="query_lengths")
        self.reply_lengths = tf.placeholder(tf.int32, [None], name="reply_lengths")
        self.input_labels = tf.placeholder(tf.int32, [None], name="labels")

        self.is_training = tf.placeholder(tf.bool)

        # Define Optimizer
        self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate)

        # Embedding layer
        embeddings = get_embeddings(self.preprocessor.vectorizer.idx2word)
        queries_embedded = tf.nn.embedding_lookup(embeddings, self.input_queries, name="queries_embedded")
        replies_embedded = tf.nn.embedding_lookup(embeddings, self.input_replies, name="replies_embedded")

        # Build LSTM layer
        with tf.variable_scope("lstm") as vs:
            lstm_cell = tf.nn.rnn_cell.LSTMCell(self.config.lstm_dim,
                                                forget_bias=2.0,
                                                use_peepholes=True,
                                                state_is_tuple=True)
            lstm_outputs, lstm_states = tf.nn.dynamic_rnn(
                cell=lstm_cell,
                inputs=tf.concat(0, [queries_embedded, replies_embedded]),
                sequence_length=tf.concat(0, [self.query_lengths, self.reply_lengths]),
                dtype=tf.float32)
            encoding_queries, encoding_replies = tf.split(0, 2, lstm_states.h)

        # Predict a response
        with tf.variable_scope("prediction") as vs:
            M = tf.get_variable("M",
                                shape=[self.config.lstm_dim, self.config.lstm_dim],
                                initializer=tf.truncated_normal_initializer())

            generated_replies = tf.matmul(encoding_queries, M)
            generated_replies = tf.expand_dims(generated_replies, 2)
            encoding_replies = tf.expand_dims(encoding_replies, 2)

            # Dot product between generated replies and actual replies
            logits = tf.matmul(generated_replies, encoding_replies, True)
            self.logits = tf.squeeze(logits, [2])

            # Apply sigmoid to convert logits to probabilities
            self.probs = tf.sigmoid(self.logits)

        # Calculate mean cross-entropy loss
        with tf.variable_scope("loss"):
            labels = tf.cast([self.input_labels], tf.int64)
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits, tf.to_float(labels))
            self.loss = tf.reduce_mean(losses)
            self.train_step = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)

        # Calculate accuracy
        with tf.name_scope("accuracy"):
            predictions = tf.argmax(self.logits, 1)
            correct_predictions = tf.equal(predictions, self.input_labels)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

