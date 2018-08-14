import tensorflow as tf
import numpy as np
import fastText
from jamo import h2j, j2hcj

from models.base import BaseModel

def get_embeddings(idx2word, config):
    embedding = np.random.uniform(-1.0, 1.0, [config.vocab_size, config.embed_dim])
    if config.pretrained_embed_dir:
        num_oov = 0
        print("Loading pre-trained embedding from {}...".format(config.pretrained_embed_dir))
        ft = fastText.load_model(config.pretrained_embed_dir)
        for i, vocab in enumerate(idx2word):
            try:
                embedding[i, :] = ft.get_word_vector(j2hcj(h2j(vocab)))
            except:
                num_oov += 1
        print("Embedding loaded, number of OOV : {} / {}".format(num_oov, len(idx2word)))
    else:
        print("No pre-trained embedding found, initialize with random distribution")
    return embedding

def pairwise_distances(queries_embedding, replies_embedding, squared=False):
    """Compute the 2D matrix of distances between queries and replies
    Args:
        queries_embedding : tensor of shape (batch_size, embed_dim)
        replies_embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = tf.matmul(queries_embedding, tf.transpose(replies_embedding))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = tf.diag_part(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)
    return distances

def make_negative_mask(distance_matrix, method="random", num_negative_samples=2):
    length = distance_matrix.shape[0]
    mask = np.zeros([length, length])
    if method == "random":
        for i in range(length):
            indices = np.random.choice([j for j in range(length) if j != i], size=num_negative_samples, replace=False)
            mask[i, indices] = -1
            mask[i, i] = 0
    elif method == "hard":
        argsort = np.argsort(distance_matrix)
        for i, indices in enumerate(argsort.tolist()):
            indices.remove(i)
            mask[i, indices[:num_negative_samples]] = -1
            mask[i, i] = 0
    elif method == "weighted":
        argsort = np.argsort(embedding, axis=1)
        interval = length//num_negative_samples
        for i, indices in enumerate(argsort.tolist()):
            indices.remove(i)
            mask[i, indices[::interval]] = -1
            mask[i, i] = 0
    return mask

class DualEncoderLSTM(BaseModel):
    def __init__(self, preprocessor, config):
        super(DualEncoderLSTM, self).__init__(preprocessor, config)
        self.build_model()

    def build_model(self):
        # Placeholders for input, output
        self.input_queries = tf.placeholder(tf.int32, [None, self.config.max_length], name="input_queries")
        self.input_replies = tf.placeholder(tf.int32, [None, self.config.max_length], name="input_replies")

        self.queries_lengths = tf.placeholder(tf.int32, [None], name="query_lengths")
        self.replies_lengths = tf.placeholder(tf.int32, [None], name="reply_lengths")

        self.is_training = tf.placeholder(tf.bool)
        batch_length = self.input_queries.shape[0]

        # Define optimizer
        self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate)

        # Embedding layer
        # embeddings = tf.Variable(tf.random_uniform([self.config.vocab_size, self.config.embed_dim], -1.0, 1.0), name="embeddings")
        embeddings = tf.Variable(get_embeddings(self.preprocessor.vectorizer.idx2word, self.config), name="embeddings")
        
        queries_embedded = tf.nn.embedding_lookup(embeddings, self.input_queries, name="queries_embedded")
        replies_embedded = tf.nn.embedding_lookup(embeddings, self.input_replies, name="replies_embedded")
        queries_embedded, replies_embedded = tf.cast(queries_embedded, tf.float32), tf.cast(replies_embedded, tf.float32)

        # Build LSTM layer
        with tf.variable_scope("lstm") as vs:
            lstm_cell = tf.nn.rnn_cell.LSTMCell(self.config.lstm_dim,
                                                forget_bias=2.0,
                                                use_peepholes=True,
                                                state_is_tuple=True)
            lstm_outputs, lstm_states = tf.nn.dynamic_rnn(
                cell=lstm_cell,
                inputs=tf.concat([queries_embedded, replies_embedded], 0),
                sequence_length=tf.concat([self.queries_lengths, self.replies_lengths], 0),
                dtype=tf.float32,
            )
            encoding_queries, encoding_replies = tf.split(lstm_states.h, 2, 0)
            
        # Predict a response
        with tf.variable_scope("prediction") as vs:
            M = tf.get_variable("M",
                                shape=[self.config.lstm_dim, self.config.lstm_dim],
                                initializer=tf.truncated_normal_initializer())

            generated_replies = tf.matmul(encoding_queries, M)
            generated_replies = tf.expand_dims(generated_replies, 2)
            encoding_replies = tf.expand_dims(encoding_replies, 2)

        with tf.variable_scope("negative_sampling") as vs:
            distances = pairwise_distances(generated_replies, encoding_replies)
            with tf.Session().as_default():
                distance_matrix = distances.eval()
                mask = make_negative_mask(distance_matrix)
            mask = tf.reshape(tf.constant(mask), [-1])

        with tf.variable_scope("loss"):
            positive_logits = tf.matmul(generated_replies, encoding_replies, True)
            positive_logits = tf.squeeze(positive_logits, [2])
            negative_logits = tf.gather(tf.reshape(distances, [-1], tf.where(mask), 1))
            num_positives = tf.shape(positive_logits)[0]
            num_negatives = tf.shape(negative_logits)[0]
            self.logits = tf.concat([positive_logits, negative_logits], 0)

        # Apply sigmoid to convert logits to probabilities
        self.probs = tf.sigmoid(positive_logits)

        # Calculate mean cross-entropy loss
        with tf.variable_scope("loss"):
            self.labels = tf.concat([tf.ones([num_positives, 1]), tf.zeros([num_negatives, 1])], 0)
            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=tf.to_float(self.labels))
            self.loss = tf.reduce_mean(losses)
            self.train_step = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)

        # Calculate accuracy
        with tf.name_scope("score"):
            self.predictions = tf.cast(tf.argmax(self.logits, 1), tf.int32)
            correct_predictions = tf.equal(self.predictions, self.labels)
            self.score = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def val(self, sess, feed_dict=None):
        result = dict()
        result["loss"] = sess.run(self.loss, feed_dict=feed_dict)
        result["score"] = sess.run(self.score, feed_dict=feed_dict)
        return result

    def infer(self, sess, feed_dict=None):
        return sess.run(self.probs, feed_dict=feed_dict)