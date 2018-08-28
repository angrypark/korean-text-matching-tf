import tensorflow as tf
import numpy as np
from gensim.models import FastText

from models.base import BaseModel
from models.tcn_ops import *
from utils.utils import JamoProcessor

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
        print("Pre-trained embedding loaded. Number of OOV : {} / {}".format(num_oov, len(idx2word)))
    else:
        print("No pre-trained embedding found, initialize with random distribution")
    return embedding

def make_negative_mask(distances, method="random", num_negative_samples=2, batch_size=256):
    if method == "random":
        mask = np.zeros([batch_size, batch_size])
        for i in range(batch_size):
            indices = np.random.choice([j for j in range(batch_size) if j != i], size=num_negative_samples, replace=False)
            mask[i, indices] = True
            mask[i, i] = False
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

def temporal_padding(x, padding=(1, 1)):
    """Pads the middle dimension of a 3D tensor.
    # Arguments
        x: Tensor or variable.
        padding: Tuple of 2 integers, how many zeros to
            add at the start and end of dim 1.
    # Returns
        A padded 3D tensor.
    """
    assert len(padding) == 2
    pattern = [[0, 0], [padding[0], padding[1]], [0, 0]]
    return tf.pad(x, pattern)

def attentionBlock(x):
    """self attention block
    # Arguments
        x: Tensor of shape [N, L, Cin]
    """

    k_size = x.get_shape()[-1].value
    v_size = x.get_shape()[-1].value

    key = tf.layers.dense(x, units=k_size, activation=None, use_bias=False, kernel_initializer=tf.random_normal_initializer(0, 0.01)) # [N, L, k_size]
    #query = tf.layers.dense(x, units=k_size, activation=None, use_bias=False, kernel_initializer=tf.random_normal_initializer(0, 0.01)) # [N, L, k_size]
    value = tf.layers.dense(x, units=v_size, activation=None, use_bias=False, kernel_initializer=tf.random_normal_initializer(0, 0.01))
    
    logits = tf.matmul(key, key, transpose_b=True)
    logits = logits / np.sqrt(k_size)
    weights = tf.nn.softmax(logits, name="attention_weights") # N, L, ksize
    output = tf.matmul(weights, value)

    return output

@add_arg_scope
def weightNormConvolution1d(x, num_filters, dilation_rate, filter_size=3, stride=[1],
                            pad='VALID', init_scale=1., init=False, gated=False,
                            counters={}, name="query"):
    name = get_name('weightnorm_conv1d'+"_{}".format(name), counters)
    with tf.variable_scope(name):
        # currently this part is never used
        if init:
            print("initializing weight norm")
            # data based initialization of parameters
            V = tf.get_variable('V', [filter_size, int(x.get_shape()[-1]), num_filters],
                                tf.float32, tf.random_normal_initializer(0, 0.01),
                                trainable=True)
            V_norm = tf.nn.l2_normalize(V.initialized_value(), [0, 1])

            # pad x
            left_pad = dilation_rate * (filter_size - 1)
            x = temporal_padding(x, (left_pad, 0))
            x_init = tf.nn.convolution(x, V_norm, pad, stride, [dilation_rate])
            #x_init = tf.nn.conv2d(x, V_norm, [1]+stride+[1], pad)
            m_init, v_init = tf.nn.moments(x_init, [0, 1])
            scale_init = init_scale/tf.sqrt(v_init + 1e-8)
            g = tf.get_variable('g', dtype=tf.float32, initializer=scale_init,
                                trainable=True)
            b = tf.get_variable('b', dtype=tf.float32, initializer=-m_init*scale_init,
                                trainable=True)
            x_init = tf.reshape(scale_init, [1, 1, num_filters]) \
                                * (x_init - tf.reshape(m_init, [1, 1, num_filters]))
            # apply nonlinearity
            x_init = tf.nn.relu(x_init)
            return x_init

        else:
            # Gating mechanism (Dauphin 2016 LM with Gated Conv. Nets)
            if gated:
                num_filters = num_filters * 2

            # size of V is L, Cin, Cout
            V = tf.get_variable('V', [filter_size, int(x.get_shape()[-1]), num_filters],
                                tf.float32, initializer=None,
                                trainable=True)
            g = tf.get_variable('g', shape=[num_filters], dtype=tf.float32,
                                initializer=tf.constant_initializer(1.), trainable=True)
            b = tf.get_variable('b', shape=[num_filters], dtype=tf.float32,
                                initializer=None, trainable=True)

            # size of input x is N, L, Cin

            # use weight normalization (Salimans & Kingma, 2016)
            W = tf.reshape(g, [1, 1, num_filters]) * tf.nn.l2_normalize(V, [0, 1])

            # pad x for causal convolution
            left_pad = dilation_rate * (filter_size  - 1)
            x = temporal_padding(x, (left_pad, 0))

            # calculate convolutional layer output
            x = tf.nn.bias_add(tf.nn.convolution(x, W, pad, stride, [dilation_rate]), b)

            # GLU
            if gated:
                split0, split1 = tf.split(x, num_or_size_splits=2, axis=2)
                split1 = tf.sigmoid(split1)
                x = tf.multiply(split0, split1)
            # ReLU
            else:
                # apply nonlinearity
                x = tf.nn.relu(x)
            # print(x.get_shape())
            
            return x

def TemporalBlock(input_layer, out_channels, filter_size, stride, dilation_rate, counters,
                  dropout, init=False, atten=False, use_highway=False, gated=False, name="query"):

    keep_prob = 1.0 - dropout

    in_channels = input_layer.get_shape()[-1]
    name = get_name('temporal_block' + '_{}'.format(name), counters)
    with tf.variable_scope(name):

        # num_filters is the hidden units in TCN
        # which is the number of out channels
        conv1 = weightNormConvolution1d(input_layer, out_channels, dilation_rate,
                                        filter_size, [stride], counters=counters,
                                        init=init, gated=gated, name=name)
        # set noise shape for spatial dropout
        # refer to https://colab.research.google.com/drive/1la33lW7FQV1RicpfzyLq9H0SH1VSD4LE#scrollTo=TcFQu3F0y-fy
        # shape should be [N, 1, C]
        noise_shape = (tf.shape(conv1)[0], tf.constant(1), tf.shape(conv1)[2])
        dropout1 = tf.nn.dropout(conv1, keep_prob, noise_shape)
        if atten:
            dropout1 = attentionBlock(dropout1)

        conv2 = weightNormConvolution1d(dropout1, out_channels, dilation_rate, filter_size,
            [stride], counters=counters, init=init, gated=gated, name=name)
        dropout2 = tf.nn.dropout(conv2, keep_prob, noise_shape)
        if atten:
            dropout2 = attentionBlock(dropout2)

        # highway connetions or residual connection
        residual = None
        if use_highway:
            W_h = tf.get_variable('W_h', [1, int(input_layer.get_shape()[-1]), out_channels],
                                  tf.float32, tf.random_normal_initializer(0, 0.01), trainable=True)
            b_h = tf.get_variable('b_h', shape=[out_channels], dtype=tf.float32,
                                  initializer=None, trainable=True)
            H = tf.nn.bias_add(tf.nn.convolution(input_layer, W_h, 'SAME'), b_h)

            W_t = tf.get_variable('W_t', [1, int(input_layer.get_shape()[-1]), out_channels],
                                  tf.float32, tf.random_normal_initializer(0, 0.01), trainable=True)
            b_t = tf.get_variable('b_t', shape=[out_channels], dtype=tf.float32,
                                  initializer=None, trainable=True)
            T = tf.nn.bias_add(tf.nn.convolution(input_layer, W_t, 'SAME'), b_t)
            T = tf.nn.sigmoid(T)
            residual = H*T + input_layer * (1.0 - T)
        elif in_channels != out_channels:
            W_h = tf.get_variable('W_h', [1, int(input_layer.get_shape()[-1]), out_channels],
                                  tf.float32, tf.random_normal_initializer(0, 0.01), trainable=True)
            b_h = tf.get_variable('b_h', shape=[out_channels], dtype=tf.float32,
                                  initializer=None, trainable=True)
            residual = tf.nn.bias_add(tf.nn.convolution(input_layer, W_h, 'SAME'), b_h)
        else:
            pass
            # print("no residual convolution")

        res = input_layer if residual is None else residual

        return tf.nn.relu(dropout2 + res)

def TemporalConvNet(input_layer, num_channels, sequence_length, kernel_size=2,
                    dropout=tf.constant(0.0, dtype=tf.float32), init=False,
                    atten=False, use_highway=False, use_gated=False, name="query"):
    num_levels = len(num_channels)
    counters = {}
    for i in range(num_levels):
        # print(i)
        dilation_size = 2 ** i
        out_channels = num_channels[i]
        input_layer = TemporalBlock(input_layer, out_channels, kernel_size, stride=1, dilation_rate=dilation_size,
                                 counters=counters, dropout=dropout, init=init, atten=atten, gated=use_gated, name=name)

    return input_layer


class DualEncoderTCN(BaseModel):
    def __init__(self, preprocessor, config):
        super(DualEncoderTCN, self).__init__(preprocessor, config)
        self.build_model()
        self.init_saver()
        
    def build_model(self):
        with tf.variable_scope("inputs"):
            # Placeholders for input, output
            self.input_queries = tf.placeholder(tf.int32, [None, self.config.max_length], name="input_queries")
            self.input_replies = tf.placeholder(tf.int32, [None, self.config.max_length], name="input_replies")

            self.queries_lengths = tf.placeholder(tf.int32, [None], name="queries_length")
            self.replies_lengths = tf.placeholder(tf.int32, [None], name="replies_length")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        cur_batch_length = tf.shape(self.input_queries)[0]
        
        # Define learning rate and optimizer
        learning_rate = tf.train.exponential_decay(self.config.learning_rate, 
                                                   self.global_step_tensor,
                                                   decay_steps=50000, 
                                                   decay_rate=0.96,
                                                   staircase=True)
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        
        # Embedding layer
        with tf.variable_scope("embedding"):
            embeddings = tf.Variable(get_embeddings(self.preprocessor.vectorizer.idx2word, self.config), trainable=True, name="embeddings")
            queries_embedded = tf.nn.embedding_lookup(embeddings, self.input_queries, name="queries_embedded")
            replies_embedded = tf.nn.embedding_lookup(embeddings, self.input_replies, name="replies_embedded")
            queries_embedded, replies_embedded = tf.cast(queries_embedded, tf.float32), tf.cast(replies_embedded, tf.float32)
        
        # Dropout
        queries_embedded = tf.nn.dropout(queries_embedded, keep_prob=self.config.dropout_keep_prob)
        replies_embedded = tf.nn.dropout(replies_embedded, keep_prob=self.config.dropout_keep_prob)
        
        # Use TCN same as rnn cell
        encoding_queries = TemporalConvNet(input_layer=queries_embedded, 
                                         num_channels=[256, 256, 256, 256, 256, 256], 
                                         sequence_length = self.queries_lengths, 
                                         kernel_size=self.config.tcn_kernel_size, 
                                         dropout=1-self.config.dropout_keep_prob, 
                                         init=False, 
                                         name="query")
        encoding_replies = TemporalConvNet(input_layer=replies_embedded, 
                                         num_channels=[256, 256, 256, 256, 256, 256], 
                                         sequence_length = self.queries_lengths, 
                                         kernel_size=self.config.tcn_kernel_size, 
                                         dropout=1-self.config.dropout_keep_prob, 
                                         init=False, 
                                         name="reply")
        
        encoding_queries = tf.reduce_mean(encoding_queries, axis=1)
        encoding_replies = tf.reduce_mean(encoding_replies, axis=1)
        
        # Predict a response
        with tf.variable_scope("prediction") as vs:
            M = tf.get_variable("M",
                                shape=[self.config.embed_dim, self.config.embed_dim],
                                initializer=tf.truncated_normal_initializer())
            encoding_queries = tf.matmul(encoding_queries, M)
            
        with tf.variable_scope("negative_sampling") as vs:
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
        # probs = sess.run(self.probs, feed_dict=feed_dict)
        return loss, score, None

    def infer(self, sess, feed_dict=None):
        return sess.run(self.positive_probs, feed_dict=feed_dict)