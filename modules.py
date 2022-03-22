import numpy as np
import tensorflow as tf

def get_token_embeddings(vocab_size, num_node, zero_pad=True):
    with tf.variable_scope("shared_weight_matrix"):
        embeddings = tf.get_variable('weight_mat',
                                   dtype=tf.float32,
                                   shape=(vocab_size, num_node),
                                   initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            embeddings = tf.concat((tf.zeros(shape=[1, num_node]),
                                    embeddings[1:, :]), 0)
    return embeddings

def positional_encoding(input, max_len, masking=True, scope='positional_encoding'):
    E = input.get_shape().as_list()[-1]
    N, T = tf.shape(input)[0], tf.shape(input)[1]
    
    with tf.variable_scope(scope, reuse=tf.REUSE):
        position_idx = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])
        position_encoder = np.array([
                                    [pos / np.power(10000, (i-i%2)/E) for i in range(E)] for pos in range(max_len)])

        position_encoder[:, 0::2] = np.sin(position_encoder[:, 0::2])  # dim 2i
        position_encoder[:, 1::2] = np.cos(position_encoder[:, 1::2])  # dim 2i+1
        position_encoder = tf.convert_to_tensor(position_encoder, tf.float32) # (maxlen, E)

        res = tf.nn.embedding_lookup(position_encoder, position_idx)
    
    res = tf.to_float(res)
    return res

def mask(input, key=None, type=None):
    pass

def self_attention(q, k, v, key, scope='self_attention'):
    with tf.variable_scope(scope, reuse=tf.REUSE):
        d_k = q.get_shape().as_list()[-1]
        output = tf.matmul(q, tf.transpose(k, [0, 2, 1]))  # (N, T_q, T_k)
        output /= d_k ** 0.5

        # key masking
        output = mask(output, key_masks=key, type="key")

        # softmax
        output = tf.nn.softmax(output)
        attention = tf.transpose(output, [0, 2, 1])
        tf.summary.image("attention", tf.expand_dims(attention[:1], -1))

        res = tf.matmul(output, v)

    return res

def multihead_attention(q, k, v, key, num_head=8, scope='multihead_attention'):
    pass

def feed_forward(input, num_node, scope='feed_forward'):
    with tf.variable_scope(scope, reuse=tf.REUSE):
        nn = tf.layers.dense(input, num_node[0], activation=tf.nn.relu)
        nn = tf.layers.dense(nn, num_node[1])
        nn += input     # residual connection
        res = layer_norm(nn)
    return res

def layer_norm(input, epsilon=1e-6, scope='ln'):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        input_shape = input.get_shape()
        params_shape = input_shape[-1:]
        
        mean, variance = tf.nn.moments(input, [-1], keep_dims=True)
        beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (input - mean) / ( (variance + epsilon) ** (.5) )
        res = gamma * normalized + beta

    return res