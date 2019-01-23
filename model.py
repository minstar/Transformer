import numpy as np
import tensorflow as tf

from config import *

# Positional Encoding
def position_encoding(scaling=True):
    # --------------------------- Output --------------------------- #
    # outputs : positional encoded matrix shape of (32, 20, 512) zero padded.

    with tf.variable_scope('positional_encoding'):
        position_idx = tf.tile(tf.expand_dims(tf.range(FLAGS.sentence_maxlen),0), [FLAGS.batch_size,1]) # (32, 20)

        # PE_(pos, 2i) = sin(pos / 10000 ^ (2i / d_model))
        # PE_(pos, 2i+1) = cos(pos / 10000 ^ (2i / d_model))
        position_enc = np.array([[pos / (10000 ** (2*i / FLAGS.model_dim)) for i in range(FLAGS.model_dim)]
                                if pos != 0 else np.zeros(FLAGS.model_dim) for pos in range(FLAGS.sentence_maxlen)],
                                dtype=np.float32)

        # index 0 is all zero
        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # sine functions to 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # cosine functions to 2i + 1

        # convert to tensor
        table = tf.convert_to_tensor(position_enc, dtype=tf.float32)

        outputs = tf.nn.embedding_lookup(table, position_idx)

        # output embedding scaling if needed
        if scaling:
            print ("position encoding scaling is executed")
            outputs *= (FLAGS.model_dim ** 0.5)

    return outputs

# Inputs and Outputs embedding lookup function
def embedding(inputs, input_vocab, padding=True, scaling=True):
    # --------------------------- Input --------------------------- #
    # inputs : (batch_size, sentence max length) shape of input dataset
    # input_vocab : class, composed of token to index dictionary and reversed dictionary

    # --------------------------- Output --------------------------- #
    # outputs : embedding matrix shape of (32, 20, 512) zero padded.
    print ("token number :",len(input_vocab.token2idx))

    with tf.variable_scope("embedding"):
        table = tf.get_variable("word_embedding", shape=[len(input_vocab.token2idx), FLAGS.model_dim], \
                                dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

        if padding:
            table = tf.concat((tf.zeros(shape=[1, FLAGS.model_dim]), table[1:]), axis=0)

        # table : (123154, 512), inputs : (32, 20)
        outputs = tf.nn.embedding_lookup(table, inputs)

        if scaling:
            print ("embedding scaling is executed")
            outputs *= (FLAGS.model_dim ** 0.5)

    return outputs

# Layer Normalization
def layer_norm(inputs):
    # --------------------------- Input --------------------------- #
    # inputs : multi-head attention outputs, shape of (batch_size, sentence max length, model dimension)

    # --------------------------- Output --------------------------- #
    # outputs : layer normalized with inputs, shape of (batch_size, sentence max length, model dimension)

    with tf.variable_scope("layer_normalization"):
        mean, variance = tf.nn.moments(inputs, axes=2, keep_dims=True) # get mean and variance per batch.
        gamma = tf.Variable(tf.ones(inputs.get_shape()[2]))
        beta  = tf.Variable(tf.zeros(inputs.get_shape()[2]))
        normalized = (inputs - mean) / tf.sqrt(variance + 1e-5)
        outputs = gamma * normalized + beta # (32, 20, 512, dtype=float32)

    return outputs

# Multi-Head Attention
def multihead_attention(inputs, masking=False, dropout=True):
    with tf.variable_scope("multihead_attention"):
        # queries, keys, values come from the same place which is input
        queries, keys, values = inputs, inputs, inputs

        # linear transformation
        Q = tf.layers.dense(queries, FLAGS.model_dim, activation=tf.nn.relu, use_bias=True) # (32, 20, 512)
        K = tf.layers.dense(keys, FLAGS.model_dim, activation=tf.nn.relu, use_bias=True) # (32, 20, 512)
        V = tf.layers.dense(values, FLAGS.model_dim, activation=tf.nn.relu, use_bias=True) # (32, 20, 512)

        Q_concat = tf.concat(tf.split(Q, FLAGS.multi_head, axis=2), axis=0) # (8 * 32, 20, 64)
        K_concat = tf.concat(tf.split(K, FLAGS.multi_head, axis=2), axis=0) # (8 * 32, 20, 64)
        V_concat = tf.concat(tf.split(V, FLAGS.multi_head, axis=2), axis=0) # (8 * 32, 20, 64)

        # Multiplication
        K_transpose = tf.transpose(K_concat, perm=[0, 2, 1]) # (256, 64, 20)
        logits = tf.matmul(Q_concat, K_transpose) # (256, 20, 20)

        # Scaling because of variance maintenance
        logits /= FLAGS.key_dim ** 0.5

        # Masking (optional. for decoding)
        if masking:
            pass

        # Softmax and multiply
        outputs = tf.nn.softmax(logits) # (256, 20, 20)

        # Dropout
        outputs = tf.layers.dropout(outputs, rate=FLAGS.dropout ,training=dropout)

        # Context Vectore, denoted as Attention(Q, K, V)
        outputs = tf.matmul(outputs, V_concat) # (256, 20, 64)

        outputs = tf.concat(tf.split(outputs, FLAGS.multi_head, axis=0), axis=2) # (32, 20, 8 * 64)

        # Residual connection
        outputs += inputs # (32, 20, 512)

        # Normalize
        outputs = layer_norm(outputs)

    return outputs

# Position-Wise Feed-Forward Networks
