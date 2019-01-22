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

# Position-Wise Feed-Forward Networks
# Scaled Dot-Product Attention
# Multi-Head Attention
