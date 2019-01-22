import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('f', '', 'kernel')
flags.DEFINE_string('data_path', './dataset/de-en/', 'IWSLT16 TED training data with preprocessed')

flags.DEFINE_integer('sentence_maxlen', 20, 'Max length of the sentence')
flags.DEFINE_integer('batch_size', 32, 'batch size of the data')
flags.DEFINE_integer('stack_layer', 6, 'a number of encoder and decoder identical layers, denoted as N')
flags.DEFINE_integer('multi_head', 8, 'a number of parallel attention layers, denoted as h')
flags.DEFINE_integer('key_dim', 64, 'reduced dimension of each head, denoted as d_k')
flags.DEFINE_integer('value_dim', 64, 'reduced dimension of each head, denoted as d_v')
flags.DEFINE_integer('inner_layer', 2048, 'Position-Wise Feed-Forward Networks inner-layer dimensionality, denoted as d_ff')
flags.DEFINE_integer('model_dim', 512, 'To faciliate residual connections, output of dimension is 512, denoted as d_model')
