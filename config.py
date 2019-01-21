import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('f', '', 'kernel')
flags.DEFINE_string('data_path', './dataset/de-en/', 'IWSLT16 TED training data with preprocessed')

flags.DEFINE_integer('sentence_maxlen', 20, 'Max length of the sentence')
flags.DEFINE_integer('batch_size', 32, 'batch size of the data')
