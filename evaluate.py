import numpy as np
import tensorflow as tf
import random
import time
import pdb
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from nltk.translate import bleu_score
from preprocess import *
from config import *
from model import *

def main(_):
    # -------------------- GPU type setting -------------------- #
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    gpu_config.gpu_options.per_process_gpu_memory_fraction = 1

    # -------------------- Import data -------------------- #
    en_vocab, de_vocab, tst_zip_file, ger_glove_dict, eng_glove_dict = preprocess()

    # -------------------- Building Training Graph -------------------- #
    with tf.Graph().as_default(), tf.Session(config=gpu_config) as sess:
        random.seed(7777)
        tf.set_random_seed(7777)
        np.random.seed(seed=7777)

        # -------------------- Make training graph -------------------- #
        with tf.variable_scope("Model"):
            tst_model = model_graph(source=de_vocab, target=en_vocab, src_glove=ger_glove_dict, trg_glove=eng_glove_dict)
            _, _ = tst_model.loss_fn()
            tst_global_step, _, _ = tst_model.train_fn()

        # -------------------- Load trained model -------------------- #
        sess.run(tf.global_variables_initializer())
        loader = tf.train.Saver()
        loader.restore(sess, tf.train.latest_checkpoint('./train_dir'))

        preds, eval_loss, ce, bleu_scores, y_s = [], [], [], [], []
        start = time.time()

        for idx, (x, y) in enumerate(tst_zip_file):
            # index 0 ~ 4109
            feed_dict = {tst_model.enc_inputs : x, tst_model.dec_inputs : y}
            loss, cross_entropy, pred = sess.run([tst_model.loss, tst_model.cross_entropy, tst_model.pred], feed_dict)

            preds.append(pred)
            eval_loss.append(loss)
            ce.append(cross_entropy)
            y_s.append(y)

            print ('global_step : %d, loss : %.3f, time : %.2f\n' % (idx, loss, time.time() - start))
            print (bleu_score.sentence_bleu([preds[0][0]], y_s[0][0]))

        print ('elapsed time :', time.time() - start)
        print ()

if __name__ == "__main__":
    tf.app.run()
