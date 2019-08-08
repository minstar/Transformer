import numpy as np
import tensorflow as tf
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from preprocess import *
from config import *
from model import *

def main(_):
    # -------------------- GPU type setting -------------------- #
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    gpu_config.gpu_options.per_process_gpu_memory_fraction = 1

    # -------------------- Import data -------------------- #
    en_vocab, de_vocab, tr_zip_file, dev_zip_file, ger_glove_dict, eng_glove_dict = preprocess()

    # -------------------- Building Training Graph -------------------- #
    with tf.Graph().as_default(), tf.Session(config=gpu_config) as sess:
        tf.set_random_seed(7777)
        np.random.seed(seed=7777)

        initializer = tf.random_normal_initializer()

        # -------------------- Make training graph -------------------- #
        with tf.variable_scope("Model", initializer=initializer):
            tr_model = model_graph(source=de_vocab, target=en_vocab, src_glove=ger_glove_dict, trg_glove=eng_glove_dict)
            _, tr_loss = tr_model.loss_fn()
            tr_global_step, _, tr_train_op = tr_model.train_fn()

        # -------------------- Save Model -------------------- #
        saver = tf.train.Saver(max_to_keep=10)
        sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter('./train_dir', graph=sess.graph)
        print ('----- Training start with initialized variable -----\n')

        # -------------------- Training Start -------------------- #
        for epoch_idx in range(FLAGS.epoch):
            train_loss, eval_loss, preds, ce_list = [], [], [], []
            start = time.time()

            for idx, (x, y) in enumerate(tr_zip_file):
                # index 0 ~ 4109
                input_ = {tr_model.enc_inputs : x, tr_model.dec_inputs : y}
                loss, global_step, _, cross_entropy, pred = sess.run([tr_loss, tr_global_step, tr_train_op, tr_model.cross_entropy, tr_model.pred], input_)

                train_loss.append(loss)

                if (idx+1) % FLAGS.verbose == 0:
                    print ('epoch : %d, global_step : %d, loss : %.3f, time : %.2f\n' % (epoch_idx, global_step, loss, time.time() - start))

            for idx, (x, y) in enumerate(dev_zip_file):
                # index 0 ~ 18
                input_ = {tr_model.enc_inputs:x, tr_model.dec_inputs:y}
                loss, cross_entropy, pred = sess.run([tr_model.loss, tr_model.cross_entropy, tr_model.pred], input_)

                eval_loss.append(loss)
                preds.append(pred)
                ce_list.append(cross_entropy)

                print ('dev-epoch : %d, loss : %.3f, time : %.2f\n' % (epoch_idx, loss, time.time() - start))

            print ('one epoch done, spend time :', time.time() - start)
            saver.save(sess, '%s/epoch%d_%.4f.model.ckpt' % ('./train_dir', epoch_idx, np.mean(train_loss) / len(tr_zip_file)))
            print ('Successfully saved model\n')
            summary = tf.Summary(value=[tf.Summary.Value(tag="Training_loss", simple_value=np.mean(train_loss) / len(tr_zip_file))])
            summary_writer.add_summary(summary, global_step)

if __name__ == "__main__":
    tf.app.run()
