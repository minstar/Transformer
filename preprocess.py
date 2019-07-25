import pdb
import time
import numpy as np
import tensorflow as tf
import sentencepiece as spm

from config import *

class Vocab:
    def __init__(self, token2idx=None, idx2token=None):
        self.token2idx = token2idx or dict()
        self.idx2token = idx2token or dict()

    def new_token(self, token):
        # token : word of dataset
        # return : index of token
        if token not in self.token2idx:
            index = len(self.token2idx)
            self.token2idx[token] = index
            self.idx2token[index] = token
        return self.token2idx[token]

    def get_token(self, index):
        # index : position number of token
        # return : word or character of index
        return self.idx2token[index]

    def get_index(self, token):
        # token : word of dataset
        # return : index of token
        return self.token2idx[token]

def make_data(tr_file_name=None):
    # --------------------------- Input --------------------------- #
    # file_name : file name of preprocessed data of TED video script

    # --------------------------- Output --------------------------- #
    # word_vocab : class, composed of token to index dictionary and reversed dictionary
    # word_list  : total sentences of training data as index list
    # word_maxlen: max length of vocabulary dictionary words
    word_vocab = Vocab()

    word_list, whole_sent = [], []

    EOS = '|' # End of sentence token
    PAD = '<PAD>' # For Padding for matching all sentence length
    UNK = '<UNK>' # For unknown word at dev and test set
    SOS = '<SOS>' # For starting token

    word_vocab.new_token(PAD)
    word_vocab.new_token(EOS) # End of sentence token to index 1
    word_vocab.new_token(UNK) # Unknown word will appear at dev and test set
    word_vocab.new_token(SOS) # End of sentence token to index 3

    word_maxlen = 0
    start = time.time()

    with open(FLAGS.data_path + tr_file_name, 'r', encoding='utf-8') as f:
        line = f.readlines()

        for one_line in line:
            one_sent = list()
            if '.en' in tr_file_name:
                one_sent.append(word_vocab.get_index(SOS))

            for word in one_line.split():
                one_sent.append(word_vocab.new_token(word))

                if len(word) > word_maxlen:
                    word_maxlen = len(word) # 61 is the longest word

            # End of Sentence
            one_sent.append(word_vocab.get_index(EOS))
            word_list.append(one_sent)
            whole_sent.extend(one_sent)

    print (tr_file_name + " train file making indexing table time: %.3f" % (time.time() - start))
    print ("dictionary size : ", len(word_vocab.token2idx))
    print ("total number of sentences : ", len(word_list))
    print ("max length of the word : ", word_maxlen)
    print ()

    return word_vocab, word_list, whole_sent, word_maxlen

def make_dev_data(dev_file_name, tr_word_vocab):
    # --------------------------- Input --------------------------- #
    # file_name : file name of preprocessed data of TED video script

    # --------------------------- Output --------------------------- #
    # word_vocab : class, composed of token to index dictionary and reversed dictionary
    # word_list  : total sentences of training data as index list
    # word_maxlen: max length of vocabulary dictionary words

    EOS = '|'
    UNK = '<UNK>'
    SOS = '<SOS>'
    dev_word_list = []

    start = time.time()
    with open(FLAGS.data_path + dev_file_name, 'r', encoding='utf-8') as f:
        line = f.readlines()
        for one_line in line:
            one_sent = list()
            if one_line.split()[0] == '<seg':
                if '.en' in dev_file_name:
                    one_sent.append(tr_word_vocab.get_index(SOS))

                for word in one_line.split()[2:-1]:
                    if word in tr_word_vocab.token2idx:
                        one_sent.append(tr_word_vocab.get_index(word))
                    else:
                        one_sent.append(tr_word_vocab.get_index(UNK))

            # End of Sentence
            one_sent.append(tr_word_vocab.get_index(EOS))
            dev_word_list.append(one_sent)

    print (dev_file_name + " dev file making indexing table time: %.3f" % (time.time() - start))
    print ("total number of sentences : %d\n" % len(dev_word_list))

    return dev_word_list

def get_data(en_list, de_list):
    # --------------------------- Input --------------------------- #
    # en_list : 196884 number of sentences at ted video, composed of English language data
    # de_list : 196884 number of sentences at ted video, composed of German language data

    # --------------------------- Output --------------------------- #
    # X : padded results of index list, composed of English lnaguage data (131549, 20)
    # Y : padded results of index list, composed of German language data  (131549, 20)

    source_sent = list()
    target_sent = list()

    for idx in range(len(de_list)):
        # Remove sentence length is longer than 40 words
        if len(de_list[idx]) == 1:
            continue
        source_sent.append(np.array(de_list[idx], dtype=np.int32))
        target_sent.append(np.array(en_list[idx], dtype=np.int32))

    # make the shape of Source matrix and Target matrix
    X = np.zeros([len(source_sent), FLAGS.sentence_maxlen], dtype=np.int32)
    Y = np.zeros([len(target_sent), FLAGS.sentence_maxlen], dtype=np.int32)

    # Padding with the shape of (sentence number, sentence length)
    for idx, (x, y) in enumerate(zip(source_sent, target_sent)):
        X[idx, :len(x[:FLAGS.sentence_maxlen])] = x[:FLAGS.sentence_maxlen]
        Y[idx, :len(y[:FLAGS.sentence_maxlen])] = y[:FLAGS.sentence_maxlen]

    print ("Source Matrix Shape (DE):", X.shape)
    print ("Target Matrix Shape (EN):", Y.shape)
    print ()
    print ('------------------------ Show the example case ------------------------')
    print (X[0])
    print (Y[0])

    print ('------------------------ Show the example case ------------------------')
    print (X[10])
    print (Y[10])

    return X, Y

def batch_loader(X, Y):

    reduced_length = len(X) // FLAGS.batch_size * FLAGS.batch_size # 131520

    X = X[:reduced_length]
    Y = Y[:reduced_length]

    print ("Reduced Source Matrix shape (DE):", X.shape)
    print ("Reduced Target Matrix shape (EN):", Y.shape)

    X = np.reshape(X, newshape=(FLAGS.batch_size, -1, FLAGS.sentence_maxlen))
    Y = np.reshape(Y, newshape=(FLAGS.batch_size, -1, FLAGS.sentence_maxlen))

    print ("Shape of Source Matrix (DE):", X.shape)
    print ("Shape of Target Matrix (EN):", Y.shape)

    X = np.transpose(X, axes=(1,0,2))
    Y = np.transpose(Y, axes=(1,0,2))

    print ("Shape of Source Matrix (DE):", X.shape)
    print ("Shape of Target Matrix (EN):", Y.shape)

    # while training, yield the shape of (batch_size, sentence max length) in X and Y
    zip_file = list(zip(X, Y))

    return X, Y, zip_file

def use_wpm(need_model=False):
    if need_model:
        templates = '--input={} --model_prefix={} --vocab_size={}'
        input_file = './dataset/de-en/train.de'
        prefix = 'WPM_de_5000'
        vocab_size = 5000
        cmd = templates.format(input_file, prefix, vocab_size)
        spm.SentencePieceTrainer.Train(cmd)

    sp = spm.SentencePieceProcessor()
    # print (sp.EncodeAsPieces(sentence))

    # TODO
    # utilize vocab and model to make source and target word embedding vocab dictionary

    return 0

def preprocess():

    de_vocab, de_list, _, _ = make_data(tr_file_name='train.de')
    en_vocab, en_list, _, _ = make_data(tr_file_name='train.en')

    if FLAGS.is_dev:
        de_list = make_dev_data('IWSLT16.TED.dev2010.de-en.de.xml', de_vocab)
        en_list = make_dev_data('IWSLT16.TED.dev2010.de-en.en.xml', en_vocab)
    elif FLAGS.is_test:
        pass

    X, Y = get_data(en_list, de_list)
    X, Y, zip_file = batch_loader(X, Y)

    return X, Y, en_vocab, de_vocab, zip_file

if __name__=="__main__":
    X, Y, en_vocab, de_vocab, zip_file = preprocess()
