#encoding=utf8
import tensorflow as tf
from collections import Counter
import random
import numpy as np
from data_helpers import CombinedTrainData, CombinedTestData, ReadableTestData
from time import time
import sys
import os
import json


class QueryRNNPair(object):
    def __init__(self, query_len, title_len, num_word, embed_size, gpu, load_path, max_grad_norm=5):
        self.queries = tf.placeholder(tf.int32, [None, query_len], name='query')
        self.titles = tf.placeholder(tf.int32, [None, title_len], name='titles')
        self.neg_titles = tf.placeholder(tf.int32, [None, title_len], name='neg_titles')
        self.margins = tf.placeholder(tf.float32, [None], name='margin')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.query_len = query_len
        self.title_len = title_len
        self.embed_size = embed_size

        # with tf.variable_scope('embedding'):
        #     with tf.device("/gpu:%s" % gpu):
        #
        #         initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
        #         word_embeding_t = tf.get_variable('word_embedding_t', [num_word, embed_size], initializer=initializer)
        #         title_word = tf.nn.embedding_lookup(word_embeding_t, self.titles, name='title_word')
        #         neg_title_word = tf.nn.embedding_lookup(word_embeding_t, self.neg_titles, name='neg_title_word')
        #
        #
        #     with tf.device("/gpu:%s" % (gpu+1)):
        #         word_embedding_q = tf.get_variable('char_embedding_q', [num_word, embed_size], initializer=initializer)
        #         query_word = tf.nn.embedding_lookup(word_embedding_q, self.queries)

        with tf.variable_scope('embedding'):
            with tf.device("/gpu:%s" % gpu):

                initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
                word_embedding = tf.get_variable('word_embedding', [num_word, embed_size], initializer=initializer)
                title_word = tf.nn.embedding_lookup(word_embedding, self.titles, name='title_word')
                neg_title_word = tf.nn.embedding_lookup(word_embedding, self.neg_titles, name='neg_title_word')
                query_word = tf.nn.embedding_lookup(word_embedding, self.queries)



        with tf.device("/gpu:%s" % (gpu)):
            with tf.variable_scope('title_rnn'):
                self.h_pos_title = self.rnn(title_word, reuse=False)
                h_neg_title = self.rnn(neg_title_word, reuse=True)

            with tf.variable_scope('query_rnn'):
                self.h_query = self.rnn(query_word, reuse=False)


            with tf.name_scope("dropout"):
                h_pos_drop = tf.nn.dropout(
                    self.h_pos_title,
                    self.dropout_keep_prob)

                h_neg_drop = tf.nn.dropout(
                    h_neg_title,
                    self.dropout_keep_prob)

                h_query_drop = tf.nn.dropout(
                    self.h_query,
                    self.dropout_keep_prob
                )

            with tf.name_scope("loss"):
                self.pos_sim = self.sim(h_query_drop, h_pos_drop)
                neg_sim = self.sim(h_query_drop, h_neg_drop)


                self.l1 = tf.reduce_sum(tf.maximum(0., neg_sim + self.margins - self.pos_sim))
                self.loss = self.l1

            tvars = tf.trainable_variables()
            # self.grads = tf.gradients(self.loss, tvars)
            self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), max_grad_norm)
            optimizer = tf.train.AdamOptimizer(1e-3)
            self.train_op = optimizer.apply_gradients(zip(self.grads, tvars))
            # self.train_op = optimizer.minimize(self.loss)

        # self.session = session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        self.session = tf.InteractiveSession(config=config)
        self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=1)
        if load_path:
            self.saver.restore(self.session, load_path)
        else:
            self.session.run(tf.initialize_all_variables())



    def sim(self, u, v):
        with tf.name_scope("cosine_layer"):
            dot = tf.reduce_sum(tf.mul(u, v), 1)
            sqrt_u = tf.sqrt(tf.reduce_sum(u**2, 1))
            sqrt_v = tf.sqrt(tf.reduce_sum(v**2, 1))
            epsilon = 1e-5
            cosine = dot / (tf.maximum(sqrt_u * sqrt_v, epsilon))
            # cosine = tf.maximum(dot / (tf.maximum(sqrt_u * sqrt_v, epsilon)), epsilon)
        return cosine

    # def sim(self, x, y):
    #     x = tf.nn.l2_normalize(x, 1, epsilon=1e-5)
    #     y = tf.nn.l2_normalize(y, 1, epsilon=1e-5)
    #     return tf.reduce_sum(x * y, reduction_indices=1)  # [batch_size]

    def rnn(self, embeded_words, reuse):
        seq_len = embeded_words.get_shape()[1]
        with tf.variable_scope('rnn', reuse=reuse):
            embeded_words = [tf.squeeze(_inp, [1]) for _inp in tf.split(1, seq_len, embeded_words)]
            title_cell = tf.nn.rnn_cell.GRUCell(self.embed_size)
            title_outputs, title_state = tf.nn.rnn(title_cell, embeded_words, dtype=tf.float32)
        # return title_state
        return tf.reduce_max(tf.pack(title_outputs), 0)

    def fit(self, queries, pos_titles, neg_titles, margins, dropout_keep_prob):
        feed_dict = {
                        self.queries: queries,
                        self.titles: pos_titles,
                        self.neg_titles: neg_titles,
                        self.margins: margins,
                        self.dropout_keep_prob:dropout_keep_prob
                    }
        _, loss_one = self.session.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss_one

    def predict(self, query, titles):
        scores = self.session.run(
            self.pos_sim,
            {self.queries: query, self.titles: titles, self.dropout_keep_prob: 1})

        return [float(s) for s in scores]

    def save(self, save_path):
        return self.saver.save(self.session, save_path)

MAX_QUERY_LEN = 8
MAX_TITLE_LEN = 20

# def test_cls(test_name):
#     if test_name == 'CombinedTestData':
#         return CombinedTestData
#
#     elif test_name == 'ReadableTestData':
#         return ReadableTestData
#     else:
#         raise ValueError('Test name is illegal')


def train(fn_train, fn_feature, fn_word,
          num_epoch, batch_size, dirname,
          embed_size, load, train_method, gpu, test_conf):

    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", dirname))
    print("Writing to {}".format(out_dir))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    save_path = os.path.join(checkpoint_dir, "model")
    dev_res_path = os.path.join(out_dir, 'dev.res')
    log_path = os.path.join(out_dir, 'train.log')
    config_path = os.path.join(out_dir, dirname+'_config.json')

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    config = locals()
    with open(config_path, 'w') as fout:
        print >> fout, json.dumps(config)


    idx2word = dict()
    word2idx = dict()
    num_word = 0
    with open(fn_word) as fin:
        for line in fin:
            w = line.decode('utf8').strip()
            idx2word[num_word] = w
            word2idx[w] = num_word
            num_word += 1
    WPAD = num_word
    num_word += 1
    idx2word[WPAD] = ''

    test_data = eval(test_conf['test_cls'])(test_conf['fn_test'], test_conf['fn_title'], word2idx, MAX_QUERY_LEN,
                                      MAX_TITLE_LEN, WPAD)
    dataset = CombinedTrainData(fn_train, fn_feature, MAX_QUERY_LEN, MAX_TITLE_LEN, WPAD, train_method)

    n_batch = dataset.num // batch_size + dataset.num % batch_size

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False

    if load:
        load_path = save_path
    else:
        load_path = None
    fout_log = open(log_path, 'a')
    # with tf.Graph().as_default():
    #     session = tf.Session(config=config)
    #     with session.as_default():
    model = QueryRNNPair(MAX_QUERY_LEN, MAX_TITLE_LEN, num_word, embed_size, gpu, load_path)

    best_acc = test_data.evaluate(model, idx2word, dev_res_path)
    print "init acc", best_acc
    total_loss = 0
    lno = 0
    tic = time()

    for epoch_index in xrange(num_epoch):
        lno = lno % 500000
        for i in xrange(n_batch):
            lno += batch_size
            if lno % 1000 == 0:
                sys.stdout.write('process to %d\r' % lno)
                sys.stdout.flush()

            queries, pos_titles, neg_titles, margins = dataset.next_batch(batch_size, n_repeat=1)

            loss_one = model.fit(queries, pos_titles, neg_titles, margins, 0.85)
            total_loss += loss_one
            if lno % 500000 == 0:
                print >> fout_log, '# %s: loss = %s, it costs %ss' % (epoch_index, total_loss, time() - tic)
                print '# %s: loss = %s, it costs %ss' % (epoch_index, total_loss, time() - tic)
                old_path = model.save('%s-%s' % (save_path, epoch_index))
                tic = time()
                acc = test_data.evaluate(model, idx2word,  dev_res_path)
                print >> fout_log, "acc = %s, it costs %ss" % (acc, time() - tic)
                print "acc = %s, it costs %ss" % (acc, time()-tic)
                if best_acc < acc:
                    best_acc = acc
                    os.rename(old_path, save_path)
                    os.rename('%s.meta' % old_path, '%s.meta' % save_path)
                    print "best mode", old_path
                total_loss = 0
                tic = time()
        print "finish one pass"


def average_rank_model():
    fn_train = 'data/combine.query.v'
    fn_feature = 'data/feature.v'
    fn_word = 'data/word.list'
    train(fn_train, fn_feature, fn_word,
          num_epoch=500, batch_size=500, dirname='com-rnn',
          embed_size=60, load=True, train_method=1, gpu=7,
          test_conf={'fn_test': 'data/cut.query.dat', 'fn_title': 'data/feature.dat', 'test_cls': 'ReadableTestData'})

def rnn_click():
    fn_train = 'data/com.query.click.v'
    fn_feature = 'data/feature.v'
    fn_word = 'data/word.list'

    # train(fn_train, fn_feature, fn_word,
    #       num_epoch=500, batch_size=500, dirname='rnn-click',
    #       embed_size=60, load=False, train_method=2, gpu=3,
    #       test_conf={'fn_test': 'data/com.query.click.test.shuffle.dat', 'fn_title': 'data/title.test.txt',
    #                  'test_cls': 'CombinedTestData'})

    train(fn_train, fn_feature, fn_word,
          num_epoch=500, batch_size=500, dirname='rnn-click-repeat1',
          embed_size=60, load=False, train_method=2, gpu=5,
          test_conf={'fn_test': 'data/com.query.click.test.shuffle.dat', 'fn_title': 'data/title.test.txt',
                     'test_cls': 'CombinedTestData'})

    # PV TEST
    # train(fn_train, fn_feature, fn_word,
    #       num_epoch=500, batch_size=500, dirname='com-rnn-click-state',
    #       embed_size=60, load=False, train_method=2, gpu=4,
    #       test_conf={'fn_test': 'data/cut.query.dat', 'fn_title': 'data/feature.dat', 'test_cls': 'ReadableTestData'})


def rnn_click_clean():

    fn_train = 'data/com.query.click.clean.v'
    fn_feature = 'data/feature.v'
    fn_word = 'data/word.list'
    train(fn_train, fn_feature, fn_word,
          num_epoch=500, batch_size=500, dirname='rnn-click-clean',
          embed_size=60, load=True, train_method=2, gpu=9,
          test_conf={'fn_test': 'data/com.query.click.test.shuffle.dat', 'fn_title': 'data/title.test.txt',
                     'test_cls': 'CombinedTestData'})

if __name__ == '__main__':
    # mini_model()
    # average_rank_model()

    # rnn_click_clean()
    rnn_click()
