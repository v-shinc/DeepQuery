#encoding=utf8
import tensorflow as tf
from collections import Counter
import random
import numpy as np
from time import time
import sys
import os
import json
from data_helpers import CombinedTrainData, CombinedTestData, ReadableTestData, load_word2vec

class QueryDNNInter(object):
    def __init__(self, query_len, title_len, num_word, embed_size, hidden_size1, hidden_size2, gpu, load_path, max_grad_norm=5):

        self.queries = tf.placeholder(tf.int32, [None, query_len], name='query')
        self.titles = tf.placeholder(tf.int32, [None, title_len], name='titles')
        self.neg_titles = tf.placeholder(tf.int32, [None, title_len], name='neg_titles')
        self.margins = tf.placeholder(tf.float32, [None], name='margin')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.query_len = query_len
        self.title_len = title_len
        self.embed_size = embed_size

        with tf.variable_scope('lookup_layer'):
            with tf.device("/gpu:%s" % gpu):
                initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
                word_embedding = tf.get_variable('word_embedding', [num_word, embed_size], initializer=initializer)
                # word_embedding = tf.Variable(init_embeding, name='word_embedding')
                pos_title_word = tf.nn.embedding_lookup(word_embedding, self.titles)
                neg_title_word = tf.nn.embedding_lookup(word_embedding, self.neg_titles)
                query_word = tf.nn.embedding_lookup(word_embedding, self.queries)



        with tf.device("/gpu:%s" % (gpu)):
            with tf.name_scope('sum_layer'):
                self.query_vec = tf.reduce_sum(query_word, 1)
                self.pos_title_vec = tf.reduce_sum(pos_title_word, 1)
                neg_title_vec = tf.reduce_sum(neg_title_word, 1)

            with tf.variable_scope('match_layer'):
                w_m = tf.get_variable('w_m', [embed_size, embed_size], initializer=initializer)
                proj_query = tf.matmul(self.query_vec, w_m)
                pos_match = tf.reshape(tf.reduce_sum(proj_query * self.pos_title_vec, 1), [-1, 1])
                neg_match = tf.reshape(tf.reduce_sum(proj_query * neg_title_vec, 1), [-1, 1])

                pos_inter = tf.reshape(tf.batch_matmul(query_word, pos_title_word, adj_y=True), [-1, title_len*query_len])
                neg_inter = tf.reshape(tf.batch_matmul(query_word, neg_title_word, adj_y=True), [-1, title_len*query_len])


            with tf.name_scope('join_layer'):
                pos_concat = tf.concat(1, [self.query_vec, pos_match, pos_inter, self.pos_title_vec])
                neg_concat = tf.concat(1, [self.query_vec, neg_match, neg_inter, neg_title_vec])

            with tf.variable_scope('hidden1'):
                initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
                w1 = tf.get_variable('w1', [embed_size * 2 + 1 + title_len*query_len, hidden_size1], initializer=initializer) * 0.1
                b1 = tf.get_variable('b1', [hidden_size1], initializer=tf.constant_initializer(0))
                pos_h1 = tf.nn.relu(tf.add(tf.matmul(pos_concat, w1), b1))
                neg_h1 = tf.nn.relu(tf.add(tf.matmul(neg_concat, w1), b1))

            with tf.variable_scope('hidden2'):
                w2 = tf.get_variable('w2', [hidden_size1, hidden_size2], initializer=initializer) * 0.1
                b2 = tf.get_variable('b2', [hidden_size2], initializer=tf.constant_initializer(0))
                pos_h2 = tf.nn.relu(tf.add(tf.matmul(pos_h1, w2), b2))
                neg_h2 = tf.nn.relu(tf.add(tf.matmul(neg_h1, w2), b2))

            with tf.name_scope("dropout"):
                h_pos_drop = tf.nn.dropout(pos_h2, self.dropout_keep_prob)
                h_neg_drop = tf.nn.dropout(neg_h2, self.dropout_keep_prob)

            with tf.variable_scope('output'):
                w3 = tf.get_variable('w3', [hidden_size2, 1], initializer=initializer) * 0.1
                b3 = tf.get_variable('b3', [1], initializer=tf.constant_initializer(0))
                self.pos_sim = tf.sigmoid(tf.add(tf.matmul(h_pos_drop, w3), b3))
                neg_sim = tf.sigmoid(tf.add(tf.matmul(h_neg_drop, w3), b3))
                # self.pos_sim = tf.sigmoid(tf.add(tf.matmul(pos_h2, w3), b3))
                # neg_sim = tf.sigmoid(tf.add(tf.matmul(neg_h2, w3), b3))

            with tf.name_scope("loss"):
                self.loss = tf.reduce_sum(tf.maximum(0., neg_sim + 0.1 - self.pos_sim))
                # self.loss = tf.reduce_sum(tf.maximum(0., neg_sim + self.margins - self.pos_sim))

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

    def fit(self, queries, pos_titles, neg_titles, margins, dropout_keep_prob):
        feed_dict = {
            self.queries: queries,
            self.titles: pos_titles,
            self.neg_titles: neg_titles,
            self.margins: margins,
            self.dropout_keep_prob: dropout_keep_prob
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

    def get_query_vec(self, queries):
        return self.session.run(self.query_vec, {self.queries: queries})

    def get_title_vec(self, titles):
        return self.session.run(self.pos_title_vec, {self.titles: titles})


MAX_QUERY_LEN = 8
MAX_TITLE_LEN = 20
fn_word2vec = 'data/index.vec.txt'
EMBED_SIZE = 128


def train(fn_train, fn_feature, fn_word,
          num_epoch, batch_size, dirname,
          embed_size, hidden_size1, hidden_size2, load, method, gpu, test_conf):

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
    PAD = num_word
    num_word += 1
    idx2word[PAD] = ''

    # embedding = load_word2vec(fn_word2vec, num_word, EMBED_SIZE)

    test_data = eval(test_conf['test_cls'])(test_conf['fn_test'], test_conf['fn_title'], word2idx, MAX_QUERY_LEN,
                                            MAX_TITLE_LEN, PAD)
    dataset = CombinedTrainData(fn_train, fn_feature, MAX_QUERY_LEN, MAX_TITLE_LEN, PAD, method)


    n_batch = dataset.num // batch_size + dataset.num % batch_size

    if load:
        load_path = save_path
    else:
        load_path = None
    fout_log = open(log_path, 'a')
    # with tf.Graph().as_default():


    model = QueryDNNInter(MAX_QUERY_LEN, MAX_TITLE_LEN, num_word, embed_size, hidden_size1, hidden_size2, gpu, load_path)


    best_acc = test_data.evaluate(model, idx2word, dev_res_path)
    print "init acc", best_acc
    tic = time()
    lno = 0
    total_loss = 0
    for epoch_index in xrange(num_epoch):
        lno = lno % 500000
        for i in xrange(n_batch):
            lno += batch_size
            if lno % 1000 == 0:
                sys.stdout.write('process to %d\r' % lno)
                sys.stdout.flush()

            queries, pos_titles, neg_titles, margins = dataset.next_batch(batch_size, n_repeat=1)
            loss_one = model.fit(queries, pos_titles, neg_titles, margins, 0.5)
            total_loss += loss_one
            if lno % 500000 == 0:
                print >> fout_log, '# %s: loss = %s, it costs %ss' % (epoch_index, total_loss, time() - tic)
                print '# %s: loss = %s, it costs %ss' % (epoch_index, total_loss, time() - tic)
                old_path = model.save('%s-%s' % (save_path, epoch_index))
                # old_path = saver.save(session, '%s-%s' % (save_path, epoch_index))

                tic = time()
                acc = test_data.evaluate(model, idx2word, dev_res_path)
                print >> fout_log, "acc = %s, it costs %ss" % (acc, time() - tic)
                print "acc = %s, it costs %ss" % (acc, time() - tic)
                if best_acc < acc:
                    best_acc = acc
                    os.rename(old_path, save_path)
                    os.rename('%s.meta' % old_path, '%s.meta' % save_path)
                    print "best mode", old_path
                    print >> fout_log, "best mode", old_path
                total_loss = 0
                tic = time()
        print "finish one pass"

def dnn_click_clean():
    fn_train = 'data/com.query.click.clean.v'
    fn_dev = 'data/cut.query.dat'
    fn_feature = 'data/feature.v'
    fn_readable_feature = 'data/feature.dat'
    fn_word = 'data/word.list'
    # fn_gid_count = 'data/gid.count.v'

    # PV TEST
    train(fn_train, fn_feature, fn_word,
          num_epoch=500, batch_size=300, dirname='dnn_click_clean',
          embed_size=100, hidden_size1=100, hidden_size2=50, load=False, method=2, gpu=8,
          test_conf={'fn_test': fn_dev, 'fn_title': fn_readable_feature ,'test_cls': 'ReadableTestData'})


    # train(fn_train, fn_feature, fn_word,
    #       num_epoch=500, batch_size=300, dirname='dnn_click_clean',
    #       embed_size=100, hidden_size1=100, hidden_size2=50, load=True, method=2, gpu=8,
    #       test_conf={'fn_test': 'data/com.query.click.test.shuffle.dat', 'fn_title': 'data/title.test.txt', 'test_cls': CombinedTestData})


def dnn_click():
    fn_train = 'data/com.query.click.v'
    fn_dev = 'data/com.query.click.test.shuffle.dat'
    fn_feature = 'data/feature.v'
    fn_dev_title = 'data/title.test.txt'
    fn_word = 'data/word.list'
    # train(fn_train, fn_feature, fn_word,
    #       num_epoch=500, batch_size=300, dirname='dnn-click',
    #       embed_size=100, hidden_size1=100, hidden_size2=50, load=False, method=2, gpu=9,
    #       test_conf={'fn_test': fn_dev, 'fn_title': fn_dev_title,
    #                  'test_cls': 'CombinedTestData'})

    # train(fn_train, fn_feature, fn_word,
    #       num_epoch=500, batch_size=300, dirname='dnn-click-inter',
    #       embed_size=100, hidden_size1=100, hidden_size2=50, load=False, method=2, gpu=8,
    #       test_conf={'fn_test': fn_dev, 'fn_title': fn_dev_title,
    #                  'test_cls': 'CombinedTestData'})

    # train(fn_train, fn_feature, fn_word,
    #       num_epoch=500, batch_size=300, dirname='dnn-click-inter-constant',
    #       embed_size=100, hidden_size1=100, hidden_size2=100, load=False, method=2, gpu=4,
    #       test_conf={'fn_test': fn_dev, 'fn_title': fn_dev_title,
    #                  'test_cls': 'CombinedTestData'})

    train('data/com.query.click.20150924.20160630.v', 'data/title.20150924.20160630.v', fn_word,
          num_epoch=500, batch_size=300, dirname='dnn-click-inter-large',
          embed_size=100, hidden_size1=100, hidden_size2=100, load=False, method=2, gpu=4,
          test_conf={'fn_test': fn_dev, 'fn_title': fn_dev_title,
                     'test_cls': 'CombinedTestData'})

if __name__ == '__main__':

    # dnn_click_clean()
    dnn_click()


