#encoding=utf8
import tensorflow as tf
from collections import Counter
import random
import numpy as np
from data_helpers import CombinedTrainData, CombinedTestData
from time import time
import sys
import os
import json


class QueryAttentionSumVec(object):
    def __init__(self, query_len, title_len, num_word, embed_size, gpu, load_path, max_grad_norm=5):
        self.queries = tf.placeholder(tf.int32, [None, query_len], name='query')
        self.titles = tf.placeholder(tf.int32, [None, title_len], name='titles')
        self.neg_titles = tf.placeholder(tf.int32, [None, title_len], name='neg_titles')
        self.margins = tf.placeholder(tf.float32, [None], name='margin')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.query_len = query_len
        self.title_len = title_len
        self.embed_size = embed_size

        with tf.variable_scope('lookup'):
            with tf.device("/gpu:%s" % gpu):

                initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)


                word_embedding = tf.get_variable('word_embedding', [num_word, embed_size], initializer=initializer)

                pos_title_word = tf.nn.embedding_lookup(word_embedding, self.titles)
                neg_title_word = tf.nn.embedding_lookup(word_embedding, self.neg_titles)
            with tf.device("/gpu:%s" % (gpu + 1)):
                query_embedding = tf.get_variable('query_embedding', [num_word, embed_size * 2],
                                                  initializer=initializer)
                query_word = tf.nn.embedding_lookup(query_embedding, self.queries)

            with tf.device("/gpu:%s" % (gpu + 2)):
                att_word_embedding = tf.get_variable('att_word_embedding', [num_word, embed_size], initializer=initializer)
                pos_att_title_word = tf.nn.embedding_lookup(att_word_embedding, self.titles)
                neg_att_title_word = tf.nn.embedding_lookup(att_word_embedding, self.neg_titles)
                att_query_word = tf.nn.embedding_lookup(att_word_embedding, self.queries)

        with tf.device("/gpu:%s" % (gpu)):
            with tf.name_scope('sumvec'):
                query_vec = tf.reduce_sum(query_word, 1)
                att_query_vec = tf.reduce_sum(att_query_word, 1)
                query_vec_expand = tf.reshape(att_query_vec, [-1, 1, embed_size])
                # [batch, title_len]
                pos_att_logit = tf.reduce_sum(tf.mul(query_vec_expand, pos_att_title_word), 2)
                neg_att_logit = tf.reduce_sum(tf.mul(query_vec_expand, neg_att_title_word), 2)
                pos_att = tf.nn.softmax(pos_att_logit)
                neg_att = tf.nn.softmax(neg_att_logit)

                _pos_att = tf.nn.softmax(-pos_att_logit)
                _neg_att = tf.nn.softmax(-neg_att_logit)

                pos_title_vec = tf.reduce_sum(tf.mul(tf.reshape(pos_att, [-1, title_len, 1]), pos_title_word), 1)
                neg_title_vec = tf.reduce_sum(tf.mul(tf.reshape(neg_att, [-1, title_len, 1]), neg_title_word), 1)
                _pos_title_vec = tf.reduce_sum(tf.mul(tf.reshape(_pos_att, [-1, title_len, 1]), pos_title_word), 1)
                _neg_title_vec = tf.reduce_sum(tf.mul(tf.reshape(_neg_att, [-1, title_len, 1]), neg_title_word), 1)


                # pos_title_vec = tf.reduce_sum(pos_title_word, 1)
                # neg_title_vec = tf.reduce_sum(neg_title_word, 1)

            with tf.name_scope("dropout"):
                h_query = tf.nn.dropout(query_vec, self.dropout_keep_prob)
                h_pos_drop = tf.nn.dropout(tf.concat(1, [pos_title_vec, _pos_title_vec]), self.dropout_keep_prob)
                h_neg_drop = tf.nn.dropout(tf.concat(1, [neg_title_vec, _neg_title_vec]), self.dropout_keep_prob)

            with tf.name_scope("loss"):
                self.pos_sim = self.sim(h_query, h_pos_drop)
                neg_sim = self.sim(h_query, h_neg_drop)
                self.loss = tf.reduce_sum(tf.maximum(0., neg_sim + self.margins - self.pos_sim))

            tvars = tf.trainable_variables()
            # self.grads = tf.gradients(self.loss, tvars)
            self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), max_grad_norm)
            optimizer = tf.train.AdamOptimizer(1e-3)
            self.train_op = optimizer.apply_gradients(zip(self.grads, tvars))
            # self.train_op = optimizer.minimize(self.loss)

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

def train(fn_train, fn_feature, fn_word,
          num_epoch, batch_size, dirname,
          embed_size, load, train_method, gpu, test_conf):

    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", dirname))
    print("Writing to {}".format(out_dir))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    save_path = os.path.join(checkpoint_dir, "model")
    dev_res_path = os.path.join(out_dir, 'dev.com.res')
    log_path = os.path.join(out_dir, 'train.log')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

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

    test_data = eval(test_conf['test_cls'])(test_conf['fn_test'], test_conf['fn_title'], word2idx, MAX_QUERY_LEN,
                                            MAX_TITLE_LEN, PAD)
    dataset = CombinedTrainData(fn_train, fn_feature, MAX_QUERY_LEN, MAX_TITLE_LEN, PAD, train_method)
    # dataset = TrainDataMemory(fn_train, fn_feature, MAX_QUERY_LEN, MAX_TITLE_LEN, WPAD, fn_gid_count, train_method)
    # test_data = TestData(fn_dev, fn_feature, MAX_QUERY_LEN, MAX_TITLE_LEN, WPAD)
    # test_data = ReadableTestData(fn_dev, fn_test_feature, word2idx, MAX_QUERY_LEN, MAX_TITLE_LEN, WPAD)
    # test_data = CombinedTestData(fn_dev, fn_test_feature, word2idx, MAX_QUERY_LEN, MAX_TITLE_LEN, WPAD)
    n_batch = dataset.num // batch_size + dataset.num % batch_size
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.allow_soft_placement = True
    # config.log_device_placement = True

    if load:
        load_path = save_path
    else:
        load_path = None

    fout_log = open(log_path, 'a')
    # with tf.Graph().as_default():
    #     session = tf.Session(config=config)
    #     with session.as_default():
    model = QueryAttentionSumVec(MAX_QUERY_LEN, MAX_TITLE_LEN, num_word, embed_size, gpu, load_path)

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

            queries, pos_titles, neg_titles, margins = dataset.next_batch(batch_size)

            loss_one = model.fit(queries, pos_titles, neg_titles, margins, 0.85)
            total_loss += loss_one
            if lno % 500000 == 0:
                print >> fout_log, '# %s: loss = %s, it costs %ss' % (epoch_index, total_loss, time() - tic)
                print '# %s: loss = %s, it costs %ss' % (epoch_index, total_loss, time() - tic)
                old_path = model.save('%s-%s' % (save_path, epoch_index))
                tic = time()
                acc = test_data.evaluate(model, idx2word,  dev_res_path)
                print >> fout_log, "acc = %s, it costs %ss" % (acc, time()-tic)
                print "acc = %s, it costs %ss" % (acc, time()-tic)
                if best_acc < acc:
                    best_acc = acc
                    os.rename(old_path, save_path)
                    os.rename('%s.meta' % old_path, '%s.meta' % save_path)
                    print >> fout_log, old_path
                    print "best mode", old_path
                total_loss = 0
                tic = time()
        print "finish one pass"
    fout_log.close()



def average_rank_model():
    fn_train = 'data/combine.query.v'
    # fn_w_dev = 'data/query.dev.v'
    fn_dev = 'data/cut.query.dat'
    # fn_w_test = 'data/query.testv'
    fn_feature = 'data/feature.v'
    fn_readable_feature = 'data/feature.dat'
    fn_word = 'data/word.list'
    fn_gid_count = 'data/gid.count.v'
    train(fn_train, fn_feature, fn_word,
          num_epoch=500, batch_size=500, dirname='com-sumvec-avg',
          embed_size=60, load=False, train_method=1, gpu=7,
          test_conf={'fn_test': 'data/cut.query.dat', 'fn_title': 'data/feature.dat', 'test_cls': 'ReadableTestData'})

def click_model():
    fn_train = 'data/com.query.click.v'
    fn_dev = 'data/cut.query.dat'

    fn_feature = 'data/feature.v'
    fn_readable_feature = 'data/feature.dat'
    fn_word = 'data/word.list'
    fn_gid_count = 'data/gid.count.v'
    train(fn_train, fn_feature, fn_word,
          num_epoch=500, batch_size=500, dirname='com-attention-click-mix',
          embed_size=50, load=True, train_method=2, gpu=0,
          test_conf={'fn_test': 'data/com.query.click.test.shuffle.dat', 'fn_title': 'data/title.test.txt',
                     'test_cls': 'CombinedTestData'}
          )


if __name__ == '__main__':
    # mini_model()
    # average_rank_model()

    click_model()
