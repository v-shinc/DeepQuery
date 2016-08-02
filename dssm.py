#encoding=utf8
import tensorflow as tf
import random
import numpy as np
from time import time
import sys
import os
import json

class DSSM(object):
    def __init__(self, query_len, title_len, num_word, embed_size, l1, l2, l3, num_class, load_path=None, gamma=5, l2_reg_lambda=0.1, max_grad_norm=5):
        self.queries = tf.placeholder(tf.int32, [None, query_len], name='queries')
        self.titles = tf.placeholder(tf.int32, [None, num_class, title_len], name='titles')
        self.labels = tf.placeholder(tf.int64, [None], name='labels')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.config = locals()
        self.config.pop('self')

        with tf.device("/gpu:3"):
            with tf.variable_scope('embedding'):
                initializer = tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf.float32)

                embeding = tf.get_variable('embedding', [num_word, embed_size], initializer=initializer)
                query_word = tf.nn.embedding_lookup(embeding, self.queries)
                title_word = tf.nn.embedding_lookup(embeding, self.titles)

            with tf.name_scope('pooling'):
                query_sum = tf.reduce_sum(query_word, 1)
                title_sum = tf.reduce_sum(title_word, 2)

                l2_loss = tf.constant(0.0)
                title_flat = tf.reshape(title_sum, [-1, embed_size])

            with tf.name_scope('multi-layers'):
                self.query_vec = tf.reshape(self.dnn(query_sum, l1, l2, l3, False), [-1, 1, embed_size])
                self.title_vec = tf.reshape(self.dnn(title_flat, l1, l2, l3, True), [-1, num_class, embed_size])

            with tf.name_scope("dropout"):
                query_drop = tf.nn.dropout(self.queries, self.dropout_keep_prob)
                title_drop = tf.nn.dropout(self.titles, self.dropout_keep_prob)

            with tf.name_scope("cosine_layer"):
                self.dot = tf.reduce_sum(tf.mul(query_drop, title_drop), 2)
                sqrt_title = tf.sqrt(tf.reduce_sum(title_drop**2, 2))
                sqrt_query = tf.sqrt(tf.reduce_sum(query_drop**2, 2))
                epsilon = 1e-5
                self.cosine = tf.maximum(self.dot / (tf.maximum(sqrt_title * sqrt_query, epsilon)), epsilon)

            with tf.name_scope("loss"):
                self.l1 = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(gamma * self.cosine, self.labels))
                self.l2 = l2_reg_lambda * l2_loss
                self.loss = self.l1

            tvars = tf.trainable_variables()
            # self.grads = tf.gradients(self.loss, tvars)
            self.grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), max_grad_norm)
            optimizer = tf.train.AdamOptimizer(1e-3)
            self.train_op = optimizer.apply_gradients(zip(self.grads, tvars))

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

    def dnn(self, input, l1, l2, l3, reuse):
        initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
        with tf.variable_scope('fc_layers', reuse=reuse):
            w1 = tf.get_variable('w1', [tf.shape(input)[1], l1], initializer=initializer) * 0.1
            b1 = tf.get_variable('b1', [l1], initializer=tf.constant_initializer(0))
            h1 = tf.nn.relu(tf.add(tf.matmul(input, w1), b1))

            w2 = tf.get_variable('w2', [l1, l2], initializer=initializer) * 0.1
            b2 = tf.get_variable('b2', [l2], initializer=tf.constant_initializer(0))
            h2 = tf.nn.relu(tf.add(tf.matmul(h1, w2), b2))

            w3 = tf.get_variable('w3', [l2, l3], initializer=initializer) * 0.1
            b3 = tf.get_variable('b3', [l3], initializer=tf.constant_initializer(0))
            h3 = tf.nn.relu(tf.add(tf.matmul(h2, w3), b3))
        return h3

    def load_from_config(self, fn_conf):
        config = json.load(fn_conf, encoding='utf8')
        return DSSM(**config)

    def fit(self, queries, titles, labels, dropout_keep_prob):
        feed_dict = {
            self.queries: queries,
            self.titles: titles,
            self.labels: labels,
            self.dropout_keep_prob: dropout_keep_prob
        }
        _, loss_one = self.session.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss_one

    def predict(self, query, titles):
        scores = self.session.run(
            self.cosine,
            {self.queries: query, self.titles: titles, self.dropout_keep_prob: 1})

        return [float(s) for s in scores]

    def save(self, save_path):
        self.config['load_path'] = save_path
        json.dump(self.config, open(os.path.join(os.path.dirname(save_path), 'conf.json'), 'w'))
        return self.saver.save(self.session, save_path)

    def get_query_vec(self, queries):
        return self.session.run(self.query_vec, {self.queries: queries})[:, 0, :]

    def get_title_vec(self, titles):
        return self.session.run(self.title_vec, {self.titles: titles})[:, 0, :]


class CombinedTrainData(object):
    def __init__(self, fn_train, fn_feature, query_len, title_len, PAD):

        self.num = 0
        with open(fn_train) as fin:
            for _ in fin:
                self.num += 1

        self.feature = dict()
        with open(fn_feature) as fin:
            for line in fin:
                gid, widx = line.strip().split('\t')
                gid = int(gid)
                widx = [int(i) for i in widx.split()][:title_len]
                self.feature[gid] = widx

        self.title_len = title_len
        self.query_len = query_len
        self.index = 0
        self.file = open(fn_train)

        # TODO: CHANGE SAMPLE METHOD
        self.sampled = self.feature.keys()
        self.PAD = PAD



    def next_batch(self, batch_size, n_repeat=3):
        # by click
        queries = []
        multi_titles = []
        labels = []
        chunk_size = batch_size * n_repeat * 3
        while len(queries) < chunk_size:
            if self.index == self.num:
                self.file.seek(0)
                self.index = 0
            line = self.file.readline()
            self.index += 1
            q, titles = line.strip().split('\t')

            q = map(int, q.split())[:self.query_len]
            titles = map(int, titles.split())
            clicks = titles[::2]
            num_clicks = titles[1::2]

            n_pos = sum([1 for c in num_clicks if c > 0])
            if n_pos == 0:
                continue

            clicks_rank = [(c, r) for c, r in zip(clicks, num_clicks) if c in self.feature]
            if len(clicks_rank) < 2:
                continue
            clicks_set = set(clicks)
            for _ in range(n_repeat):
                g1, g2 = random.sample(clicks_rank, 2)
                if g1[1] != g2[1]:
                    if g1[1] > g2[1]:
                        strong_gid = g1[0]
                        weak_gid = g2[0]
                    else:
                        strong_gid = g2[0]
                        weak_gid = g1[0]
                    # strong vs weak
                    queries.append(q)
                    multi_titles.append([self.feature[strong_gid], self.feature[weak_gid]])
                    labels.append([1, 0])

                # show vs not_show
                g1, g2 = random.sample(clicks_rank, 2)
                idx = np.random.randint(len(self.sampled))
                negative_gid = self.sampled[idx]
                while negative_gid in clicks_set:
                    idx = np.random.randint(len(self.sampled))
                    negative_gid = self.sampled[idx]

                queries.append(q)
                multi_titles.append([self.feature[g1[0]], self.feature[negative_gid]])
                labels.append([1, 0])

                queries.append(q)
                multi_titles.append([self.feature[g2[0]], self.feature[negative_gid]])
                labels.append([1, 0])

        for j, ts in enumerate(multi_titles):
            for k, t in enumerate(ts):
                l = len(t)
                multi_titles[j][k] += (self.title_len - l) * [self.PAD]



        for j, q in enumerate(queries):
            l = len(q)
            queries[j] += (self.query_len - l) * [self.PAD]
        return [np.array(e) for e in [queries, multi_titles, labels]]


class CombinedTestData(object):
    def __init__(self, fn_test, fn_feature, word2idx, query_len, title_len, PAD):
        self.num = 0

        # with open(fn_test) as fin:
        #     for _ in fin:
        #         self.num += 1
        self.feature = dict()
        with open(fn_feature) as fin:
            for line in fin:
                ll = line.decode('utf8').strip().split('\t')
                if len(ll) != 2:
                    continue
                gid = int(ll[0])
                widx = [word2idx[w] for w in ll[1] if w in word2idx]
                self.feature[gid] = widx
        self.word2idx = word2idx
        self.title_len = title_len
        self.query_len = query_len
        self.file = open(fn_test)
        self.PAD = PAD

    def reset(self):
        self.file.seek(0)


    def __call__(self, n_limit=5000):
        lno = 0
        for line in self.file:
            if lno == n_limit:
                break
            data = json.loads(line, encoding='utf8')

            q = [self.word2idx[w] for w in data['query'].split() if w in self.word2idx][:self.query_len]
            if len(q) <= 2:
                continue

            gids = []
            num_clicks = []
            for gid, c in data['title']:
                if gid not in self.feature:
                    continue
                gids.append(gid)
                num_clicks.append(c)

            titles = []
            for c in gids:
                t = self.feature[c][:self.title_len]
                l = len(t)
                t += (self.title_len - l) * [self.PAD]
                titles.append(t)

            q += (self.query_len - len(q)) * [self.PAD]
            if len(titles) <= 1:
                continue
            lno += 1
            yield np.array([q] * len(titles)), np.array(titles), num_clicks

    # TODO: remove idx2word
    def evaluate(self, model, idx2word, fn_res, n_limit=5000):
        avg_acc = 0.
        num = 0
        self.reset()
        lno = 0

        def indices2sent(l):
            return ''.join([idx2word[int(i)] for i in l])
        with open(fn_res, 'w') as fout:
            for q, t, num_clicks in self(n_limit):

                if lno % 100 == 0:
                    sys.stdout.write('process to %d\r' % lno)
                    sys.stdout.flush()
                scores = model.predict(q, t)
                n_acc = 0
                n_titles = len(t)
                n_pair = 0
                for i in xrange(n_titles):
                    for j in xrange(i+1, n_titles):
                        if num_clicks[i] == num_clicks[j]:
                            continue
                        if (scores[i] - scores[j]) * (num_clicks[i] - num_clicks[j]) > 0:
                            n_acc += 1
                        n_pair += 1
                if n_pair == 0:
                    # print indices2sent(q[0])
                    # print t, num_clicks
                    continue
                avg_acc += n_acc * 1.0 / n_pair
                num += 1
                scores = list(scores)
                title_scores = sorted(zip(t, num_clicks, scores), key=lambda x: x[2], reverse=True)

                res = {}
                res['q'] = indices2sent(q[0])
                res['titles'] = []
                lno += 1
                for ti, c, s in title_scores:
                    res['titles'].append(indices2sent(ti) + ':' + str(c)+ ':'+str(s))

                # print >> fout, json.dumps(res, ensure_ascii=False).encode('utf8')
                print >> fout, ('%s\t%s' % (res['q'], '\t'.join(res['titles']))).encode('utf8')
            return avg_acc / num

MAX_QUERY_LEN = 8
MAX_TITLE_LEN = 20

def train(fn_train, fn_feature, fn_word,
          num_epoch, batch_size, dirname,
          embed_size, hidden_size1, hidden_size2, hidden_size3, load, test_conf, n_repeat=1):

    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", dirname))
    print("Writing to {}".format(out_dir))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    save_path = os.path.join(checkpoint_dir, "model")
    dev_res_path = os.path.join(out_dir, 'dev.res')
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
    dataset = CombinedTrainData(fn_train, fn_feature, MAX_QUERY_LEN, MAX_TITLE_LEN, PAD)

    n_batch = dataset.num // batch_size + dataset.num % batch_size

    if load:
        load_path = save_path
    else:
        load_path = None
    fout_log = open(log_path, 'a')

    model = DSSM(MAX_QUERY_LEN, MAX_TITLE_LEN, num_word, embed_size, hidden_size1, hidden_size2, hidden_size3, num_class=2, load_path=load_path)

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

            queries, titles, labels = dataset.next_batch(batch_size, n_repeat)
            loss_one = model.fit(queries, titles, labels, 0.5)
            total_loss += loss_one
            if lno % 500000 == 0:
                print >> fout_log, '# %s: loss = %s, it costs %ss' % (epoch_index, total_loss, time() - tic)
                print '# %s: loss = %s, it costs %ss' % (epoch_index, total_loss, time() - tic)
                old_path = model.save('%s-%s' % (save_path, epoch_index))

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

if __name__ == '__main__':
    fn_train = 'data/com.query.click.20150924.20160630.v'
    fn_title = 'data/title.20150924.20160630.v'
    fn_word = 'data/word.list'
    fn_dev = 'data/com.query.click.test.shuffle.dat'
    fn_dev_title = 'data/title.test.txt'
    train(fn_train, fn_title, fn_word,
          num_epoch=200, batch_size=300, dirname='dssm',
          embed_size=100, hidden_size1=100, hidden_size2=100, hidden_size3=100, load=False, test_conf={'fn_test': fn_dev, 'fn_title': fn_dev_title,
                     'test_cls': 'CombinedTestData'}, n_repeat=1)