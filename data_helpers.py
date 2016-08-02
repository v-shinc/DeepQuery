from time import time
import numpy as np
import random
import sys

import json
class PVTrainData(object):
    def __init__(self, fn_train, fn_feature, query_len, title_len, PAD, fn_sample):

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
        self.sampled = []
        self.sample(fn_sample)
        self.PAD = PAD

    def next_batch(self, batch_size):
        queries = []
        pos_titles = []
        neg_titles = []
        margins = []
        while len(queries) < batch_size * 3:
            if self.index == self.num:
                self.file.seek(0)
                self.index = 0
            line = self.file.readline()
            self.index += 1
            ll = line.strip().split('\t')
            if len(ll) != 3:
                continue
            q, strong_clicks, weak_clicks = ll

            q = map(int, q.split())[:self.query_len]
            strong_clicks = map(int, strong_clicks.split())
            weak_clicks = map(int, weak_clicks.split())

            strong_clicks = [c for c in strong_clicks if c in self.feature]
            weak_clicks = [c for c in weak_clicks if c in self.feature]
            if len(strong_clicks) == 0 or len(weak_clicks) == 0:
                continue
            idx = np.random.randint(len(strong_clicks))
            strong_gid = strong_clicks[idx]
            idx = np.random.randint(len(weak_clicks))
            weak_gid = weak_clicks[idx]


            clicks_set = set(strong_clicks)
            clicks_set.update(weak_clicks)
            idx = np.random.randint(len(self.sampled))
            negative_gid = self.sampled[idx]
            while negative_gid in clicks_set:
                idx = np.random.randint(len(self.sampled))
                negative_gid = self.sampled[idx]

            # strong vs weak
            queries.append(q)
            pos_titles.append(self.feature[strong_gid])
            neg_titles.append(self.feature[weak_gid])
            margins.append(0.3)

            # weak vs negative
            queries.append(q)
            pos_titles.append(self.feature[weak_gid])
            neg_titles.append(self.feature[negative_gid])
            margins.append(0.3)

            # strong vs negative
            queries.append(q)
            pos_titles.append(self.feature[strong_gid])
            neg_titles.append(self.feature[negative_gid])
            margins.append(0.3)

        for j, t in enumerate(pos_titles):
            l = len(t)
            pos_titles[j] += (self.title_len - l) * [self.PAD]

        for j, t in enumerate(neg_titles):
            l = len(t)
            neg_titles[j] += (self.title_len - l) * [self.PAD]

        for j, q in enumerate(queries):
            l = len(q)
            queries[j] += (self.query_len - l) * [self.PAD]
        return [np.array(e) for e in [queries, pos_titles, neg_titles, margins]]

    def sample(self, fn_sample):
        counter = dict()
        with open(fn_sample) as fin:
            for line in fin:
                gid, cnt = line.strip().split('\t')
                gid = int(gid)
                if gid not in self.feature:
                    continue
                counter[gid] = int(cnt)

        sum = 0
        size = 1 << 24
        for _, cnt in counter.iteritems():
            sum += cnt ** 0.75

        it = counter.iteritems()
        cur, cnt = it.next()
        prob = cnt ** 0.75 / sum
        self.sampled.append(cur)
        try:
            for i in xrange(size):
                if (i + 1.0) / (size + 0.0) > prob:
                    cur, cnt = it.next()
                    prob += cnt ** 0.75 / sum
                self.sampled.append(cur)
        except StopIteration:
            print '# sampled', len(self.sampled)

#TODO: n_repeat
class TrainDataMemory(object):
    def __init__(self, fn_train, fn_title, query_len, title_len, PAD, fn_sample, method):
        tic = time()
        self.feature = dict()
        with open(fn_title) as fin:
            for line in fin:
                gid, widx = line.strip().split('\t')
                gid = int(gid)
                widx = [int(i) for i in widx.split()][:title_len]
                self.feature[gid] = widx
        print "It took %ss to load %s titles" % (time() - tic, len(self.feature))

        tic = time()
        self.ranks = []
        self.queries = []
        if method == 1:
            with open(fn_train) as fin:
                for line in fin:
                    q, ranks = line.strip().split('\t')
                    q = map(int, q.split())[:query_len]
                    ranks = ranks.split()
                    titles = map(int, ranks[::2])
                    avg_rank = map(float, ranks[1::2])
                    clicks_rank = [(c, r) for c, r in zip(titles, avg_rank) if c in self.feature][:10]
                    if len(clicks_rank) < 2:
                        continue
                    self.queries.append(q)
                    self.ranks.append(clicks_rank)
        elif method == 2:
            with open(fn_train) as fin:
                for line in fin:
                    q, ranks = line.strip().split('\t')
                    q = map(int, q.split())[:query_len]
                    ranks = map(int, ranks.split())
                    titles = ranks[::2]
                    num_clicks = ranks[1::2]
                    n_pos = sum([1 for c in num_clicks if c > 0])
                    if n_pos == 0:
                        continue
                    clicks_rank = [(c, r) for c, r in zip(titles, num_clicks) if c in self.feature][:max(10, n_pos)]
                    if len(clicks_rank) < 2:
                        continue
                    self.queries.append(q)
                    self.ranks.append(clicks_rank)
        else:
            raise ValueError('method should be 1 or 2')

        print "It took %ss to load %s queries" % (time() - tic, len(self.queries))
        self.num = len(self.queries)
        self.title_len = title_len
        self.query_len = query_len
        self.index = 0
        self.sampled = self.feature.keys()
        self.PAD = PAD
        self.method = method

    def next_batch(self, batch_size):
        if self.method == 1:
            return self.next_batch_1(batch_size)
        elif self.method == 2:
            return self.next_batch_2(batch_size)
        else:
            raise ValueError('method = 1 or 2')

    def next_batch_1(self, batch_size):
        queries = []
        pos_titles = []
        neg_titles = []
        margins = []
        while len(queries) < batch_size * 9:
            if self.index == self.num:
                self.index = 0
            q = self.queries[self.index]
            rank = self.ranks[self.index]
            clicks_set = set([c for c, _ in rank])
            self.index += 1
            for _ in range(3):
                g1, g2 = random.sample(rank, 2)
                # n_try = 3
                # while g1[1] == g2[1] and n_try:
                #     g1, g2 = random.sample(clicks_rank, 2)
                #     n_try -= 1
                #
                # if n_try == 0:
                #     continue
                if g1[1] < g2[1]:
                    strong_gid = g1[0]
                    weak_gid = g2[0]
                else:
                    strong_gid = g2[0]
                    weak_gid = g1[0]

                idx = np.random.randint(len(self.sampled))
                negative_gid = self.sampled[idx]
                while negative_gid in clicks_set:
                    idx = np.random.randint(len(self.sampled))
                    negative_gid = self.sampled[idx]

                # strong vs weak
                queries.append(q)
                pos_titles.append(self.feature[strong_gid])
                neg_titles.append(self.feature[weak_gid])
                margins.append(0.1)

                # weak vs negative
                queries.append(q)
                pos_titles.append(self.feature[weak_gid])
                neg_titles.append(self.feature[negative_gid])
                margins.append(0.1)

                # strong vs negative
                queries.append(q)
                pos_titles.append(self.feature[strong_gid])
                neg_titles.append(self.feature[negative_gid])
                margins.append(0.1)


        for j, t in enumerate(pos_titles):
            l = len(t)
            pos_titles[j] += (self.title_len - l) * [self.PAD]

        for j, t in enumerate(neg_titles):
            l = len(t)
            neg_titles[j] += (self.title_len - l) * [self.PAD]

        for j, q in enumerate(queries):
            l = len(q)
            queries[j] += (self.query_len - l) * [self.PAD]
        return [np.array(e) for e in [queries, pos_titles, neg_titles, margins]]

    def next_batch_2(self, batch_size):
        # by click
        queries = []
        pos_titles = []
        neg_titles = []
        margins = []
        while len(queries) < batch_size * 9:
            if self.index == self.num:
                self.index = 0
            rank = self.ranks[self.index]
            q = self.queries[self.index]
            clicks_set = set([c for c, _ in rank])
            self.index += 1
            for _ in range(3):
                g1, g2 = random.sample(rank, 2)

                idx = np.random.randint(len(self.sampled))
                negative_gid = self.sampled[idx]
                while negative_gid in clicks_set:
                    idx = np.random.randint(len(self.sampled))
                    negative_gid = self.sampled[idx]

                if g1[1] != g2[1]:
                    if g1[1] > g2[1]:
                        strong_gid = g1[0]
                        weak_gid = g2[0]
                    else:
                        strong_gid = g2[0]
                        weak_gid = g1[0]
                    # strong vs weak
                    queries.append(q)
                    pos_titles.append(self.feature[strong_gid])
                    neg_titles.append(self.feature[weak_gid])
                    margins.append(0.1)

                # show vs not_show
                queries.append(q)
                pos_titles.append(self.feature[g1[0]])
                neg_titles.append(self.feature[negative_gid])
                margins.append(0.1)

                # show vs not_show
                queries.append(q)
                pos_titles.append(self.feature[g2[0]])
                neg_titles.append(self.feature[negative_gid])
                margins.append(0.1)


        for j, t in enumerate(pos_titles):
            l = len(t)
            pos_titles[j] += (self.title_len - l) * [self.PAD]

        for j, t in enumerate(neg_titles):
            l = len(t)
            neg_titles[j] += (self.title_len - l) * [self.PAD]

        for j, q in enumerate(queries):
            l = len(q)
            queries[j] += (self.query_len - l) * [self.PAD]
        return [np.array(e) for e in [queries, pos_titles, neg_titles, margins]]


class CombinedTrainData(object):
    def __init__(self, fn_train, fn_feature, query_len, title_len, PAD, method):

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
        # self.sampled = []
        # self.sample(fn_sample)
        # self.sample(fn_train)
        self.PAD = PAD
        self.method = method

    def next_batch(self, batch_size, n_repeat=3):
        if self.method == 1:
            return self.next_batch_1(batch_size, n_repeat)
        elif self.method == 2:
            return self.next_batch_2(batch_size, n_repeat)
        else:
            raise ValueError('method = 1 or 2')

    def next_batch_2(self, batch_size, n_repeat=3):
        # by click
        queries = []
        pos_titles = []
        neg_titles = []
        margins = []
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
                    pos_titles.append(self.feature[strong_gid])
                    neg_titles.append(self.feature[weak_gid])
                    margins.append(0.1)

                # show vs not_show
                g1, g2 = random.sample(clicks_rank, 2)
                idx = np.random.randint(len(self.sampled))
                negative_gid = self.sampled[idx]
                while negative_gid in clicks_set:
                    idx = np.random.randint(len(self.sampled))
                    negative_gid = self.sampled[idx]

                queries.append(q)
                pos_titles.append(self.feature[g1[0]])
                neg_titles.append(self.feature[negative_gid])
                margins.append(0.1)

                queries.append(q)
                pos_titles.append(self.feature[g2[0]])
                neg_titles.append(self.feature[negative_gid])
                margins.append(0.1)

        for j, t in enumerate(pos_titles):
            l = len(t)
            pos_titles[j] += (self.title_len - l) * [self.PAD]

        for j, t in enumerate(neg_titles):
            l = len(t)
            neg_titles[j] += (self.title_len - l) * [self.PAD]

        for j, q in enumerate(queries):
            l = len(q)
            queries[j] += (self.query_len - l) * [self.PAD]
        return [np.array(e) for e in [queries, pos_titles, neg_titles, margins]]

    def next_batch_1(self, batch_size, n_repeat):
        queries = []
        pos_titles = []
        neg_titles = []
        margins = []
        chunk_size = batch_size * n_repeat * 3
        while len(queries) < chunk_size:
            if self.index == self.num:
                self.file.seek(0)
                self.index = 0
            line = self.file.readline()
            self.index += 1
            q, titles = line.strip().split('\t')
            q = map(int, q.split())[:self.query_len]
            titles = titles.split()
            try:
                clicks = map(int, titles[::2])
            except ValueError as e:
                print e
                print titles
                print titles[::2]
                raise ValueError
            ranks = map(float, titles[1::2])

            clicks_rank = [(c, r) for c, r in zip(clicks, ranks) if c in self.feature]
            if len(clicks_rank) < 2:
                continue
            clicks_set = set(clicks)
            for _ in range(n_repeat):
                g1, g2 = random.sample(clicks_rank, 2)
                # n_try = 3
                # while g1[1] == g2[1] and n_try:
                #     g1, g2 = random.sample(clicks_rank, 2)
                #     n_try -= 1
                #
                # if n_try == 0:
                #     continue
                if g1[1] < g2[1]:
                    strong_gid = g1[0]
                    weak_gid = g2[0]
                else:
                    strong_gid = g2[0]
                    weak_gid = g1[0]

                idx = np.random.randint(len(self.sampled))
                negative_gid = self.sampled[idx]
                while negative_gid in clicks_set:
                    idx = np.random.randint(len(self.sampled))
                    negative_gid = self.sampled[idx]

                # strong vs weak
                queries.append(q)
                pos_titles.append(self.feature[strong_gid])
                neg_titles.append(self.feature[weak_gid])
                margins.append(0.1)

                # weak vs negative
                queries.append(q)
                pos_titles.append(self.feature[weak_gid])
                neg_titles.append(self.feature[negative_gid])
                margins.append(0.1)

                # strong vs negative
                queries.append(q)
                pos_titles.append(self.feature[strong_gid])
                neg_titles.append(self.feature[negative_gid])
                margins.append(0.1)

        for j, t in enumerate(pos_titles):
            l = len(t)
            pos_titles[j] += (self.title_len - l) * [self.PAD]

        for j, t in enumerate(neg_titles):
            l = len(t)
            neg_titles[j] += (self.title_len - l) * [self.PAD]

        for j, q in enumerate(queries):
            l = len(q)
            queries[j] += (self.query_len - l) * [self.PAD]
        return [np.array(e) for e in [queries, pos_titles, neg_titles, margins]]

# PV Test
class ReadableTestData(object):
    def __init__(self, fn_test, fn_feature, word2idx, query_len, title_len, PAD):
        self.num = 0

        # with open(fn_test) as fin:
        #     for _ in fin:
        #         self.num += 1
        self.feature = dict()
        with open(fn_feature) as fin:
            for line in fin:
                data = json.loads(line, encoding='utf8')
                gid = data['id']
                widx = [word2idx[w] for w in data['cut_title'].split() if w in word2idx][:title_len]
                if len(widx) <= 2:
                    continue
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
            q = [self.word2idx[w] for w in data['cut_query'].split() if w in self.word2idx][:self.query_len]
            if len(q) <= 2:
                continue
            lno += 1
            strong_clicks = data['positive']
            weak_clicks = data['negative']

            strong_clicks = [c for c in strong_clicks if c in self.feature]
            weak_clicks = [c for c in weak_clicks if c in self.feature]
            if len(strong_clicks) == 0 or len(weak_clicks) == 0:
                continue

            titles = []
            for c in strong_clicks:
                t = self.feature[c]
                l = len(t)
                t += (self.title_len - l) * [self.PAD]
                titles.append(t)
            n_pos = len(titles)
            for c in weak_clicks:
                t = self.feature[c]
                l = len(t)
                t += (self.title_len - l) * [self.PAD]
                titles.append(t)
            q += (self.query_len - len(q)) * [self.PAD]
            yield np.array([q] * len(titles)), np.array(titles), n_pos

    def evaluate(self, model, idx2word, fn_res, n_limit=5000):
        avg_acc = 0.
        num = 0
        self.reset()
        lno = 0
        def indices2sent(l):
            return ''.join([idx2word[int(i)] for i in l])
        with open(fn_res, 'w') as fout:
            for q, t, n_pos in self(n_limit):
                lno += 1
                if lno % 100 == 0:
                    sys.stdout.write('process to %d\r' % lno)
                    sys.stdout.flush()
                n_neg = len(t) - n_pos
                scores = model.predict(q, t)
                n_acc = 0
                for i in xrange(n_pos):
                    for j in xrange(n_pos, len(t)):
                        if scores[i] > scores[j]:
                            n_acc += 1
                avg_acc += n_acc * 1.0 / (n_pos * n_neg)
                num += 1

                pos_title_scores = zip(t[:n_pos, :], scores[:n_pos])
                neg_title_scores = zip(t[n_pos:, :], scores[n_pos:])
                # pos_title_scores = sorted(pos_title_scores, key=lambda x: x[1], reverse=True)
                # neg_title_scores = sorted(neg_title_scores, key=lambda x: x[1], reverse=True)

                # res = {}
                # res['q'] = indices2sent(q[0])
                # res['pos'] = []
                # res['neg'] = []
                # for ti, s in pos_title_scores:
                #     res['pos'].append(indices2sent(ti) + ':' + str(s))
                # for ti, s in neg_title_scores:
                #     res['neg'].append(indices2sent(ti) + ':' + str(s))
                # print >> fout, json.dumps(res, ensure_ascii=False).encode('utf8')

                rank = [('$' + indices2sent(ti), si) for ti, si in pos_title_scores]
                rank += [(indices2sent(ti), si) for ti, si in neg_title_scores]
                rank = sorted(rank, key=lambda x: x[1], reverse=True)
                rank = [ti + ':' + str(si) for ti, si in rank]
                print >> fout, ('%s\t%s' % (indices2sent(q[0]), '\t'.join(rank))).encode('utf8')

            return avg_acc / num

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

class TestData(object):
    def __init__(self, fn_test, fn_feature, query_len, title_len, PAD):

        self.num = 0
        with open(fn_test) as fin:
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
        self.file = open(fn_test)
        self.PAD = PAD

    def reset(self):
        self.file.seek(0)

    def __call__(self, n_limit=5000):
        lno = 0
        for line in self.file:
            if lno == n_limit:
                break
            lno += 1
            ll = line.strip().split('\t')
            if len(ll) != 3:
                continue
            q, strong_clicks, weak_clicks = ll

            q = map(int, q.split())[:self.query_len]
            strong_clicks = map(int, strong_clicks.split())
            weak_clicks = map(int, weak_clicks.split())

            strong_clicks = [c for c in strong_clicks if c in self.feature]
            weak_clicks = [c for c in weak_clicks if c in self.feature]
            if len(strong_clicks) == 0 or len(weak_clicks) == 0:
                continue

            titles = []
            for c in strong_clicks:
                t = self.feature[c]
                l = len(t)
                t += (self.title_len - l) * [self.PAD]
                titles.append(t)
            n_pos = len(titles)
            for c in weak_clicks:
                t = self.feature[c]
                l = len(t)
                t += (self.title_len - l) * [self.PAD]
                titles.append(t)
            q += (self.query_len - len(q)) * [self.PAD]
            yield np.array([q] * len(titles)), np.array(titles), n_pos

def load_word2vec(fn_idx2word, n_word, dim):
    embedding = np.random.normal(0, 0.1, (n_word, dim))
    with open(fn_idx2word) as fin:
        for line in fin:
            idx, vec = line.strip().split('\t')
            vec = map(float, vec.split())
            embedding[int(idx), :] = vec
        embedding[-1, :] = np.zeros(dim)
    return embedding.astype(np.float32)