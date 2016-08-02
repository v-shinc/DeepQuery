#coding=utf8
import sys
sys.path.insert(0, '..')
import numpy as np
import json
from itertools import izip

from annoy import AnnoyIndex
from com_baidu_dnn import QueryDNN
from conf import text_cutter
MAX_QUERY_LEN = 8
MAX_TITLE_LEN = 20

class QueryModel(object):
    def __init__(self, fn_word, model_name, model_path):
        print 'Initialize Vectorizer'
        # load word
        # idx2word = dict()
        word2idx = dict()
        num_word = 0
        with open(fn_word) as fin:
            for line in fin:
                w = line.decode('utf8').strip()
                # idx2word[num_word] = w
                word2idx[w] = num_word
                num_word += 1
        PAD = num_word
        num_word += 1
        # idx2word[PAD] = ''

        self.word2idx = word2idx
        self.PAD = PAD


        if model_name == 'QueryDNN':

            self.model = QueryDNN(MAX_QUERY_LEN, MAX_TITLE_LEN, num_word, embed_size=100, hidden_size1=100,
                         hidden_size2=50, gpu=4, load_path=model_path)
        elif model_name == 'QueryRNNPair':
            pass
        else:
            raise ValueError('Model must be ...')
        self.dim = self.model.embed_size

    def rank_titles(self, query, titles):
        query_index = [self.word2idx[w] for w in query.split() if w in self.word2idx][:MAX_QUERY_LEN]
        if len(query_index) == 0:
            return None, None
        query_index += (MAX_QUERY_LEN - len(query_index)) * [self.PAD]
        titles_index = []
        valid_titles = []
        for s in titles:
            si = [self.word2idx[w] for w in s.split() if w in self.word2idx][:MAX_TITLE_LEN]
            l = len(si)
            if l == 0:
                continue
            si += (MAX_TITLE_LEN - l) * [self.PAD]
            titles_index.append(si)
            valid_titles.append(s)

        if len(valid_titles) == 0:
            return None, None
        scores = self.model.predict([query_index] * len(valid_titles), titles_index)
        ranks = sorted(zip(valid_titles, scores), key=lambda x:x[1], reverse=True)
        return ranks


    def get_query_vec(self, queries):
        sentence_index = []
        valid_sentence = []
        for s in queries:
            si = [self.word2idx[w] for w in s.split() if w in self.word2idx][:MAX_QUERY_LEN]
            l = len(si)
            if l == 0:
                continue
            si += (MAX_QUERY_LEN - l) * [self.PAD]
            sentence_index.append(si)
            valid_sentence.append(s)
        if len(valid_sentence) == 0:
            return None, []
        return self.model.get_query_vec(np.array(sentence_index)), valid_sentence

    def get_title_vec(self, titles):
        sentence_index = []
        valid_sentence = []
        for s in titles:
            si = [self.word2idx[w] for w in s.split() if w in self.word2idx][:MAX_TITLE_LEN]
            l = len(si)
            if l == 0:
                continue
            si += (MAX_TITLE_LEN - l) * [self.PAD]
            sentence_index.append(si)
            valid_sentence.append(s)
        if len(valid_sentence) == 0:
            return None, []
        return self.model.get_title_vec(np.array(sentence_index)), valid_sentence


class NearSentence(object):
    def __init__(self, fn_word, model_name, model_path):
        self.model = QueryModel(fn_word, model_name, model_path)
        self.queries = []
        self.titles = []

        self.query_index = 0
        self.title_index = 0
        self.query_ann = AnnoyIndex(self.model.dim, metric='euclidean')
        self.title_ann = AnnoyIndex(self.model.dim, metric='euclidean')

    def load_queries(self, fn_query, column):
        print '[In load_queries] Load candidate queries'
        sentences = []
        chunk = []

        vecs = []
        with open(fn_query) as fin:
            for line in fin:
                ll = line.decode('utf8').strip().split('\t')
                if len(ll) < column:
                    continue
                chunk.append(ll[column - 1])
                if len(chunk) == 1000:
                    vec, valid_sentence = self.model.get_query_vec(chunk)
                    vec = vec / np.sqrt(np.sum(vec**2, 1, keepdims=True))
                    vecs.extend(list(vec))
                    sentences.extend(valid_sentence)
                    chunk = []
        if len(chunk) > 0:
            vec, valid_sentence = self.model.get_query_vec(chunk)
            vecs.extend(list(vec))
            sentences.extend(valid_sentence)

        print '[In load_queries] Build query annoy tree'
        for s, v in izip(sentences, vecs):
            self.queries.append(s)
            # if vecs == [0] * self.vectorizer.dim:
            #     continue
            self.query_ann.add_item(self.query_index, v)
            self.query_index += 1

        self.query_ann.build(10)
        print '[In load_queries] Size of tree =', self.query_ann.get_n_items()

    def load_titles(self, fn_title, column):
        print '[In load_titles] Load candidate titles'
        sentences = []

        chunk = []
        vecs = []
        with open(fn_title) as fin:
            for line in fin:
                ll = line.decode('utf8').strip().split('\t')
                if len(ll) < column:
                    continue
                chunk.append(ll[column - 1])
                if len(chunk) == 1000:
                    vec, valid_sentence = self.model.get_title_vec(chunk)
                    vec = vec / np.sqrt(np.sum(vec ** 2, 1, keepdims=True))
                    vecs.extend(list(vec))
                    sentences.extend(valid_sentence)
                    chunk = []
            if len(chunk) > 0:
                vec, valid_sentence = self.model.get_title_vec(chunk)
                vec = vec / np.sqrt(np.sum(vec ** 2, 1, keepdims=True))
                vecs.extend(list(vec))
                sentences.extend(valid_sentence)

        print '[In load_titles] Build titles annoy tree, size =', len(vecs)

        for s, v in izip(sentences, vecs):
            self.titles.append(s)
            self.title_ann.add_item(self.title_index, v)     # v is a list
            self.title_index += 1
        self.title_ann.build(10)
        print '[In load_titles] Size of tree =', self.title_ann.get_n_items()



    def get_k_nearest_query(self, query, k):

        if isinstance(query, unicode):
            query = query.encode('utf8')

        cut_data = text_cutter.process({'title': query})
        cut_query = cut_data['cut_title'].decode('utf8')
        vecs, valid_queries= self.model.get_query_vec([cut_query])
        if len(valid_queries) == 0:
            return []
        vecs = vecs / np.sqrt(np.sum(vecs ** 2, 1, keepdims=True))
        vec = list(vecs)[0]

        k_neighbors, scores = self.query_ann.get_nns_by_vector(vec, n=k, include_distances=True)
        neighbors = []
        for i in k_neighbors:
            neighbors.append(self.queries[i])
        return sorted(zip(neighbors, scores), key=lambda x: x[-1])

    # def sim(self, u, v):
    #     norm_u = u / np.sqrt(np.sum(u ** 2, keepdims=True))
    #     norm_v = u /np.sqrt(np.sum(v ** 2, keepdims=True))
    #     return np.dot(norm_u, norm_v)

    def get_k_nearest_title(self, title, k):
        if isinstance(title, unicode):
            title = title.encode('utf8')

        cut_data = text_cutter.process({'title': title})
        title = cut_data['cut_title'].decode('utf8')
        vecs, valid_titles = self.model.get_title_vec([title])
        if len(valid_titles) == 0:
            return []
        vecs = vecs / np.sqrt(np.sum(vecs ** 2, 1, keepdims=True))
        vec = list(vecs)[0]
        k_neighbors, scores = self.title_ann.get_nns_by_vector(vec, n=k, include_distances=True)
        neighbors = []
        for i in k_neighbors:
            neighbors.append(self.titles[i])
        return sorted(zip(neighbors, scores), key=lambda x: x[-1])



    def get_answers(self, query, k):
        if isinstance(query, unicode):
            query = query.encode('utf8')

        cut_data = text_cutter.process({'title': query})
        cut_query = cut_data['cut_title'].decode('utf8')
        vecs, valid_queries = self.model.get_query_vec([cut_query])
        if len(valid_queries)==0:
            return []

        vecs = vecs / np.sqrt(np.sum(vecs ** 2, 1, keepdims=True))
        vec = list(vecs)[0]
        # recall titles according to cosine similarity
        candidate_titles_index, scores = self.title_ann.get_nns_by_vector(vec, n=k*10, include_distances=True)

        # rank candidate titles using model
        candidate_titles = []
        for i in candidate_titles_index:
            candidate_titles.append(self.titles[i])

        ranks = self.model.rank_titles(cut_query, candidate_titles)[:k]
        return ranks


    def process(self, data):
        res = {}
        if 'titles' in data:
            res['title_nns'] = self.get_k_nearest_title(data['titles'], 10)
        if 'queries' in data:
            res['query_nns'] = self.get_k_nearest_query(data['queries'], 10)
        return json.dumps(res, ensure_ascii=False).encode('utf8')

if __name__ == '__main__':
    fn_word = 'data/word.list'
    # fn_title = 'data/title.test.txt'
    fn_title = 'data/title.demo'
    ns = NearSentence(fn_word, 'QueryDNN', 'runs/com-dnn-query-click/checkpoints/model')
    ns.load_titles(fn_title, 2)
    titles, scores = ns.get_k_nearest_title(u'被 微软 开除 的 高管 已经 开始 使用 苹果 iphone', 10)
    ns.get_k_nearest_query(u'被 微软 开除 的 高管 已经 开始 使用 苹果 iphone', 10)
    for t, s in sorted(zip(titles, scores), key=lambda x:x[1]):
        print t, s

