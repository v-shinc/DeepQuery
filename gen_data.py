import sys
sys.path.insert(0, '..')
import json
from random import random, shuffle
from conf import groupdb_dal, text_cutter
def gen_gid_list(fn, fn_out, n_limit=200000):
    gid_set = set()
    lno = 0
    with open(fn) as fin:
        for line in fin:
            if lno == n_limit:
                break
            data = json.loads(line, encoding='utf8')
            lno += 1
            gid_set.update(data['positive'])
            gid_set.update(data['negative'])
    with open(fn_out, 'w') as fout:
        for gid in gid_set:
            print >> fout, gid

def gen_gid_list_from_long_query(fn, fn_out, n_limit=200000):
    gid_set = set()
    lno = 0
    with open(fn) as fin:
        for line in fin:
            if lno == n_limit:
                break
            data = json.loads(line, encoding='utf8')
            if len(data['cut_query'].split()) < 2:
                continue
            lno += 1
            gid_set.update(data['positive'])
            gid_set.update(data['negative'])
    with open(fn_out, 'w') as fout:
        for gid in gid_set:
            print >> fout, gid

def get_news_from_gid_file(fn_id, fn_out):

    gids = []
    with open(fn_id) as fin:
        for line in fin:
            gids.append(int(line.strip()))

    sql = "select id, title from ss_article_group where id=%s;"
    with open(fn_out, 'w') as fout:
        for gid in gids:
            groupdb_dal.execute(sql, gid)
            row = groupdb_dal.cursor.fetchone()
            if row == None or row['title']==None:
                continue
            if len(row['title']) == 0:
                continue
            cut_data = text_cutter.process({'title': row['title'].encode('utf8')})

            row['cut_title'] = cut_data['cut_title'].decode('utf8')
            print >> fout, json.dumps(row, ensure_ascii=False).encode('utf8')

def cut_query(fn, fn_out):
    with open(fn_out, 'w') as fout:
        with open(fn) as fin:
            for line in fin:
                data = json.loads(line, encoding='utf8')
                cut_data = text_cutter.process({'title': data['query'].encode('utf8')})
                data['cut_query'] = cut_data['cut_title'].decode('utf8')
                data.pop('query')
                print >> fout, json.dumps(data, ensure_ascii=False).encode('utf8')

def get_valid_gid(fn_feature, fn_out):
    with open(fn_out, 'w') as fout:
        with open(fn_feature) as fin:
            for line in fin:
                data = json.loads(line, encoding='utf8')
                print >> fout, data['id']

def count_click_freq(fn_query, fn_valid_gid, fn_out, n_limit):
    gid2idx = dict()
    n_gid = 0
    with open(fn_valid_gid) as fin:
        for line in fin:
            gid2idx[int(line.strip())] = n_gid
            n_gid += 1
    counter = dict()
    lno = 0
    with open(fn_query) as fin:
        for line in fin:
            if lno == n_limit:
                break
            lno += 1
            data = json.loads(line, encoding='utf8')
            for c in data['positive']:
                if c in gid2idx:
                    c = gid2idx[c]
                    counter[c] = counter.get(c, 0) + 1

            for c in data['negative']:
                if c in gid2idx:
                    c = gid2idx[c]
                    counter[c] = counter.get(c, 0) + 1
    counter = sorted(counter.items(), key=lambda x:x[1], reverse=True)
    with open(fn_out, 'w') as fout:
        for g, cnt in counter:
            print >> fout, '%s\t%s' % (g, cnt)

def gen_word_count_list(fn, fn_out):
    counter = dict()
    with open(fn) as fin:
        for line in fin:
            ll = line.decode('utf8').strip().split('\t')
            if len(ll) !=2:
                continue
            _, cut_title = ll
            for w in cut_title.split():
                counter[w] = counter.get(w, 0) + 1
    counter = sorted(counter.items(), key=lambda x:x[1], reverse=True)
    with open(fn_out, 'w') as fout:
        for w, cnt in counter:
            print >> fout, ('%s\t%s' % (w, cnt)).encode('utf8')

def get_x_list_from_count(fn_count, fn_x, threshold):
    with open(fn_x, 'w') as fout:
        with open(fn_count) as fin:
            for line in fin:
                ll = line.decode('utf8').strip().split('\t')
                if len(ll)!=2:
                    continue
                w, cnt = ll
                cnt = int(cnt)
                if cnt >= threshold:
                    print >> fout, w.encode('utf8')

def gen_char_count_list(fn_query, fn_out):
    counter = dict()
    with open(fn_query) as fin:
        for line in fin:
            data = json.loads(line, encoding='utf8')

            for c in data['query']:
                counter[c] = counter.get(c, 0) + 1
    counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    with open(fn_out, 'w') as fout:
        for c, cnt in counter:
            print >> fout, ('%s\t%s' % (c, cnt)).encode('utf8')

def transform_char_query(fn_query, fn_char, fn_valid_gid, fn_out, n_limit):
    gid2idx = dict()
    n_gid = 0
    with open(fn_valid_gid) as fin:
        for line in fin:
            gid2idx[int(line.strip())] = n_gid
            n_gid += 1
    char2idx = dict()
    n_char = 0
    with open(fn_char) as fin:
        for line in fin:
            char2idx[line.decode('utf8').strip()] = n_char
            n_char += 1
    if ' ' not in char2idx:
        char2idx[' '] = n_char
        n_char += 1
    l2s = lambda l: ' '.join([str(i) for i in l])
    lno = 0
    with open(fn_out, 'w') as fout:
        with open(fn_query) as fin:
            for line in fin:
                if lno == n_limit:
                    break
                lno += 1
                data = json.loads(line, encoding='utf8')
                q = [char2idx[c] for c in data['query'] if c in char2idx]
                if len(q) <= 1:
                    continue
                pos = [gid2idx[c] for c in data['positive'] if c in gid2idx]
                if len(pos) == 0:
                    continue
                neg = [gid2idx[c] for c in data['negative'] if c in gid2idx]
                # print >> fout, json.dumps({'q':q, 'pos':pos, 'neg':neg}, ensure_ascii=False).encode('utf8')
                print >> fout, ('%s\t%s\t%s' % (l2s(q), l2s(pos), l2s(neg))).encode('utf8')

def transform_word_query(fn_query, fn_word, fn_valid_gid, fn_out, n_limit):
    gid2idx = dict()
    n_gid = 0
    with open(fn_valid_gid) as fin:
        for line in fin:
            gid2idx[int(line.strip())] = n_gid
            n_gid += 1
    word2idx = dict()
    n_word = 0
    with open(fn_word) as fin:
        for line in fin:
            word2idx[line.decode('utf8').strip()] = n_word
            n_word += 1
    if ' ' not in word2idx:
        word2idx[' '] = n_word
        n_word += 1
    l2s = lambda l: ' '.join([str(i) for i in l])
    lno = 0
    with open(fn_out, 'w') as fout:
        with open(fn_query) as fin:
            for line in fin:
                if lno == n_limit:
                    break

                data = json.loads(line, encoding='utf8')
                if data['cut_query'][0] == '#' and data['cut_query'][-1] == '#':
                    continue
                q = [word2idx[w] for w in data['cut_query'].split() if w in word2idx]
                if len(q) < 3:
                    continue
                pos = [gid2idx[c] for c in data['positive'] if c in gid2idx]
                if len(pos) == 0:
                    continue
                lno += 1
                neg = [gid2idx[c] for c in data['negative'] if c in gid2idx]
                # print >> fout, json.dumps({'q':q, 'pos':pos, 'neg':neg}, ensure_ascii=False).encode('utf8')
                print >> fout, ('%s\t%s\t%s' % (l2s(q), l2s(pos), l2s(neg))).encode('utf8')


def transfrom_query_tmp(fn_query, fn_word, fn_valid_gid, fn_out, n_limit):
    gid2idx = dict()
    n_gid = 0
    with open(fn_valid_gid) as fin:
        for line in fin:
            gid2idx[int(line.strip())] = n_gid
            n_gid += 1
    word2idx = dict()
    n_word = 0
    with open(fn_word) as fin:
        for line in fin:
            word2idx[line.decode('utf8').strip()] = n_word
            n_word += 1
    if ' ' not in word2idx:
        word2idx[' '] = n_word
        n_word += 1
    l2s = lambda l: ' '.join([str(i) for i in l])
    lno = 0
    lines = []
    with open(fn_out, 'w') as fout:
        with open(fn_query) as fin:
            for line in fin:
                if lno == n_limit:
                    break

                data = json.loads(line, encoding='utf8')

                q = [word2idx[w] for w in data['query_terms'].split() if w in word2idx]
                if len(q) < 2:
                    continue
                pos = [gid2idx[c] for c in data['positive'] if c in gid2idx]
                if len(pos) == 0:
                    continue
                lno += 1
                neg = [gid2idx[c] for c in data['skip'] if c in gid2idx] + [gid2idx[c] for c in data['negative'] if c in gid2idx]
                # print >> fout, json.dumps({'q':q, 'pos':pos, 'neg':neg}, ensure_ascii=False).encode('utf8')
                # print >> fout, ('%s\t%s\t%s' % (l2s(q), l2s(pos), l2s(neg))).encode('utf8')
                lines.append(('%s\t%s\t%s' % (l2s(q), l2s(pos), l2s(neg))).encode('utf8'))
        shuffle(lines)
        for line in lines:
            print >> fout, line

def transform_feature(fn_feature, fn_word, fn_valid_gid, fn_out):
    gid2idx = dict()
    n_gid = 0
    with open(fn_valid_gid) as fin:
        for line in fin:
            gid2idx[int(line.strip())] = n_gid
            n_gid += 1
    word2idx = dict()
    n_word = 0
    with open(fn_word) as fin:
        for line in fin:
            word2idx[line.decode('utf8').strip()] = n_word
            n_word += 1
    l2s = lambda l: ' '.join([str(i) for i in l])
    with open(fn_out, 'w') as fout:
        with open(fn_feature) as fin:
            for line in fin:
                data = json.loads(line, encoding='utf8')
                title = [word2idx[w] for w in data['cut_title'].split() if w in word2idx]
                if len(title) < 2:
                    continue

                print >> fout, ('%s\t%s' % (gid2idx[data['id']], l2s(title))).encode('utf8')


def split_train_data(fn, fn_train, fn_dev, fn_test):
    lines = []
    with open(fn) as fin:
        for line in fin:
            lines.append(line.strip())
    shuffle(lines)

    with open(fn_dev, 'w') as fout_dev:
        for i in xrange(5000):
            print >> fout_dev, lines[i]

    with open(fn_test, 'w') as fout_test:
        for i in xrange(5000, 10000):
            print >> fout_test, lines[i]
    l = len(lines)
    with open(fn_train, 'w') as fout_train:
        for i in xrange(10000, l):
            print >> fout_train, lines[i]
    # with open(fn_train, 'w') as fout_train, open(fn_dev, 'w') as fout_dev, open(fn_test, 'w') as fout_test:
    #     # with open(fn) as fin:
    #     for line in lines:
    #         r = random()
    #         if r > 0.95:
    #             print >> fout_test, line.strip()
    #         elif r > 0.9:
    #             print >> fout_dev, line.strip()
    #         else:
    #             print >> fout_train, line.strip()


def filter_short_query(fn_cut_query, fn_out):
    with open(fn_out, 'w') as fout:
        with open(fn_cut_query) as fin:
            for line in fin:
                data = json.loads(line, encoding='utf8')
                q = data['cut_query'].split()
                if len(q) < 2:
                    continue
                print >> fout, line.strip()


def get_news_from_gid_file_local(fn_gid, fn_feature, fn_out):
    gid_set = set()
    with open(fn_gid) as fin:
        for line in fin:
            gid = int(line.strip())
            gid_set.add(gid)

    with open(fn_out, 'w') as fout:
        with open(fn_feature) as fin:
            for line in fin:
                data = json.loads(line, encoding='utf8')
                if data['id'] in gid_set:
                    print >> fout, line.strip()

def combine_raw_query(fn_cut_query, fn_out):
    queries = dict()
    with open(fn_cut_query) as fin:
        for line in fin:
            data = json.loads(line, encoding='utf8')
            q = data['cut_query']
            if q not in queries:
                queries[q] = dict()
            rank = 0
            for c in data['positive']:
                if c not in queries[q]:
                    queries[q][c] = [0, 0]
                queries[q][c][0] += rank
                rank += 1
                queries[q][c][1] += 1   # freq

            for c in data['negative']:
                if c not in queries[q]:
                    queries[q][c] = [0, 0]
                queries[q][c][0] += rank
                rank += 1
                queries[q][c][1] += 1  # freq

    with open(fn_out, 'w') as fout:
        for q, vs in queries.iteritems():
            rank = [[c, r*1.0/f] for c, (r, f) in vs.iteritems()]

            rank = sorted(rank, key=lambda x:x[1])
            print >> fout, json.dumps({'query': q, 'title':rank}, ensure_ascii=False).encode('utf8')

    # queries = dict()
    # with open(fn_cut_query) as fin:
    #     for line in fin:
    #         data = json.loads(line, encoding='utf8')
    #         q = data['cut_query']
    #         if q not in queries:
    #             queries[q] = dict()
    #         for c in data['positive']:
    #             if c not in queries[q]:
    #                 queries[q][c] = 0
    #             queries[q][c] += 1
    #
    #         for c in data['negative']:
    #             if c not in queries[q]:
    #                 queries[q][c] = 0
    #             queries[q][c] -= 1
    # with open(fn_out, 'w') as fout:
    #     for q, vs in queries.iteritems():
    #         rank = sorted(vs.items(), key=lambda x:x[1], reverse=True)
    #         print >> fout, json.dumps({'query': q, 'title':rank}, ensure_ascii=False).encode('utf8')

def transform_combined_query(fn_combine, fn_word, fn_valid_gid, fn_out, n_limit):
    gid2idx = dict()
    n_gid = 0
    with open(fn_valid_gid) as fin:
        for line in fin:
            gid2idx[int(line.strip())] = n_gid
            n_gid += 1
    word2idx = dict()
    n_word = 0
    with open(fn_word) as fin:
        for line in fin:
            word2idx[line.decode('utf8').strip()] = n_word
            n_word += 1
    if ' ' not in word2idx:
        word2idx[' '] = n_word
        n_word += 1
    l2s = lambda l: ' '.join([str(i) for i in l])
    lno = 0
    with open(fn_out, 'w') as fout:
        with open(fn_combine) as fin:
            for line in fin:
                if lno == n_limit:
                    break

                data = json.loads(line, encoding='utf8')
                # if data['query'][0] == '#' and data['cut_query'][-1] == '#':
                #     continue
                q = [word2idx[w] for w in data['query'].split() if w in word2idx]
                if len(q) < 2:
                    continue

                titles = [str(gid2idx[c]) + ' ' + str(r) for c, r in data['title'] if c in gid2idx]
                if len(titles) == 0:
                    continue
                lno += 1
                # print >> fout, json.dumps({'q':q, 'pos':pos, 'neg':neg}, ensure_ascii=False).encode('utf8')
                print >> fout, ('%s\t%s' % (l2s(q), ' '.join(titles))).encode('utf8')


def gen_mini_data():
    fn_query = 'data/query.dat'
    fn_cut_query = 'data/cut.query.dat'
    fn_gid = 'data/gid.mini.list'
    fn_feature = 'data/feature.mini.dat'
    fn_valid_gid = 'data/gid.valid.mini.list'
    fn_gid_count = 'data/gid.count.mini.v'
    fn_title = 'data/title.txt'
    fn_word_count = 'data/word.count.list'
    fn_word = 'data/word.list'
    fn_char_count = 'data/char.count.list'
    fn_char = 'data/char.list'
    fn_query_v = 'data/query.mini.v'
    fn_feature_v = 'data/feature.mini.v'
    fn_train = 'data/query.train.mini.v'
    fn_dev = 'data/query.dev.mini.v'
    fn_test = 'data/query.test.mini.v'
    # gen_gid_list(fn_query, fn_gid)
    # get_news_from_gid_file(fn_gid, fn_feature)
    # get_valid_gid(fn_feature, fn_valid_gid)
    # count_click_freq(fn_query, fn_valid_gid, fn_gid_count, 200000)
    # gen_word_count_list(fn_title, fn_word_count)
    # get_x_list_from_count(fn_word_count, fn_word, 10)
    # gen_char_count_list(fn_query, fn_char_count)
    # get_x_list_from_count(fn_char_count, fn_char, 10)
    # transform_char_query(fn_query, fn_char, fn_valid_gid, fn_query_v, 200000)
    # transform_feature(fn_feature, fn_word, fn_valid_gid, fn_feature_v)
    # split_train_data(fn_query_v, fn_train, fn_dev, fn_test)

    # word version query
    fn_query_w_v ='data/query.mini.w.v'
    fn_w_train = 'data/query.train.mini.w.v'
    fn_w_dev = 'data/query.dev.mini.w.v'
    fn_w_test = 'data/query.test.mini.w.v'
    fn_feature_complete  = 'data/feature.dat'
    # cut_query(fn_query, fn_cut_query)
    gen_gid_list_from_long_query(fn_cut_query, fn_gid, 200000)
    get_news_from_gid_file_local(fn_gid, fn_feature_complete, fn_feature)
    get_valid_gid(fn_feature, fn_valid_gid)
    count_click_freq(fn_query, fn_valid_gid, fn_gid_count, 200000)
    transform_word_query(fn_cut_query, fn_word, fn_valid_gid, fn_query_w_v, 200000)
    split_train_data(fn_query_w_v, fn_w_train, fn_w_dev, fn_w_test)
    transform_feature(fn_feature, fn_word, fn_valid_gid, fn_feature_v)

def get_title_for_query_file(fn_combine, fn_exist_titles_list, fn_out):

    gids = set()
    for fn_title in fn_exist_titles_list:
        with open(fn_title) as fin:
            for line in fin:
                try:
                    data = json.loads(line, encoding='utf8')
                    gid = data['id']
                    gids.add(gid)
                except ValueError as e:
                    print e
                    print line.strip()


    sql = "select id, title from ss_article_group where id=%s;"
    lno = 0
    with open(fn_out, 'w') as fout:
        with open(fn_combine) as fin:
            for line in fin:
                if lno % 1000 == 0:
                    sys.stdout.write('process to %d\r' % lno)
                    sys.stdout.flush()
                lno += 1
                data = json.loads(line, encoding='utf8')
                for c, r in data['title']:
                    if c not in gids:
                        groupdb_dal.execute(sql, c)
                        row = groupdb_dal.cursor.fetchone()
                        if row == None or row['title']==None:
                            continue
                        if len(row['title']) == 0:
                            continue
                        cut_data = text_cutter.process({'title': row['title'].encode('utf8')})

                        row['cut_title'] = cut_data['cut_title'].decode('utf8')
                        row['cut_title'] = row['cut_title'].replace('\n', '').replace('\r', '')
                        # print >> fout, ("%s\t%s" % (c, row['cut_title'])).encode('utf8')
                        print >> fout, json.dumps(row, ensure_ascii=False).encode('utf8')
                        gids.add(c)

def merge_titles(fn_title_list, fn_out):
    gids = set()
    with open(fn_out, 'w') as fout:
        for fn in fn_title_list:
            with open(fn) as fin:
                for line in fin:
                    try:
                        # gid = int(line.split('\t')[0])
                        data = json.loads(line, encoding='utf8')
                        gid = data['id']
                        if gid in gids:
                            continue
                        print >> fout, line.strip()
                        gids.add(gid)
                    except ValueError as e:
                        print e
                        print line.strip()

def gen_data():
    fn_query = 'data/query.dat'
    fn_cut_query = 'data/cut.query.dat'
    fn_combine = 'data/com.cut.query.dat'
    fn_gid = 'data/gid.list'
    fn_feature = 'data/feature.dat'
    fn_valid_gid = 'data/gid.valid.list'
    fn_gid_count = 'data/gid.count.v'
    # fn_title = 'data/title.txt'
    # fn_word_count = 'data/word.count.list'
    fn_word = 'data/word.list'
    # fn_char_count = 'data/char.count.list'
    fn_char = 'data/char.list'
    fn_query_v = 'data/query.v'

    fn_feature_v = 'data/feature.v'
    fn_train = 'data/query.train.v'
    fn_dev = 'data/query.dev.v'
    fn_test = 'data/query.test.v'

    fn_combine_query_v = 'data/combine.query.v'


    # combine_raw_query(fn_cut_query, fn_combine)
    # gen_gid_list(fn_query, fn_gid, n_limit=9000000)
    # get_news_from_gid_file(fn_gid, fn_feature)
    # get_valid_gid(fn_feature, fn_valid_gid)
    # count_click_freq(fn_query, fn_valid_gid, fn_gid_count, 9000000)
    # gen_word_count_list(fn_title, fn_word_count)
    # get_x_list_from_count(fn_word_count, fn_word, 10)
    # gen_char_count_list(fn_query, fn_char_count)
    # get_x_list_from_count(fn_char_count, fn_char, 10)
    # transform_char_query(fn_query, fn_char, fn_valid_gid, fn_query_v, 9000000)
    # transform_word_query(fn_cut_query, fn_word, fn_valid_gid, fn_query_v, 9000000)
    # transform_feature(fn_feature, fn_word, fn_valid_gid, fn_feature_v)
    # split_train_data(fn_query_v, fn_train, fn_dev, fn_test)
    # transform_combined_query(fn_combine, fn_word, fn_valid_gid, fn_combine_query_v, 9000000)
    # transfrom_query_tmp('data/query.test.dat', fn_word, fn_valid_gid, fn_test, 30000)
    get_title_for_query_file('data/com.query.click.shuffle.0.dat', ['data/feature.dat'], 'data/feature.0.dat')


if __name__ == '__main__':

    # fn_query = 'data/query.dat'
    # fn_gid_mini = 'data/gid.mini.list'
    # fn_feature_mini = 'data/feature.mini.dat'
    # gen_gid_list(fn_query, fn_gid_mini)
    # get_news_from_gid_file(fn_gid_mini, fn_feature_mini)

    # with open('data/query.dat', 'w') as fout:
    #     with open('../data/traindata20160701') as fin:
    #         for line in fin:
    #             data = json.loads(line, encoding='utf8')
    #             print >> fout, json.dumps(data, ensure_ascii=False).encode('utf8')

    # gen_mini_data()
    gen_data()