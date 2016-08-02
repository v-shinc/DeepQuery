import sys
sys.path.insert(0, '..')
import json
import hashlib
import os
from conf import groupdb_dal, text_cutter


def combine_raw_query(fn_query, fn_out, num):
    queries = dict()
    hash = lambda s: int(hashlib.sha1(s).hexdigest(), 16) % (num)
    with open(fn_out, 'w') as fout:
        with open(fn_query) as fin:
            for i in range(num):
                fin.seek(0)
                queries.clear()
                for line in fin:
                    data = json.loads(line, encoding='utf8')
                    q = data['query'].encode('utf8')
                    code = hash(q)
                    if code != i:
                        continue

                    q = data['query_terms']
                    if q not in queries:
                        queries[q] = dict()
                    rank = 0
                    for c in data['positive']:
                        if c not in queries[q]:
                            queries[q][c] = [0, 0]
                        queries[q][c][0] += rank
                        rank += 1
                        queries[q][c][1] += 1   # freq

                    for c in data['skip']:
                        if c not in queries[q]:
                            queries[q][c] = [0, 0]
                        queries[q][c][0] += rank
                        rank += 1
                        queries[q][c][1] += 1  # freq

                    for c in data['negative']:
                        if c not in queries[q]:
                            queries[q][c] = [0, 0]
                        queries[q][c][0] += rank
                        rank += 1
                        queries[q][c][1] += 1  # freq

                for q, vs in queries.iteritems():
                    rank = [[c, r*1.0/f] for c, (r, f) in vs.iteritems()]

                    rank = sorted(rank, key=lambda x:x[1])
                    print >> fout, json.dumps({'query': q, 'title':rank}, ensure_ascii=False).encode('utf8')
                # yield json.dumps({'query': q, 'title':rank}, ensure_ascii=False).encode('utf8')

def map_query(fn_query, num):
    fouts = []
    fns = []
    if not os.path.exists('data/map_query'):
        os.makedirs('data/map_query')
    for i in range(num):
        fouts.append(open('data/map_query/query-%s' % i, 'w'))
        fns.append('data/map_query/query-%s' % num)
    hash = lambda s: int(hashlib.sha1(s).hexdigest(), 16) % (num)
    with open(fn_query) as fin:
        for line in fin:
            data = json.loads(line, encoding='utf8')
            q = data['query'].encode('utf8')
            code = hash(q)
            print >> fouts[code], line.strip()

    return fns


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



def split_train_data(fn, fn_train, fn_dev, fn_test):
    with open(fn) as fin:

        with open(fn_dev, 'w') as fout_dev:
            for i in xrange(5000):
                print >> fout_dev, fin.readline().strip()

        with open(fn_test, 'w') as fout_test:
            for i in xrange(5000, 10000):
                print >> fout_test, fin.readline().strip()

        with open(fn_train, 'w') as fout_train:
            for line in fin:
                print >> fout_train, line.strip()

# def gen_gid_list(fn_title_list, fn_gid):
#     gid_set = set()
#     for fn in fn_title_list:
#         with open(fn) as fin:
#             for line in fin:
#                 gid = line.strip().decode('utf8').split('\t')
#                 gid_set.add(gid)
#
#     with open(fn_gid, 'w') as fout:
#         for gid in gid_set:
#             print >> fout, gid

def transform_title(fn_title_list, fn_word, fn_gid, fn_out):
    word2idx = dict()
    n_word = 0
    with open(fn_word) as fin:
        for line in fin:
            word2idx[line.decode('utf8').strip()] = n_word
            n_word += 1
    l2s = lambda l: ' '.join([str(i) for i in l])
    # gid2idx = dict()
    gid_set = set()
    n_gid = 0
    with open(fn_gid, 'w') as fout_gid:
        with open(fn_out, 'w') as fout:
            for fn_title in fn_title_list:
                with open(fn_title) as fin:
                    for line in fin:
                        ll = line.strip().decode('utf8').split('\t')
                        if len(ll) != 2:
                            continue
                        gid, cut_title = ll
                        gid = int(gid)
                        if gid in gid_set:
                            continue
                        cut_title = [word2idx[w] for w in cut_title.split() if w in word2idx]
                        if len(cut_title) < 2:
                            continue

                        print >> fout, ('%s\t%s' % (n_gid, l2s(cut_title))).encode('utf8')
                        print >> fout_gid, gid
                        n_gid += 1
                        gid_set.add(gid)

def transform_combined_query(fn_combine, fn_word, fn_valid_gid, fn_out):
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
        with open(fn_combine) as fin:
            for line in fin:
                data = json.loads(line, encoding='utf8')

                q = [word2idx[w] for w in data['query'].split() if w in word2idx]
                if len(q) < 2:
                    continue
                # for c, r in data['title']:
                #     if c not in gid2idx:
                #         print >> fout, c
                titles = [str(gid2idx[c]) + ' ' + str(r) for c, r in data['title'] if c in gid2idx]
                if len(titles) == 0:
                    continue
                # print >> fout, json.dumps({'q':q, 'pos':pos, 'neg':neg}, ensure_ascii=False).encode('utf8')
                print >> fout, ('%s\t%s' % (l2s(q), ' '.join(titles))).encode('utf8')

def transform_combined_clean_query(fn_combine, fn_word, fn_valid_gid, fn_out):
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
        with open(fn_combine) as fin:
            for line in fin:
                data = json.loads(line, encoding='utf8')

                q = [word2idx[w] for w in data['query'].split() if w in word2idx]
                if len(q) < 2:
                    continue
                # for c, r in data['title']:
                #     if c not in gid2idx:
                #         print >> fout, c
                titles = [str(gid2idx[c]) + ' ' + str(r) for c, r, s in data['title'] if c in gid2idx and (s>0.5 or c > 0)]
                if len(titles) == 0:
                    continue
                # print >> fout, json.dumps({'q':q, 'pos':pos, 'neg':neg}, ensure_ascii=False).encode('utf8')
                print >> fout, ('%s\t%s' % (l2s(q), ' '.join(titles))).encode('utf8')


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



def recons_combined_query(fn_com_query, fn_feature, fn_word, fn_out):
    idx2word = dict()
    n_word = 0
    with open(fn_word) as fin:
        for line in fin:
            idx2word[n_word] = line.decode('utf8').strip()
            n_word += 1

    indices2sent = lambda ll: ''.join([idx2word[i] for i in ll])

    features = dict()
    with open(fn_feature) as fin:
        for line in fin:
            idx, widx = line.strip().split('\t')
            features[int(idx)] = map(int, widx.split())
    with open(fn_out, 'w') as fout:
        with open(fn_com_query) as fin:
            for i in xrange(100):
                line = fin.readline()
                q, vs = line.strip().split('\t')
                q = map(int, q.split())
                q = indices2sent(q)
                vs = vs.split()
                ts = map(int, vs[::2])
                rank = vs[1::2]
                for j, t in enumerate(ts):
                    t = indices2sent(features[t])
                    ts[j] = '('+t+', '+rank[j]+')'

                print >> fout, ('%s\t%s' % (q, ' '.join(ts))).encode('utf8')

def shuffle_file(fn, fn_out, chunk_size):
    from random import shuffle
    lines = []
    with open(fn_out, 'w') as fout:
        with open(fn) as fin:
            for line in fin:
                lines.append(line.strip())
                if len(lines) > chunk_size:
                    shuffle(lines)

                    for l in lines:
                        print >> fout, l
                    lines = []

            for l in lines:
                print >> fout, l


def shuffle_large_file(fn, fn_out):
    fn_tmp = 'data/shuffle.tmp'
    shuffle_file(fn, fn_out, 123412)
    shuffle_file(fn_out, fn_tmp, 432143)
    shuffle_file(fn_tmp, fn_out, 234523)
    os.remove(fn_tmp)



def get_title_for_query_file(fn_combine, fn_exist_titles_list, fn_out):

    gids = set()
    for fn_title in fn_exist_titles_list:
        with open(fn_title) as fin:
            for line in fin:
                try:
                    gid = int(line.split('\t')[0])
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
                        print >> fout, ("%s\t%s" % (c, row['cut_title'])).encode('utf8')
                        gids.add(c)

def merge_titles(fn_title_list, fn_out):
    gids = set()
    with open(fn_out, 'w') as fout:
        for fn in fn_title_list:
            with open(fn) as fin:
                for line in fin:
                    try:
                        gid = int(line.split('\t')[0])
                        if gid in gids:
                            continue
                        print >> fout, line.strip()
                        gids.add(gid)
                    except ValueError as e:
                        print e
                        print line.strip()


def transform_word2vec(fn_word2vec, fn_word, fn_idx2vec):
    word2idx = dict()
    n_word = 0
    with open(fn_word) as fin:
        for line in fin:
            w = line.decode('utf8').strip()
            word2idx[w] = n_word
            n_word += 1

    with open(fn_idx2vec, 'w') as fout:
        with open(fn_word2vec) as fin:
            for line in fin:
                ll = line.decode('utf8').strip().split(' ', 1)
                if ll[0] in word2idx:

                    print >> fout, "%s\t%s" % (word2idx[ll[0]], ll[1])

def gen_query_list(fn_query, fn_out):
    with open(fn_out, 'w') as fout:
        with open(fn_query) as fin:
            for line in fin:
                print >> fout, json.loads(line)['query'].encode('utf8')

if __name__ == '__main__':
    # fn_test_query = 'data/cut.query.dat'
    # fn_query = 'data/train_data.txt'
    # fn_com_query = 'data/com.query.dat'
    # fn_title = 'data/title.txt'
    # fn_title2 = 'data/title2.txt'
    # fn_title3 = 'data/title3.txt'
    # fn_word = 'data/word.list'
    # fn_valid_gid = 'data/gid.valid.list'
    #
    # fn_feature_v = 'data/feature.v'
    # fn_train = 'data/combine.query.v'
    # fn_dev = 'data/query.dev.v'
    # fn_test = 'data/query.test.v'
    #
    # fn_com_click_query0 = 'data/com.query.click.shuffle.0.dat'
    # fn_com_click_query = 'data/com.query.click.shuffle.dat'
    # fn_com_click_query_v = 'data/com.query.click.v'
    # fn_com_click_query_v0 = 'data/com.query.click.0.v'
    # combine_raw_query(fn_query, fn_com_query, 5)
    # transform_feature(['data/title.all.txt', 'data/title4.txt'], fn_word, fn_valid_gid, fn_feature_v)
    # transform_combined_query(fn_com_query, fn_word, fn_valid_gid, fn_train)
    # split_train_data(fn_query_v, fn_train, fn_dev, fn_test)
    # transform_word_query(fn_test_query, fn_word, fn_valid_gid, fn_dev, 5000)

    # recons_combined_query(fn_train, fn_feature_v, fn_word, 'data/recons1.txt')
    # shuffle_large_file('data/com.query.click.dat', 'data/com.query.click.shuffle.dat')
    # transform_combined_query(fn_com_click_query, fn_word, fn_valid_gid, fn_com_click_query_v)

    # get_title_for_query_file(fn_com_click_query, ['data/title.all.txt'], 'data/title5.txt')

    # transform_combined_query(fn_com_click_query, fn_word, fn_valid_gid, 'data/debug.txt')

    # recons_combined_query(fn_com_click_query_v, fn_feature_v, fn_word, 'data/recons.click.txt')
    # merge_titles(['data/title.all.txt', 'data/title4.txt'], 'data/title5.txt')
    # transform_feature(['data/title5.txt'], fn_word, fn_valid_gid, fn_feature_v)
    # transform_combined_query(fn_com_click_query, fn_word, fn_valid_gid, fn_com_click_query_v)
    # shuffle_large_file('data/com.query.click.0.dat', 'data/com.query.click.shuffle.0.dat')

    # transform_combined_query(fn_com_click_query0, fn_word, fn_valid_gid, fn_com_click_query_v0)

    # transform_word2vec('data/content.vec.txt', fn_word, 'data/index.vec.txt')

    # fn_clean = 'data/com.query.click.clean.dat'
    # fn_clean_v = 'data/com.query.click.clean.v'
    # transform_combined_clean_query(fn_clean, fn_word, fn_valid_gid, fn_clean_v)

    # shuffle_large_file('data/com.query.click.test.dat', 'data/com.query.click.test.shuffle.dat')
    # gen_query_list('data/com.query.click.test.shuffle.dat', 'data/query.test.list')


    # process train data 20150924_20160630
    fn_word = 'data/word.list'
    fn_title = 'data/title_20150924_20160630.txt'
    fn_valid_gid = 'data/gid.valid.20150924.20160630.list'
    fn_title_v = 'data/title.20150924.20160630.v'
    fn_com_click_train_query = 'data/com.query.click.20150924.20160630.dat'
    fn_com_click_train_shuffle_query = 'data/com.query.click.20150924.20160630.shuffle.dat'
    fn_click_train_query_v = 'data/com.query.click.20150924.20160630.v'
    # transform_title([fn_title], fn_word, fn_valid_gid, fn_title_v)
    # shuffle_large_file(fn_com_click_train_query, fn_com_click_train_shuffle_query)
    # transform_combined_query(fn_com_click_train_shuffle_query, fn_word, fn_valid_gid, fn_click_train_query_v)

    recons_combined_query(fn_click_train_query_v, fn_title_v, fn_word, 'data/recons.click.txt')

