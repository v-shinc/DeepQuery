import os
import sys
import json
import tensorflow as tf

from data_helpers import ReadableTestData, CombinedTestData


MAX_QUERY_LEN = 8
MAX_TITLE_LEN = 20

def run_single_model(test_name, test_data, idx2word, dirname, init_model, n_limit=10000):
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", dirname))
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    save_path = os.path.join(checkpoint_dir, "model")
    rank_path = os.path.join(out_dir, test_name + '.rank.res')
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.allow_soft_placement = True
    # config.log_device_placement = True
    with tf.Graph().as_default():
        model = init_model(save_path)
        acc = test_data.evaluate(model, idx2word, rank_path, n_limit)
        # session = tf.Session(config=config)
        # with session.as_default():
        #     model = init_model(session, save_path)
        #     acc = test_data.evaluate(model, idx2word, rank_path, n_limit)
    return acc

def merge_result(names, fnames, fn_out, n_limit):
    files = []

    for fn in fnames:
        files.append(open(fn))

    with open(fn_out, 'w') as fout:
        print >> fout, " ".join(names)
        for i in xrange(n_limit):
            for f in files:
                line = f.readline()
                print >> fout, line.strip()

    for f in files:
        f.close()

def run_models(test_name, test_conf, fn_out):

    # load basic data
    fn_word = 'data/word.list'
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

    EMBED_SIZE = 128
    embedding = np.zeros((num_word, 128)).astype(np.float32)

    # set models
    config = []
    #TODO: read configuration from file
    config.append([
        'com-dnn-query-click',
        'com-dnn-query-click',
        lambda load_path:
        QueryDNN(MAX_QUERY_LEN, MAX_TITLE_LEN, num_word, embed_size=100, hidden_size1=100,
                 hidden_size2=50,
                 gpu=10, load_path=load_path)])

    # config.append([
    #     'com-rnn-click-complete',
    #     'com-rnn-click-complete',
    #     lambda load_path:
    #     QueryRNNPair(MAX_QUERY_LEN, MAX_TITLE_LEN, num_word, embed_size=60, gpu=10, load_path=load_path)])

    config.append([
        'com-attention-click-mix',
        'com-attention-click-mix',
        lambda load_path:
        QueryAttentionSumVec(MAX_QUERY_LEN, MAX_TITLE_LEN, num_word, embed_size=50, gpu=10, load_path=load_path)])

    config.append([
        'dnn-click-sim',
        'dnn-click-sim',
        lambda load_path:
        QueryDNNSim(MAX_QUERY_LEN, MAX_TITLE_LEN, num_word, embed_size=100, gpu=10, hidden_size1=100,
                 hidden_size2=50, load_path=load_path)])

    config.append([
        'dnn-click-sim-constant',
        'dnn-click-sim-constant',
        lambda load_path:
        QueryDNNSim(MAX_QUERY_LEN, MAX_TITLE_LEN, num_word, embed_size=100, gpu=10, hidden_size1=100,
                    hidden_size2=100, load_path=load_path)])

    config.append([
        'dnn-click-inter',
        'dnn-click-inter',
        lambda load_path:
        QueryDNNInter(MAX_QUERY_LEN, MAX_TITLE_LEN, num_word, embed_size=100, gpu=10, hidden_size1=100,
                    hidden_size2=50, load_path=load_path)])

    config.append([
        'dnn-click-inter-constant',
        'dnn-click-inter-constant',
        lambda load_path:
        QueryDNNInter(MAX_QUERY_LEN, MAX_TITLE_LEN, num_word, embed_size=100, gpu=10, hidden_size1=100,
                      hidden_size2=100, load_path=load_path)])




    # test_data = ReadableTestData(fn_test, fn_feature, word2idx, MAX_QUERY_LEN, MAX_TITLE_LEN, PAD)
    test_data = eval(test_conf['test_cls'])(test_conf['fn_test'], test_conf['fn_title'], word2idx, MAX_QUERY_LEN, MAX_TITLE_LEN, PAD)
    res = []
    for c in config:
        print "Start test", c[1]
        acc = run_single_model(test_name, test_data, idx2word, c[1], c[2])
        res.append(acc)
    names = [n for n, _, _ in config]
    fnames = [os.path.abspath(os.path.join(os.path.curdir, "runs", dirname,'rank.res')) for _, dirname, _ in config]

    for n, s in zip(names, res):
        print ('%s\t%s' % (n, s)).encode('utf8')
    merge_result(names, fnames, fn_out, 500)

from query_attention_sumvec import QueryAttentionSumVec
from com_baidu_dnn import QueryDNN
from query_rnn import QueryRNNPair
from dnn_sim import QueryDNNSim
from dnn_inter import QueryDNNInter
import numpy as np

if __name__ == '__main__':

    # COMBINED TEST
    # run_models(
    #     test_name='com.rank',
    #     test_conf={'fn_test': 'data/com.query.click.test.shuffle.dat',
    #                'fn_title': 'data/title.test.txt',
    #                'test_cls': 'CombinedTestData'},
    #     fn_out='data/com.model.rank'
    # )

    # PV TEST
    run_models(
        test_name='pv.rank',
        test_conf={'fn_test': 'data/cut.query.dat',
                   'fn_title': 'data/feature.dat',
                   'test_cls': 'ReadableTestData'},
        fn_out='data/pv.model.rank'
    )

