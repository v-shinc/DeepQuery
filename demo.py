
from flask import Flask, jsonify
from flask.ext.bootstrap import Bootstrap
from flask import render_template,request
from flask import abort, redirect, url_for
import json
from find_similar_sentence import NearSentence

static_path = 'static'

class QueryDemo(object):

    def __init__(self):
        self.app = Flask(__name__, template_folder=static_path, static_folder=static_path)
        self.bootstrap = Bootstrap(self.app)
        fn_word = 'data/word.list'
        # fn_title = 'data/title.test.txt'
        fn_title = 'data/title.demo'
        # fn_query = 'data/query.test.list'
        fn_query = 'data/query.demo'
        print '[In QueryDemo] init NearSentence'
        self.ns = NearSentence(fn_word, 'QueryDNN', 'runs/com-dnn-query-click/checkpoints/model')
        print '[In QueryDemo] load titles'
        self.ns.load_titles(fn_title, 2)
        print '[In QueryDemo] load queries'
        self.ns.load_queries(fn_query, 1)

        @self.app.route("/t/query_demo/")
        def main():
            return render_template('main.html', params={})

        @self.app.route('/t/get_query_nns/', methods=['GET', 'POST'])
        def get_query_nns():
            query = request.args.get('query')
            print query
            nns = self.ns.get_k_nearest_query(query, 20)


            print '[in get_query_nns]'
            # for q, s in nns:
            #     print q, s
            nns = [{'query': q, 'score': s} for q, s in nns]
            return json.dumps(nns)


        @self.app.route('/t/get_title_nns/', methods=['GET', 'POST'])
        def get_title_nns():
            # query = request.args.get('query', '', type=str)
            title = request.args.get('title')
            # print query
            # print request.args
            # nns : [(title, score),...,]
            nns = self.ns.get_k_nearest_title(title, 20)

            # for t, s in nns:
            #     print t, s
            nns = [{'title': t, 'score': s} for t, s in nns]
            return json.dumps(nns)

        @self.app.route('/t/get_answers/', methods=['GET', 'POST'])
        def get_answers():
            query = request.args.get('query')
            print query
            nns = self.ns.get_answers(query, 20)

            print '[in get_answers]'
            for q, s in nns:
                print q, s
            nns = [{'query': q, 'score': s} for q, s in nns]
            return json.dumps(nns)

    def run(self, port,debug=False):
        print '[In run] start to run app'
        self.app.run(host='0.0.0.0',port=port, debug=debug)

if __name__ == '__main__':
    obj = QueryDemo()
    obj.run(7778, True)
