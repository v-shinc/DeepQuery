import sys
import os
import json
from itertools import groupby
from operator import itemgetter

HADOOP_HOME = '/opt/tiger/yarn_deploy/hadoop-2.6.0-cdh5.4.4'
HADOOP_STREAMING = HADOOP_HOME + '/share/hadoop/tools/lib/hadoop-streaming-2.6.0-cdh5.4.4.jar'



def mapper():
    for line in sys.stdin:
        data = json.loads(line, encoding='utf8')
        k = data.get('query_terms', data.get('cut_query', ''))
        if k == '':
            continue
        for c in data['positive']:
            print ("%s\t%s\t%s" % (k, c, 1)).encode('utf8')

        for c in data['negative']:
            print ("%s\t%s\t%s" % (k, c, 0)).encode('utf8')
        if 'skip' in data:
            for c in data['skip']:
                print ("%s\t%s\t%s" % (k, c, 0)).encode('utf8')


def read_mapper_output(fin):
    for line in fin:
        q, t, c = line.decode('utf8').rstrip().split('\t')

        yield [q, int(t), int(c)]
        # yield line.decode('utf8').rstrip().split('\t')

def reducer():
    data = read_mapper_output(sys.stdin)
    for query, group in groupby(data, itemgetter(0)):
        rank = dict()
        for _, t, c in group:
            if t not in rank:
                rank[t] = 0
            rank[t] += c
        rank = sorted(rank.items(), key=lambda x: x[1], reverse=True)
        print json.dumps({'query': query, 'title': rank}, ensure_ascii=False).encode('utf8')




def main():
    cmd = "HADOOP_USER_NAME=tiger %s/bin/hadoop fs -rmr /user/chenshini/combined" % (HADOOP_HOME)
    os.system(cmd)

    cmd = 'HADOOP_USER_NAME=tiger %s/bin/hadoop jar %s ' % (HADOOP_HOME, HADOOP_STREAMING) +\
    ' -Dmapred.job.queue.name=offline.data' +\
    ' -Dmapred.job.name="combine_query_click_chenshini"' +\
    ' -Dmapred.min.split.size=1036870912 ' +\
    ' -Dmapred.job.priority=HIGH ' +\
    ' -Dmapred.reduce.tasks=8 ' +\
    ' -Dmapred.output.compress=false' +\
    ' -Dstream.num.map.output.key.fields=1 ' +\
    ' -Dmapreduce.reduce.java.opts=-Xmx4096M' +\
    ' -Dmapreduce.reduce.memory.mb=4512' +\
    ' -Dnum.key.fields.for.partition=1' +\
    ' -partitioner org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner ' + \
    ' -input "/user/chenshini/query_20150924_20160630.txt"' +\
    ' -output "/user/chenshini/combined"' +\
    ' -mapper "python combine_query_by_click.py mapper"' +\
    ' -reducer "python combine_query_by_click.py reducer"' +\
    ' -file "combine_query_by_click.py"'

    ret = os.system(cmd)
    if ret != 0:
        print >> sys.stderr, "err"
        return False

    cmd = 'HADOOP_USER_NAME=tiger /opt/tiger/yarn_deploy/hadoop-2.6.0-cdh5.4.4/bin/hadoop fs -cat /user/chenshini/combined/part* > ' + 'data/com.query.click.20150924.20160630.dat'
    ret = os.system(cmd)
    if ret != 0:
        print >> sys.stderr, "err"
        return False

    return True

if __name__ == '__main__':
    # if sys.argv[1] == "process":
    if len(sys.argv) == 1:
        print 'main is running'
        main()
    elif sys.argv[1] == "mapper":
        mapper()
    elif sys.argv[1] == "reducer":
        reducer()