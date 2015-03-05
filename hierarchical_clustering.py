#!/usr/bin/python
#encoding:utf8

"""
clustering diagonosis whith  hierarchical clustering  algorithm
"""

import sys,os
import re
from datetime import datetime
import time
import math
import ast
import logging
import gc
from array import array

logging.basicConfig(level=logging.DEBUG)

reload(sys)
sys.setdefaultencoding('utf8')



# min - best precision, max - best recall
CLUSTER_SIMI_MAX = 0
CLUSTER_SIMI_AVG = 1
CLUSTER_SIMI_MIN = 2

MIN_SIMI = 0.
MAX_SIMI = float(sys.maxint)



number_pt = re.compile(u'[0-9\uff10-\uff19]')
all_punctuation = dict((ord(char), u'') for char in u'\'!"#$%&\'(*+,-./:;<=>?@[\\]^_`{，、|}~\'')




def big_group(groups):
    '''groups size greater than 1'''
    rst = [group for group in groups if len(group) > 1]
    return rst



class HierarchicalCluster:
    def __init__(self, threshold, op_cluster_simi = CLUSTER_SIMI_AVG):
        self.__threshold = threshold
        self.__op_cluster_simi = op_cluster_simi
        self.__matrix = None
        self.__indexclusters = []
        self.__samples = []
        self.__sample_size = 0
        self.__simifunc = None


    def cluster(self, samples, simifunc, given_struct = None):
        ts0 = datetime.now()
        if not samples or not simifunc:
            return []
        logging.debug('cluster: %d records' % len(samples))

        self.__sample_size = len(samples)
        self.__samples = samples
        self.__simifunc = simifunc

        ts0 = datetime.now()

        logging.info("build distance matrix...")
        sys.stdout.flush()
        self.__build_simi_matrix()
        ts1 = datetime.now()


        logging.info('dist matrix:')
        #self.print_simi_matrix()

        logging.info('init cluseter...')
        sys.stdout.flush()
        self.__init_clusters(given_struct)

        #print self.__indexclusters

        ts2 = datetime.now()

        logging.info('merge cluster...')
        sys.stdout.flush()

        self.__merge()
        ts3 = datetime.now()


        #get cluster data
        clustered_samples = []
        for indexlist in self.__indexclusters:
            group = []
            for index in indexlist:
                elem = self.__samples[index]
                group.append(elem)
            clustered_samples.append(group)

        self.clear()
        ts4 = datetime.now()

        logging.info('cost: %ds total, %ds matrix, %ds init_clusetr %ds merge %ds append' % ( (ts4-ts0).seconds,(ts1-ts0).seconds, (ts2-ts1).seconds,(ts3-ts2).seconds,(ts4-ts3).seconds))

        logging.info("\n\n\n")
        return clustered_samples


    def clear(self):
        """clear internal data for one task"""
        self.__indexclusters[:] = []
        self.__sample_size = 0
        self.__samples[:] = []
        self.__simifunc = None

    def __build_simi_matrix(self):
        """build similarity matrix"""
        matrix_size = self.__sample_size * (self.__sample_size - 1) / 2
        self.__matrix = array('f')
        self.__matrix.extend([0.0]*matrix_size)
        for i in range(1, self.__sample_size):
            x  = self.__samples[i][0]
            #id_i = self.__samples[i]
            for j in range(0, i):
                y = self.__samples[j][0]
                #id_j = self.__samples[j]
                #self.__matrix[self.__pos(i, j)] = self.__simifunc(dict_data[id_i],dict_data[id_j])
                self.__matrix[self.__pos(i, j)] = self.__simifunc(x, y)



    def __init_clusters(self, given_struct):
        if given_struct is None:
            self.__indexclusters[:] = [[ele] for ele in range(self.__sample_size)]
        else:
            self.__indexclusters[:] = given_struct

    def __pos(self, i, j):
        """convert a 2-D matrix coordinates to 1-D list index"""
        return i * (i - 1) / 2 + j


    def __merge(self):
        """merge phase"""
        while True:
            #import pdb; pdb.set_trace()
            pairs_to_merge = self.__find_similar_pairs()
            if pairs_to_merge:
                clusters_to_remove = set()
                for (x, y) in pairs_to_merge:
                    self.__indexclusters[x].extend(self.__indexclusters[y])
                    clusters_to_remove.add(y)
                self.__indexclusters[:] = [cluster for index, cluster
                        in enumerate(self.__indexclusters) if index not in clusters_to_remove]
                #logging.debug('%d pairs merged in this round' % (len(pairs_to_merge)))
                #for cluster in self.__indexclusters:
                    #logging.debug(cluster)
            else:
                #logging.debug('final structure:')
                #for cluster in self.__indexclusters:
                #    logging.debug(cluster)
                return


    def __find_similar_pairs(self):
        """find non-overlapped similar pairs in desc order of simi"""
        size = len(self.__indexclusters)
        candidates = []
        for i in range(size):
            for j in range(i+1, size):
                simi = self.__cluster_simi(i, j)
                #print simi, self.__indexclusters[i],self.__indexclusters[j]
                if simi >= self.__threshold:
                    candidates.append((simi, i, j))
        candidates.sort(reverse = True, key = lambda x: x[0])


        # filter overlapped pairs
        to_remove = set()
        appeared = set()
        for index, cand in enumerate(candidates):
            if cand[1] not in appeared and cand[2] not in appeared:
                appeared.add(cand[1])
                appeared.add(cand[2])
            else:
                to_remove.add(index)

        #print 'ahha'
        #print [(cand[1], cand[2]) for index, cand in enumerate(candidates)  if index not in to_remove]

        return [(cand[1], cand[2]) for index, cand in enumerate(candidates)
                if index not in to_remove]


    def __cluster_simi(self, i, j):
        """avg similarity between clusters"""
        sum_ = 0.
        for si in self.__indexclusters[i]:
            for sj in self.__indexclusters[j]:
                simi = self.__sample_simi(si, sj)
                sum_ += simi
        return sum_ / (len(self.__indexclusters[i]) * len(self.__indexclusters[j]))



    def __sample_simi(self, i, j):
        """similarity between samples"""
        if j > i:
            i, j = j, i
        return self.__matrix[self.__pos(i, j)]


class ClusteringUser:
    def __init__(self, threshold,op_cluster_simi, words_file_name, rst_file_name ):
            self.__threshold = threshold
            self.__samples = []
            self.__simifunc = None
            self.words_file_name = words_file_name 
            self.rst_file_name = rst_file_name

            self.cluster_rst = []

    def filter_clusters(self,clustered_samples):
        '''keep big clusters'''
        count = 0
        for  cluster in clustered_samples:
            cluser_size = len(cluster)
            if cluser_size > 1:
                self.cluster_rst.append(cluster)
                count += 1
        logging.debug("%d big clusters" % count)


    def clustering(self, data):
        hcluster = HierarchicalCluster(self.__threshold, CLUSTER_SIMI_AVG)
        clustered_samples = hcluster.cluster(data, new_sim_func)
        #print_cluster(clustered_samples)
        self.filter_clusters(clustered_samples)


    def save_cluster2file(self):
        logging.debug("save cluster rst file to %s..." % self.rst_file_name)
        rst_file = open(self.rst_file_name,'w')
        for group in self.cluster_rst:
            tmp_group = sorted(group,key = lambda x:x[1],reverse = True)
            for elem in tmp_group:
                desc, cnt = elem 
                line = '%s\t%s\n' % (desc, cnt)
                rst_file.write(line)

            # blank line
            rst_file.write('\n\n')
        rst_file.close()

    def print_rst_clusters(self):
        rst_file = open('cluster_rst','w')
        user_num = 0
        for index, cluster in enumerate(self.cluster_rst):
            user_num += len(cluster)
            logging.debug('c:%d size:%d %s\n' % (index,len(cluster),cluster))
            rst_file.write('c:%d size:%d %s\n' % (index,len(cluster),cluster))
        logging.info("total cluster num:%d" % len(self.cluster_rst))
        logging.info("total user num:%d" % user_num)
        rst_file.close()



    def max_small_cluster_size(self,groups):
        max_size = max([len(g) for g in groups])
        return max_size

       #self.print_rst_clusters()
        

    def cluster_data(self, simifunc):
        read_words_data(self.words_file_name)
        self.clustering(data)
        #self.save_cluster2file()



def str_sim(str1, str2):
    """相似度=两句子的交集/两句子的并集"""
    #去掉数字
    #express_pat = re.compile("\[[^\]^\[]*\]")
    str1 = number_pt.sub("", unicode(str1))
    str2 = number_pt.sub("", unicode(str2))

    str1 = str1.translate(all_punctuation)
    str2 = str2.translate(all_punctuation)

    word_d = {}
    i = 0
    for s_t in str1:
        word_d[s_t] = 1
    for s_t in str2:
        if word_d.has_key(s_t):
            word_d[s_t] = 2
        else:
            word_d[s_t] = 1
    for w_t in word_d:
        if word_d[w_t] > 1:
            i += 1

    sim =    1.0 * i / len(word_d)
    return sim





def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
        return True
    else:
        return False

def contains_ch(str):
    for ch in unicode(str):
        if is_chinese(ch):
            return True
    return False



def read_user_data(data_file_name):
    data = []
    with open(data_file_name,'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 15:
                print parts
                continue
            data.append(parts)
    return data


weak_clusters = []
ids_set = set()
data = []


def read_words_data(file_name):
    with open(file_name,'r') as f:
        for line in f:
            parts  = line.strip().split('\t')
            if len(parts) != 2:
                print parts
                continue

            desc, cnt = parts
            try:
                cnt = int(cnt)  
            except Exception as e:
                print desc, cnt
            data.append((desc, cnt))






def reg_time_sim(t1_str,t2_str):
    try:
        t1 = datetime.strptime(t1_str,'%Y-%m-%d %H:%M:%S')
        t2 = datetime.strptime(t2_str,'%Y-%m-%d %H:%M:%S')
    except Exception as e:
        logging.info(t1_str)
        logging.info(t2_str)
        return 0

    diff = t1 - t2
    total_seconds = abs(diff.total_seconds())
    if total_seconds > 86400:
        return 0
    else:
        sim = 1 - (total_seconds/3600/24.0) ** 0.5
    return sim








def value_equal_sim(a,b):
    if a == 'NULL' or b == 'NULL' or a == '' or b == '':
        return 0
    if a == b:
        return 1
    else:
        return 0


def new_sim_func(a,b):
    #sim = str_sim(a[0],b[0])
    sim = str_sim(a,b)
    return sim 

 



def print_cluster(clustered_samples):
    """print cluster info"""
    logging.info('*** clustered samples ***')
    rst_file = open('cluster_rst','a')
    for index, cluster in enumerate(clustered_samples):
        cluser_size = len(cluster)
        if cluser_size > 1:
            #print 'C%d: size:%s' % (index, cluser_size)
            line = 'C%d: size:%s' % (index, cluser_size)
            rst_file.write(line)
            #print cluster
            rst_file.write("%s" % cluster)
            rst_file.write("\n\n")


def getopts():
    if len(sys.argv) < 3:
        print 'input err!'
        sys.exit()
    return sys.argv[1],sys.argv[2]



def cluste_user():
    threshold = 0.55
    words_file_name,  rst_file_name = getopts()

    hcluster = ClusteringUser(threshold, CLUSTER_SIMI_AVG, words_file_name,  rst_file_name)
    cluster_rst = hcluster.cluster_data(new_sim_func)
    hcluster.save_cluster2file()


def main():
    start_time = time.time()
    cluste_user()
    end_time = time.time()
    logging.info('total time cost:%s' % (end_time - start_time))


if __name__ == '__main__':
    main()
