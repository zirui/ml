#/usr/bin/python
#encoding:utf8

import os
import jieba
import jieba.posseg as pseg
import sys
import string
import numpy as np
from sklearn import feature_extraction
from sklearn.cluster import AffinityPropagation
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import metrics
import time




reload(sys)

sys.setdefaultencoding('utf8')



def cut_words(src_file) :
    rst = []
    with open(src_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                print 'invalid line'
                print line
                continue

            sentence, freq = parts
            #对文档进行分词处理，采用默认模式
            seg_list = jieba.cut(sentence,cut_all=False)

            tokens = []
            for seg in seg_list :
                seg = ''.join(seg.split())
                if (seg != '' and seg != "\n" and seg != "\n\n") :
                    tokens.append(seg)
                #print result
            rst.append((sentence,freq,tokens))

    return rst





def gen_tf_feature(sentence_data):
    corpus = [] 
    for data in sentence_data:
        token_list = data[2]
        line = ' '.join(token_list)
        corpus.append(line)


    ch_stop_words = [u'左',u'右',u'侧',u'病',u'症','伤']
    vectorizer = CountVectorizer(binary=True, stop_words=ch_stop_words)    
    X = vectorizer.fit_transform(corpus)
    
    print 'Size of fea_train:' + repr(X.shape) 
    print'doc num: %s' %  X.toarray()[0:3]
    print 'The average feature sparsity is {0:.3f}%'.format(X.nnz/float(X.shape[0]*X.shape[1])*100); 

    return X

    #tv2 = TfidfVectorizer(vocabulary = tv.vocabulary_); 
    #kmeans_clustering(corpus, X,len(corpus)/5, rst_file)
    #Affinity_Propagation_clutering(corpus, X, rst_file)
    #hier_clustering(corpus, X, len(corpus)/5, rst_file) #dense array .toarray()
    #hier_clustering(corpus, X.toarray(), len(corpus)/5, rst_file) #dense array .toarray()



def kmeans_clustering(sentence_data, X, cluster_num, rst_file):
    #if 1:
    if 0:
        km = MiniBatchKMeans(n_clusters=cluster_num, init='k-means++', n_init=1,
                             init_size=1000, batch_size=1000, verbose=1)
    else:
        km = KMeans(n_clusters=cluster_num, init='k-means++', max_iter=200, n_init=1,
                    verbose=1)


    t0 = time.time()
    km.fit(X)
    print("clustering done in %0.3fs" % (time.time() - t0))



    print 'write clustering rst to file...'
    with open(rst_file, 'w') as f:
        for i in xrange(0,len(km.labels_)):
            label = km.labels_[i]
            sentence = sentence_data[i][0]
            freq = sentence_data[i][1]
            line = '%s\t%s\t%s\n' % (label, sentence, freq)
            f.write(line)



def hier_clustering(corpus, X, cluster_num, rst_file):
    ag = AgglomerativeClustering(linkage='average',n_clusters=cluster_num, affinity= 'cosine')

    print("clustering..." )
    t0 = time.time()
    ag.fit(X)
    print("done in %0.3fs" % (time.time() - t0))


    print 'write clustering rst to file...'
    with open(rst_file, 'w') as f:
        for i in xrange(0,len(ag.labels_)):
            label = ag.labels_[i]
            sentence = corpus[i]
            line = '%s\t%s\n' % (label, sentence)
            f.write(line)




def Affinity_Propagation_clutering(corpus, X,  rst_file):
    af = AffinityPropagation(preference=-50).fit(X)

    print 'write clustering rst to file...'
    with open(rst_file, 'w') as f:
        for i in xrange(0,len(af.labels_)):
            label = af.labels_[i]
            sentence = corpus[i]
            line = '%s\t%s\n' % (label, sentence)
            f.write(line)





def dbscan_clustering(X):
    """
    # Compute DBSCAN
    db = DBSCAN(eps=0.3, min_samples=10).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: %d' % n_clusters_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels))
    """



def hiclustering(X):
    #clustering = AgglomerativeClustering(linkage=linkage, n_clusters=10)
    t0 = time.time()
    #clustering.fit(X)
    #print("%s : %.2fs" % (linkage, time() - t0))





def tf_idf(file_name, rst_file):
    corpus = [] 
    with open(file_name, 'r') as f:
        for line in f:
            line = line.strip()
            corpus.append(line)

    #vectorizer = CountVectorizer()    
    #transformer = TfidfTransformer()
    #tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))

    #word = vectorizer.get_feature_names() #所有文本的关键字
    #weight = tfidf.toarray()              #对应的tfidf矩阵
    #vectorizer = TfidfVectorizer(max_df=0.5, min_df=2,use_idf=True)
    vectorizer = TfidfVectorizer(use_idf=True)
    X = vectorizer.fit_transform(corpus)
    



    #print'word num: %s' %  len(X)
    print'doc num: %s' %  X.shape[0]
    print X
    #print'doc num: %s' %  len(weight)
    """
    print word[:10]
    print weight[0:3]
    print len(weight[0])
    return
    """

    """
    doc_num = len(weight)
    with open(rst_file, 'w') as rst_file:
        for i in range(len(weight)):
            if i % 1000 == 0:
                ratio =  1.0 * i/doc_num
                print '%s/%s %s' % (i, doc_num, ration) 
            rst_file.write('%s\t' % i)
            for j in range(len(word)):
                line += '%s\t' % weight[i][j]
            line = line.rstrip()
            line += '\n'
            rst_file.write(line)
    """


    clustering_words(corpus, X)


def clustering_words(corpus,doc_vectors):
    start = time.time() # Start time

    # Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
    # average of 5 words per cluster
    num_clusters = doc_vectors.shape[0] / 5
    #k_means = cluster.KMeans(k=3)

    # Initalize a k-means object and use it to extract centroids
    #kmeans_clustering = KMeans( n_clusters = num_clusters )
    km = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=1, verbose=False)
    km.fit(doc_vectors) 
    labels = km.labels_
    print labels[0:10]

    # Get the end time and print how long the process took
    end = time.time()
    elapsed = end - start
    print "Time taken for K Means clustering: ", elapsed, "seconds."




def clustering_sentence(src_file, rst_file):
    print 'cut words...'
    sentence_data = cut_words(src_file)

    print 'extract features...'
    X = gen_tf_feature(sentence_data)

    print 'clustering sentence...'
    kmeans_clustering(sentence_data, X, len(sentence_data)/5, rst_file)





def getopts():
    if len(sys.argv) < 3:
        print 'invalid input'
        sys.exit()
    return sys.argv[1], sys.argv[2]



def main():
    src_file, rst_file = getopts()
    #tf_idf(src_file, rst_file)
    clustering_sentence(src_file, rst_file)



if __name__ == '__main__':
    main()
