# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 12:30:18 2018

"""
#main

    from __future__ import division, unicode_literals
    from sklearn import metrics
    from sklearn.metrics import r2_score
    from textblob import TextBlob as tb
    import numpy as np
    import sys
    import glob
    import codecs
    import json
    import os
    from nltk.tokenize  import word_tokenize
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import  KMeans
    from sklearn.cluster import AgglomerativeClustering
    import gensim
    import sortedcollections
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    from itertools import combinations
    import itertools
    import math 
    from gensim.models import LogEntropyModel
    from gensim.test.utils import common_texts
    from gensim.corpora import Dictionary

def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)

def info(eval_dir):
    path=os.path.join(eval_dir,"info.json")
    s=open(path,'r')
    data = json.load(s) 
    dict_file = {}
    for i in xrange(len(data)):
        dict_file[data[i][str("folder")]] = data[i][str("language")]  
    return dict_file

def merge_prblm_docs(eval_dir,k):
    prblm_path=os.path.join(eval_dir,k)
    d=glob.glob(prblm_path + '\\*')
    doc=[]
    for i in d:
        f=codecs.open(i, 'r', 'utf-8')
        text=f.read()
        doc.append(text)
        f.close()  
    return doc

def pre_process_w2v(doc):
    doc=[i.lower() for i in doc]
    x=0
    for i in doc:
        temp=''
        for j in i:
            if j.isalpha() or j==' ' or j=="'":
                temp+=j
        doc[x]=temp
        x=x+1
    return doc
    
def  stopwords_r(doc,lang):
    words=[word_tokenize(i) for i in doc]
    if lang == "en":
        with open("stopwords/stopwords_en.txt") as f:
            stoplist = f.readlines()
            stoplist = [x.strip('\n') for x in stoplist]
        
    elif lang == "nl":
        with open("stopwords/stopwords_nl.txt") as f:
            stoplist = f.readlines()
            stoplist = [x.strip('\n') for x in stoplist]
    elif lang == "gr":
        with open("stopwords/stopwords_gr.txt") as f:
            stoplist = f.readlines()
            stoplist = [x.strip('\n') for x in stoplist]        
    x=0    
    for i in words:
        temp=[]
        for j in i:
            if j not in stoplist:
              temp.append(j)
        words[x]=temp
        x=x+1    
    return words        
    
def vec_fec(doc_bag,model,review_list):
    sen_vec=[]
    review_list=[tb(sentence) for sentence in review_list]
    for sen,blob in zip(doc_bag,review_list):
        dummy_vec = np.zeros((300))
        for word in sen:
            try:
               #new code added 
                
                t=tfidf(word,blob,review_list)
                x=((model[word])*t)
                #print x
                dummy_vec +=x
            except:
                pass
        sen_vec.append(dummy_vec/len(sen))
        
    sen_vec=[i for i in sen_vec]    
    return sen_vec
    

def clustering_word2vec(data_features):
    d={}    
    for i in range(6,len(data_features)):
        #spectral = KMeans(n_clusters=i).fit((np.array(data_features)))
        spectral = AgglomerativeClustering(n_clusters=i, linkage='ward').fit(np.array(data_features))
        label = spectral.fit_predict((np.array(data_features)))
        #score=metrics.calinski_harabaz_score((np.array(data_features)), label)
        score= metrics.silhouette_score((np.array(data_features)), label,metric='euclidean')
        d[i]=score
    n_c=0   
    for key,val in d.iteritems():
        if(val==max(d.values())):
            n_c=key
            break
    print n_c
    spectral = AgglomerativeClustering(n_clusters=n_c, linkage='ward').fit(np.array(data_features))
    #spectral = KMeans(n_clusters=n_c).fit(data_features)        
    label = spectral.fit_predict((np.array(data_features)))
    return label
    
def prod_output(eval_dir,out_dir,k,labels):
    prblm_path=os.path.join(eval_dir,k)
    doc_path=glob.glob(prblm_path + '\\*')
    doc_list_name=[]
    for i in doc_path:
        m=i.split('\\')
        m=m[-1]
        doc_list_name.append(m)
    dic={}
    for i,j in zip(doc_list_name,labels):
        dic[i]=j 
        
    list_all = []
    list_val = []
    for v in dic.values():
        list_val.append(v)
        set_val = set(list_val)
    for val in set_val:
        list_per_cluster = []
        for key, value in dic.iteritems():
            if val == value:
                list_per_cluster.append(key)
        list_all.append(list_per_cluster)
     
    list_all_output = []
    for i in xrange(len(list_all)):
        list_cluster = []
        for j in xrange(len(list_all[i])):
            dict_per_doc = {}
            dict_per_doc["document"] = list_all[i][j]
            list_cluster.append(dict_per_doc)
        list_all_output.append(list_cluster)
    if(os.path.exists(out_dir+"\\"+k)==False):    
        os.mkdir(out_dir+"\\"+k)    
    
    out_folder=out_dir + '\\' +k
    out_path = out_folder + "\\clustering.json"
    out_file = open(out_path, "w")

    json.dump(list_all_output, out_file, indent=4)
    return list_all_output   
    
def similarity_score(list_all, dict_features):
    list_all_comb = []
    for i in xrange(len(list_all)):
        list_comb_percluster = []
        if len(list_all[i]) > 1:
            for j in xrange(len(list_all[i])):
                list_comb_percluster.append(list_all[i][j]["document"])
            list_all_comb.append(list_comb_percluster)
    combs = []
    for i in xrange(len(list_all_comb)):
        comb = list(combinations(list_all_comb[i], 2))
        combs.append(comb)
    comb_list = list(itertools.chain(*combs))
    all_sim = []
    for i in xrange(len(comb_list)):
        doc1 = comb_list[i][0].split(",")
        doc2 = comb_list[i][1].split(",")
        vec1 = dict_features[doc1[0]]
        vec2 = dict_features[doc2[0]]
        vec1=[vec1]
        vec2=[vec2]
        sim = cosine_similarity(vec1, vec2)
        all_sim.append(sim)
    return comb_list, all_sim

def write_ranking(comb_list, all_sim, out_dir,k):
    list_all_output = []
    for i in xrange(len(comb_list)):
        dict_sim_perpair = {}
        dict_sim_perpair["document1"] = comb_list[i][0]
        dict_sim_perpair["document2"] = comb_list[i][1]
        dict_sim_perpair["score"] = round(all_sim[i][0][0],6)
        list_all_output.append(dict_sim_perpair)
    out_folder=out_dir + '\\' +k
    out_path = out_folder + "\\ranking.json"
    out_file = open(out_path, "w")

    json.dump(list_all_output, out_file, indent=4)
    return list_all_output    

     
# the Below code is to check if the model has been loaded or not
try:
    word1 = model1['bag']
    word2 = model2['slechte']
    word3 = model3[u'φύλο']
    print 'using loaded model....'
except:
    pass
    model1 = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz",binary=True)    
    model2=gensim.models.KeyedVectors.load_word2vec_format("wiki.nl.bin", binary=True)
    model3=gensim.models.KeyedVectors.load_word2vec_format("wiki.el.bin", binary=True)
    

eval_dir="pan17-author-clustering-test-dataset-2017-03-14"
out_dir="tfidf_weight_x_word2vec_output_pan17-author-clustering-test-dataset-2017-03-14"
dict_f=info(eval_dir)  
dict_f=sortedcollections.OrderedDict(sorted(dict_f.items()))
#checkpoint 1    
for k,v in dict_f.iteritems():
    doc_list=merge_prblm_docs(eval_dir,k)
  
    if v=="en":
        review_en=pre_process_w2v(doc_list)
        review_en_new=stopwords_r(review_en,v)
        vectors=vec_fec(review_en_new,model1,review_en)
        labels=clustering_word2vec(vectors)
        
       
    
    elif v=="nl":
        review_nl=pre_process_w2v(doc_list)      
        review_nl_new=stopwords_r(review_en,v)
        vectors=vec_fec(review_nl_new,model2,review_nl)
        labels=clustering_word2vec(vectors)
         
    elif v=="gr":
        review_gr=pre_process_w2v(doc_list)
        vectors=vec_fec(review_gr_new,model3,review_gr)
        labels=clustering_word2vec(vectors)

    list_all=prod_output(eval_dir,out_dir,k,labels)  

    dict_features={}

    prblm_path=os.path.join(eval_dir,k)
    doc_path=glob.glob(prblm_path + '\\*')
    doc_list_name=[]
    for i in doc_path:
        m=i.split('\\')
        m=m[-1]
        doc_list_name.append(m)
    i=0    
    for j in doc_list_name:
        dict_features[j]=vectors[i]
        i=i+1
    # similarity between documents
    list_comb, all_sim = similarity_score(list_all, dict_features)
    list_sim = write_ranking(list_comb, all_sim, out_dir,k)
