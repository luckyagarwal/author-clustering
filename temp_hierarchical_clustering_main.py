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
    
def clustering(list_problem):
   
   vectorizer = TfidfVectorizer(analyzer="char", tokenizer=None, preprocessor=None, stop_words=None,
                                 max_features=5000,min_df=2,ngram_range=(2, 8))

   data_features=vectorizer.fit_transform(list_problem)
   data_features = data_features.toarray()
   #spectral = KMeans(n_clusters=8).fit(data_features)
   d={}    
   for i in range(5,len(data_features)):
       spectral = KMeans(n_clusters=i).fit(np.array(data_features))
       #spectral = AgglomerativeClustering(n_clusters=i, linkage='ward').fit(np.array(data_features))
       label = spectral.fit_predict(np.array(data_features))
       score=metrics.calinski_harabaz_score((np.array(data_features)), label)
       #score= metrics.silhouette_score(data_features, label, metric='euclidean')
       d[i]=score
   n_c=0   
   for key,val in d.iteritems():
       if(val==max(d.values())):
           n_c=key
           break
   print n_c
   spectral = KMeans(n_clusters=n_c).fit(np.array(data_features))
   #spectral = AgglomerativeClustering(n_clusters=n_c, linkage='ward').fit(np.array(data_features))
   label = spectral.fit_predict(np.array(data_features))
   return data_features,label


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
def merge_word2_vec_Tfidf(review_en,vectors):
     vectorizer = TfidfVectorizer(analyzer="char", tokenizer=None, preprocessor=None, stop_words=None,
                                 max_features=5000, ngram_range=(2, 8))
     data_features=vectorizer.fit_transform(review_en)
     data_features=data_features.toarray()
     for i,j  in zip(vectors,data_features):
         i.extend(list(j))
     
     return vectors   
     
def tfidf_weight_doc(review_list):
    review_list=[tb(sentence) for sentence in review_list]
    vector=[]
    for i,blob in enumerate(review_list):
        x=[]
        for word in blob.words:
            t=tfidf(word,blob,review_list)
            x.append(t)
        vector.append(x)  
    length = len(sorted(vector,key=len, reverse=True)[0])
    vector=np.array([xi+[0]*(length-len(xi)) for xi in vector])          
    return vector   

def log_entropy(doc):
    doc=pre_process_w2v(doc)
    doc_token=[i.split() for i in doc]
    dct = Dictionary(doc_token)
    corpus = [dct.doc2bow(row) for row in doc]
    model = LogEntropyModel(corpus)
    vec=[]    
    for row in corpus:
        vector = model[corpus[row]]
        x=[]
        for t in vector:
            x.append(t[1])
        vec.append(x)
        
    length = len(sorted(vec,key=len, reverse=True)[0])
    vec=np.array([xi+[0]*(length-len(xi)) for xi in vec]) 
    return vec    
def affix_prefix(sen):
   #affix:prefix
    a_p=[]
    sen=sen.strip('\n')
    for i in sen.split():
        if(len(i)>3):
            a_p.append(i[0:3])
    return a_p
        
def affix_suffix(sen):
    #affix:suffix
    a_s=[]            
    sen=sen.strip('\n')
    for i in sen.split():
        if(len(i)>3):
            a_s.append(i[-3:len(sen)])
    return a_s            

def affix_space_prefix(sen):    #affix:space-prefix(without preprocess doc_list)
    a_s_p=[]            
    count=0
    temp=''
    sen=sen.strip('\n')
    for i in sen: 
        if count!=len(sen)-1:
            if i==' ' and sen[count+1].isalpha() and sen[count+2].isalpha():
                temp='_'
                temp=temp+sen[count+1]
                temp=temp+sen[count+2]
                #append
                a_s_p.append(temp)
                temp=''
        count+=1
    return a_s_p

def affix_space_suffix(sen):        
    #affix:space-suffix(without preproess doc list)
    a_s_s=[]        
    count=0
    temp=''
    sen=sen.strip('\n')
    for i in sen: 
        if count!=len(sen)-1:
            if i==' 'and sen[count-1].isalpha() and sen[count-2].isalpha():
                temp=temp+sen[count-2]
                temp=temp+sen[count-1]
                temp=temp+'_'
                a_s_s.append(temp)
                temp=''
        count+=1    
    return a_s_s    

def punct_beg(sen):
    #punct:beg-punct(without preprocess  doc list)        
    p_b=[]        
    count=0
    temp=''
    sen=sen.strip('\n')
    for i in sen:
        if count<len(sen)-2:
            if (i.isalpha()==False and i!=' ')  and ((sen[count+1].isalpha())==True or (sen[count+1]==' ')):
                temp=temp+i
                if (sen[count+1].isalpha())==True:
                    temp=temp+sen[count+1]
                elif (sen[count+1]==' '):
                    temp=temp+'_'
            
                temp=temp+sen[count+2]
                p_b.append(temp)
                temp=''
        count+=1
    return p_b

def punct_mid(sen):    
    #punct:mid-punct(without preprocess doc list)
    p_m=[]
    count=0
    temp=''
    sen=sen.strip('\n')
    for i in sen:
        if count!=len(sen)-1:
            if (i.isalpha())==False and i!=' ':
                if sen[count-1]==' ':
                    temp=temp+'_'
                else:
                    temp=temp+sen[count-1]
                
                temp=temp+i
                
                if sen[count+1]==' ':
                    temp=temp+'_'
                else:
                    temp=temp+sen[count+1]
                p_m.append(temp)
                temp=''
        count+=1
    return p_m            
                
def punct_end(sen):                    
    #punct:end-punct(without preprocess doc list)
    p_e=[]    
    count=0
    temp=''
    sen=sen.strip('\n')
    for i in sen:
        if (i.isalpha()==False and i!=' ') and (sen[count-2].isalpha()==True) and (sen[count-1].isalpha()==True):
            temp=temp+sen[count-2]
            temp=temp+sen[count-1]
            temp=temp+i
            p_e.append(temp)
            temp=''
        count+=1
    return p_e  
#untyped character n grams
def u_n_gram(sen):
    u_n_gram=[]
    sen=sen.strip('\n')
    for i in range(3,8):
        count=0
        while(count<len(sen)-i):
            u_n_gram.append(sen[count:count+i].replace(' ','_'))
            count=count+i
    return u_n_gram            
            
                  
def n_grams(review_l,doc_l):
    doc_n=[]
    for review,doc in zip(review_l,doc_l):
        n_gram=[]
        n_gram.extend(affix_prefix(review))
        n_gram.extend(affix_suffix(review))
        n_gram.extend(affix_space_prefix(doc))
        n_gram.extend(affix_space_suffix(doc))
        n_gram.extend(punct_beg(doc))
        n_gram.extend(punct_mid(doc))
        n_gram.extend(punct_end(doc))
        n_gram.extend(u_n_gram(doc))
        
        doc_n.append(n_gram)
    return doc_n
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
        #print review_en_new
        #vectors,labels=clustering(doc_list)  #tfidf_ 1
        vectors=vec_fec(review_en_new,model1,review_en)
       #vectors=merge_word2_vec_Tfidf(review_en,vectors)
       #vectors=tfidf_weight_doc(review_en)
       #vectors=[np.array(i) for i in vectors]
       #vectors=np.array(vectors)
        labels=clustering_word2vec(vectors)
        #print 
       
    
    elif v=="nl":
        review_nl=pre_process_w2v(doc_list)
        #review_nl_new=stopwords_r(review_nl,v)
        #print review_nl_new        
        n_gram_list=n_grams(review_nl,doc_list)
        vectors,labels=clustering(doc_list)        
       #review_nl=pre_process_w2v(doc_list)
       #vectors,labels=clustering(doc_list)
       #review_nl_new=stopwords_r(review_nl,v)
        #
        #vectors=vec_fec(review_nl_new,model2,review_nl)
       #vectors=merge_word2_vec_Tfidf(review_nl,vectors)
       #vectors=tfidf_weight_doc(review_nl)
        #labels=clustering_word2vec(vectors)
        #print vectors
        break    
    elif v=="gr":
        review_gr=pre_process_w2v(doc_list)
        vectors,labels=clustering(doc_list)
       #review_gr=pre_process_w2v(doc_list)
       #vectors,labels=clustering(doc_list)
        review_gr_new= [word_tokenize(i) for i in doc_list]
        vectors=vec_fec(review_gr_new,model3,review_gr)
       #vectors=merge_word2_vec_Tfidf(review_gr,vectors)
       #vectors=tfidf_weight_doc(review_gr)
        labels=clustering_word2vec(vectors)
        #print vectors
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
