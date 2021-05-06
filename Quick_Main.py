from __future__ import division, unicode_literals
from sklearn import metrics
from sklearn.metrics import r2_score

#from textblob import TextBlob

import numpy as np
import glob
import json
import os
from sklearn.cluster import  KMeans
from sklearn.cluster import AgglomerativeClustering
import sortedcollections
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
import itertools





def info(eval_dir):
    path=os.path.join(eval_dir,"info.json")
    s=open(path,'r')
    data = json.load(s) 
    dict_file = {}
    for i in range(len(data)):
        dict_file[data[i][str("folder")]] = data[i][str("language")]  
    return dict_file



def clustering_word2vec(data_features,clustering_type='Agglomerative',optimizer_type = 'silhouette'):
    d={}
    # Finding the scores for cluster number ranging from 6 to Number of documents in a problem folder    
    for i in range(6,len(data_features)):
        if clustering_type == 'KMeans':
          spectral = KMeans(n_clusters=i).fit((np.array(data_features)))
        if clustering_type == 'Agglomerative':
          spectral = AgglomerativeClustering(n_clusters=i, linkage='ward').fit(np.array(data_features))

        label = spectral.fit_predict((np.array(data_features)))

        if optimizer_type == 'silhouette':
          score= metrics.silhouette_score((np.array(data_features)), label,metric='euclidean')
        if optimizer_type == 'calinski':
          score=metrics.calinski_harabasz_score((np.array(data_features)), label)

        d[i]=score

    # Finding the cluster Number with the highest score
    n_c=0   
    for key,val in d.items():
        if(val==max(d.values())):
            n_c=key
            break

    print("_"*100)
    print("Optimized Cluster Number: ", n_c)
    print("_"*100)

    # Finally choosing the optimized cluster Number for doing the final clustering 
    if clustering_type == 'KMeans':
      spectral == KMeans(n_clusters=n_c).fit((np.array(data_features)))
    if clustering_type == 'Agglomerative':
      spectral = AgglomerativeClustering(n_clusters=n_c, linkage='ward').fit(np.array(data_features))

    label = spectral.fit_predict((np.array(data_features)))

    return label
    
def prod_output(eval_dir,out_dir,k,labels):
    prblm_path=os.path.join(eval_dir,k)
    doc_path=glob.glob(prblm_path + '/*')
    doc_list_name=[]
    for i in doc_path:
        m=i.split('/')
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
        for key, value in dic.items():
            if val == value:
                list_per_cluster.append(key)
        list_all.append(list_per_cluster)
     
    list_all_output = []
    for i in range(len(list_all)):
        list_cluster = []
        for j in range(len(list_all[i])):
            dict_per_doc = {}
            dict_per_doc["document"] = list_all[i][j]
            list_cluster.append(dict_per_doc)
        list_all_output.append(list_cluster)
    if(os.path.exists(out_dir+"/"+k)==False):    
        os.mkdir(out_dir+"/"+k)    
    
    out_folder=out_dir + '/' +k
    out_path = out_folder + "/clustering.json"
    out_file = open(out_path, "w")

    json.dump(list_all_output, out_file, indent=4)
    return list_all_output   

def similarity_score(list_all, dict_features):
    list_all_comb = []
    for i in range(len(list_all)):
        list_comb_percluster = []
        if len(list_all[i]) > 1:
            for j in range(len(list_all[i])):
                list_comb_percluster.append(list_all[i][j]["document"])
            list_all_comb.append(list_comb_percluster)
    combs = []
    for i in range(len(list_all_comb)):
        comb = list(combinations(list_all_comb[i], 2))
        combs.append(comb)
    comb_list = list(itertools.chain(*combs))
    
    all_sim = []
    for i in range(len(comb_list)):
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
    for i in range(len(comb_list)):
        dict_sim_perpair = {}
        dict_sim_perpair["document1"] = comb_list[i][0]
        dict_sim_perpair["document2"] = comb_list[i][1]
        dict_sim_perpair["score"] = round(all_sim[i][0][0],6)
        list_all_output.append(dict_sim_perpair)
    out_folder=out_dir + '/' +k
    out_path = out_folder + "/ranking.json"
    out_file = open(out_path, "w")

    json.dump(list_all_output, out_file, indent=4)
    return list_all_output    




if __name__=="__main__":
    eval_dir="./quick_run_main/pan17-author-clustering-test-dataset-2017-03-14"
    out_dir="./quick_run_main/tfidf_weight_x_word2vec_output_pan17-author-clustering-test-dataset-2017-03-14"

    dict_f=info(eval_dir)  
    dict_f=sortedcollections.OrderedDict(sorted(dict_f.items()))

    en_=open("./quick_run_main/english_weighted.json",'r')
    dt_=open("./quick_run_main/dutch_weighted.json",'r')
    gr_=open("./quick_run_main/greek_weighted.json",'r')

    vec_en=json.load(en_)
    vec_dt=json.load(dt_)
    vec_gr=json.load(gr_)

    for k,v in dict_f.items():
        
        if v=="en":
            vectors=vec_en[k]
            labels=clustering_word2vec(vectors)
           
        
            
        
        elif v=="nl":
            vectors=vec_dt[k]
            labels=clustering_word2vec(vectors)
            
            
        elif v=="gr":
            vectors=vec_gr[k]
            labels=clustering_word2vec(vectors)
            
        list_all=prod_output(eval_dir,out_dir,k,labels)  

        dict_features={}

        prblm_path=os.path.join(eval_dir,k)
        doc_path=glob.glob(prblm_path + '/*')
        doc_list_name=[]
        for i in doc_path:
            m=i.split('/')
            m=m[-1]
            doc_list_name.append(m)
        i=0    
        for j in doc_list_name:
            dict_features[j]=vectors[i]
            i=i+1
        # similarity between documents
        list_comb, all_sim = similarity_score(list_all, dict_features)
        list_sim = write_ranking(list_comb, all_sim, out_dir,k)


