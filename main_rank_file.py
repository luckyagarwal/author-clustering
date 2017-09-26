#main
import glob
import codecs
import json
import os
from nltk.tokenize  import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import  KMeans
import gensim
import sortedcollections
import numpy as np


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
    d=glob.glob(prblm_path + '/*')
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
    x=0    
    for i in words:
        temp=[]
        for j in i:
            if j not in stoplist:
              temp.append(j)
        words[x]=temp
        x=x+1    
    return words
    
def vec_fec(doc_bag,model):
    sen_vec=[]
    for sen in doc_bag:
        dummy_vec = np.zeros((300))
        for word in sen:
            try:
                dummy_vec += model[word]
            except:
                continue
        sen_vec.append(dummy_vec/len(sen))
    return sen_vec
    
def clustering(list_problem):
    vectorizer = TfidfVectorizer(analyzer="char", tokenizer=None, preprocessor=None, stop_words=None,
                                 max_features=5000, ngram_range=(3, 8))

    data_features=vectorizer.fit_transform(list_problem)
    spectral = KMeans(n_clusters=8).fit(data_features)
    label = spectral.fit_predict(data_features)
    return label


def clustering_word2vec(data_features):
    spectral = KMeans(n_clusters=8).fit(data_features)
    label = spectral.fit_predict(data_features)
    return label
    
def prod_output(eval_dir,out_dir,k,labels):
    prblm_path=os.path.join(eval_dir,k)
    doc_path=glob.glob(prblm_path + '/*')
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
    os.mkdir(out_dir+"/"+k)    
    out_folder=out_dir + '/' +k
    out_path = out_folder + "/clustering.json"
    out_file = open(out_path, "w")

    json.dump(list_all_output, out_file, indent=4)
    return list_all_output    
	
try:
    word = model1['bag']
    print 'using loaded model....'
except:
    model1 = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin",binary=True)    
    
model2 = gensim.models.word2vec.Word2Vec.load('nl-word2vec-model-300-word.bin')

eval_dir="pan"
out_dir="problem"
dict_f=info(eval_dir)  
dict_f=sortedcollections.OrderedDict(sorted(dict_f.items()))   
for k,v in dict_f.iteritems():
    doc_list=merge_prblm_docs(eval_dir,k)
    if v=="en":
       review_en=pre_process_w2v(doc_list)
       review_en=stopwords_r(review_en,v)
       vectors_en=vec_fec(review_en,model1)
       labels=clustering_word2vec(vectors_en)
       
    elif v=="nl":
       review_nl=pre_process_w2v(doc_list)
       review_nl=stopwords_r(review_nl,v)
       vectors_nl=vec_fec(review_nl,model2)
       labels=clustering_word2vec(vectors_nl)
       
    elif v=="gr":
        labels=clustering(doc_list)
        
    final_list=prod_output(eval_dir,out_dir,k,labels)  
    
    
        
        
        
    