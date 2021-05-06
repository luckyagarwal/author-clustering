# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 13:04:50 2017

@author: Lucky and Kartik
"""
import glob
import os
import sortedcollections
from itertools import combinations
import itertools
import json
from nltk.tokenize  import word_tokenize
import codecs

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
    words=[word_tokenize(i) for i in doc]
    return words

def func(doc_list,new_list):
    word_bag=[]
    final_list=[]
    for i in new_list:
        final_list.extend(i)
    review_1=pre_process_w2v(final_list)
    for i in review_1:
        word_bag.extend(i)
        word_set=set(word_bag)
        word_list=list(word_set)
    dict_word={}
    for i in word_list:
        dict_word[i]=0
    for i in word_bag:
        for k,v in dict_word.iteritems():
            if k==i:
                dict_word[k]=dict_word[k]+1
    return dict_word
eval_dir="pan17-author-clustering-training-dataset-2017-02-15"
dict_f=info(eval_dir)  
dict_f=sortedcollections.OrderedDict(sorted(dict_f.items()))
new_list_en=[]  
new_list_nl=[]  
new_list_gr=[]  

for k,v in dict_f.iteritems():
    if(v=='en'):
        doc_list_en=merge_prblm_docs(eval_dir,k)
        new_list_en.append(doc_list_en)
    if(v=='nl'):
        doc_list_nl=merge_prblm_docs(eval_dir,k)
        new_list_nl.append(doc_list_nl)
    if(v=='gr'):
        doc_list_gr=merge_prblm_docs(eval_dir,k)
        new_list_gr.append(doc_list_gr)
    
dict_word_en=func(doc_list_en,new_list_en)
dict_word_nl=func(doc_list_nl,new_list_nl)
dict_word_gr=func(doc_list_gr,new_list_gr)  
#dict_f=sortedcollections.OrderedDict(sorted(dict_word_en.items()))  
