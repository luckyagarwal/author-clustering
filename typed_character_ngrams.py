#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 120 11:10:29 2018

@author: kartik
"""


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

if "__name__==__main__":
	doc=n_grams(review_l,doc_l)