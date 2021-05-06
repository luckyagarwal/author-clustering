#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 12:18:34 2018

@author: kartik
"""

import numpy
import json

json1_file=open("./result/out.json")
json1_str=json1_file.read()
# english
eng_f=[]
eng_r=[]
eng_p=[]
eng_avg_pre=[]

for i in range(40):
    json1_data=json.loads(json1_str)[i]
    eng_f.append(json1_data['F-Bcubed'])
    eng_r.append(json1_data['R-Bcubed'])
    eng_p.append(json1_data['P-Bcubed'])
    eng_avg_pre.append(json1_data['Av-Precision'])
    
e_f=mean_avg_eng=(sum(eng_f))/40    
e_r=sum(eng_r)/40
e_p=sum(eng_p)/40
e_ap=sum(eng_avg_pre)/40

print "ENGLISH"
print ''
print "AVG-F-Bcubed: ",e_f
print "AVG-P-Bcubed: ",e_r
print "AVG-R-Bcubed: ",e_p
print "AVG-PRECISION: ",e_ap

dutch_f=[]
dutch_r=[]
dutch_p =[]
dutch_avg_pre=[]

for i in range(40,80):
    json1_data=json.loads(json1_str)[i]
    dutch_f.append(json1_data['F-Bcubed'])
    dutch_r.append(json1_data['R-Bcubed'])
    dutch_p.append(json1_data['P-Bcubed'])
    dutch_avg_pre.append(json1_data['Av-Precision'])
    
d_f=mean_avg_dutch=(sum(dutch_f))/40
d_r=sum(dutch_p)/40
d_p=sum(dutch_r)/40
d_ap=sum(dutch_avg_pre)/40

print ''
print "DUTCH"
print ''
print "AVG-F-Bcubed: ",d_f
print "AVG-P-Bcubed: ",d_r
print "AVG-R-Bcubed: ",d_p
print "AVG-PRECISION: ",d_ap

greek_f=[]
greek_r=[]
greek_p=[]
greek_avg_pre=[]

for i in range(80,120):
    json1_data=json.loads(json1_str)[i]
    greek_f.append(json1_data['F-Bcubed'])
    greek_r.append(json1_data['R-Bcubed'])
    greek_p.append(json1_data['P-Bcubed'])
    greek_avg_pre.append(json1_data['Av-Precision'])

g_f=mean_avg_greek=(sum(greek_f))/40    
g_r=sum(dutch_r)/40
g_p=sum(dutch_p)/40
g_ap=sum(dutch_avg_pre)/40

print ''
print "GREEK"
print ''
print "AVG-F-Bcubed: ",g_f
print "AVG-P-Bcubed: ",g_r
print "AVG-R-Bcubed: ",g_p
print "AVG-PRECISION: ",g_ap