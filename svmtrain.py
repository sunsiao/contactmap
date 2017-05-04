from sklearn.externals import joblib
from sklearn import svm
import numpy as np
import os
import ast
import random

feature = []
label = []
tmplist1 = []
tmplist2 = []

f = open(r'E:\protein\rawdata\feature\training\total.txt', 'r')
###分别读取为列表###
s = f.readlines()
for i in range(len(s)):
    tmplist1 = ast.literal_eval(s[i])  # 转成list
    feature += tmplist1
f.close()

g = open(r'E:\protein\rawdata\feature\training\label.txt', 'r')
t = g.readlines()
for j in range(len(t)):
    tmplist2 = ast.literal_eval(t[j])
    label += tmplist2
g.close()

X = feature
y = label
clf = svm.SVC(kernel= 'rbf',gamma= 0.025)
clf.fit(X,y)
joblib.dump(clf,"train_model.m")
