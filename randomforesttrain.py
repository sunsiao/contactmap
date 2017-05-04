from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import ast


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
clf = RandomForestClassifier(n_estimators=10)
clf.fit(X,y)
joblib.dump(clf,"foresttrain_model.m")
