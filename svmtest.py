from sklearn.externals import joblib
import numpy
from sklearn import svm
import ast
import os

names = os.listdir(r'E:\protein\rawdata\feature\testing')
path1 = r'E:\protein\rawdata\feature\testing\{dir}\Posfeatures.txt'
path2 = r'E:\protein\rawdata\feature\testing\{dir}\Negfeatures.txt'

path3 = r'E:\protein\rawdata\feature\testing\{dir}\svmtest_posresult.txt'
path4 = r'E:\protein\rawdata\feature\testing\{dir}\svmtest_negresult.txt'


for name in names:
    f = open(path1.format(dir=name), 'r')
    s = f.readlines()
    tmplist1 = []
    inputposfeatures = []
    for i in range(len(s)):
        tmplist1 = ast.literal_eval(s[i])  # 转成list
        inputposfeatures.append(tmplist1)  # 得到了一条蛋白质中的contact residue pair的特征
    f.close()


    clf = joblib.load("train_model.m")
    numpy.savetxt(path3.format(dir=name),clf.predict(inputposfeatures))

########################################################
    tmplist2 = []
    inputnegfeatures = []
    h = open(path2.format(dir=name), 'r')
    t = h.readlines()
    for j in range(len(t)):
        tmplist2 = ast.literal_eval(t[i])
        inputnegfeatures.append(tmplist2)
    h.close()

    numpy.savetxt(path4.format(dir=name), clf.predict(inputnegfeatures))
