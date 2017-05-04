def features(name):
    import re
    import math
    import numpy as np
    import scipy.stats.stats as stats
    from collections import Counter



    path1 = r'E:\protein\rawdata\{dir}\{dir}.aln'
    path2 = r'E:\protein\rawdata\{dir}\{dir}.horiz'
    path3 = r'E:\protein\rawdata\{dir}\{dir}.solv'
    path4 = r'E:\protein\rawdata\{dir}\{dir}.pdb'
    path5 = r'E:\protein\rawdata\{dir}\Posfeatures.txt'
    path6 = r'E:\protein\rawdata\{dir}\Negfeatures.txt'

    ProDict = ['G', 'A', 'V', 'L', 'I', 'P', 'M', 'F', 'W', 'S', 'T', 'N', 'Q', 'C', 'Y', 'D', 'E', 'K', 'R', 'H', '-']  # 共有20种残基，加上一个'-';这样排列是为了
    # 之后提取residue type feature方便;
    f = open(path1.format( dir = name), 'r')  # 注意，路径之前要有一个'r',否则路径出错;
    mulsequence = f.readlines()  # 把所有的序列读入，每条序列之后有字符'/n'隔开;
    Seqnum = len(mulsequence)  # 序列比对中共有Seqnum条序列;
    Seqlen = len(mulsequence[0]) - 1  # 目标序列的长度;

    for i in range(Seqnum):
        if 'X' in mulsequence[i]:
            mulsequence[i] = mulsequence[i].replace('X', '-')
        if 'B' in mulsequence[i]:
            mulsequence[i] = mulsequence[i].replace('B', '-')
        if 'J' in mulsequence[i]:
            mulsequence[i] = mulsequence[i].replace('J', '-')
        if 'O' in mulsequence[i]:
            mulsequence[i] = mulsequence[i].replace('O', '-')
        if 'U' in mulsequence[i]:
            mulsequence[i] = mulsequence[i].replace('U', '-')
        if 'Z' in mulsequence[i]:
            mulsequence[i] = mulsequence[i].replace('Z', '-')

    ###把序列比对中的X替换为'-',作为'-'处理####

    Tlist = []  # 用于存储所有的残基处的特征
    Temlist = []  # 用来存储当前列；
    for n in range(Seqlen):
        for m in range(Seqnum):
            Temlist.append((mulsequence[m][n]))  # 得到了Temlist,其中存储了第n列的所有残基
        ## 通过比对ProDict列表，得到每一种残基在Temlist中的比重，即概率（包括'-'）  ##
        p = []  # 用于存储概率
        sum = 0
        for i in range(21):
            for j in range(Seqnum):
                if ProDict[i] == Temlist[j]:
                    sum = sum + 1
            p.append(sum / Seqnum)
            sum = 0
        ###得到了一列概率，顺序为ProDict中的顺序###
        Temlist = []
        Tlist.append(p)
    f.close()  # 关闭序列比对文件，释放内存
    ###Tlist里存储了全部Seqlen列概率，Tlist[i]表示目标序列i位置处的特征（现在只包含21位的概率）###
    ###
    ###以下处理二级结构、相对溶液可及性###

    ##提取文件中包含二级结构的行，并构成一个列表，其顺序与残基序列对应;模式：以Pred+空格开头，以'\n'结尾（正则表达式）###
    f = open(path2.format(dir=name), 'r')
    s = str(f.readlines())  # 转化为字符串，方便处理
    list2 = re.findall(r'Pred: (.+?)\\n', s)  # 正则表达式，提取需要的行
    horiz = ''
    for i in list2:
        horiz = horiz + i  # 将需要的行整合为一行数据
    for i in range(Seqlen):
        if horiz[i] == 'H':
            Tlist[i].extend([1, 0, 0])
        elif horiz[i] == 'E':
            Tlist[i].extend([0, 1, 0])
        else:
            Tlist[i].extend([0, 0, 1])
    f.close()
    ##二级结构特征已经加入，下面处理相对溶液可及性，读取文件，提取第三列（行号与残基序列一一对应）#
    # 对第三列数据，以0.25为界，大于0.25位exposed，编码10，否则为buried，编码01 #
    f = open(path3.format( dir = name), 'r')
    solv = [float(l.split()[2]) for l in f]
    for i in range(Seqlen):
        if solv[i] >= 0.25:
            Tlist[i].extend([1, 0])
        else:
            Tlist[i].extend([0, 1])
    f.close()
    ###二级结构与相对溶液可及性提取完毕，下面提取熵###

    entropy = 0

    for i in range(Seqlen):
        for j in range(21):
            if Tlist[i][j] > 0:
                entropy -= (math.log(Tlist[i][j])) * (Tlist[i][j])
        Tlist[i].append(entropy)
        entropy = 0
    ###熵提取完毕###

    #######27位的特征提取完毕，存入了Tlist之中；Tlist[i]表示残基序列i位置处的特征,Tlist[i][0:20]表示概率#######


    # 生成标签#
    #######读取pdb文件，根据3、6、7、8列数据，计算residue pair之间的欧氏距离，找到contact的residue pairs(i,j)并保存#######

    f = open(path4.format( dir = name), 'r')
    CorDict = []
    number = 0
    for i in f:
        tmp = i.split()
        if (tmp[2] == 'CB') or (tmp[2] == 'CA' and tmp[3] == 'GLY'):  # 注意，tmp[5]-2才是残基的序号（从0开始）
            # print (tmp[2],tmp[5],tmp[6:9])
            tmplist = [float(tmp[6]), float(tmp[7]), float(tmp[8])]
            CorDict.append(tmplist)
            number = number + 1
            ###得到了所有残基Ca原子的三维坐标；CorDict是个二维列表，CorDict[i][0:3]是三维坐标，下面求出所有可能的(i,j)对之间的欧氏距离，###
            # 并与8比较，从而得到所有contact residue pairs###
    Conlist = []  # 记录contac的残基对(i,j),元素格式为(i, j)
    Noclist = []  # 记录不contact的残基对(i,j),元素格式为(i,j)

    for i in range(Seqlen - 7):
        if mulsequence[0][i] != '-':
            for j in range(i + 7, Seqlen):
                if mulsequence[0][j] != '-':
                    if ((CorDict[i][0] - CorDict[j][0]) ** 2 + (CorDict[i][1] - CorDict[j][1]) ** 2 + (CorDict[i][2] - CorDict[j][2]) ** 2) < 64:
                        Conlist.append((i, j))
                    else:
                        Noclist.append((i, j))
    f.close()
    #####
    #####得到了contact的(i,j)和不contact的(i,j)，分别保存在两个list中，之后提取完特征之后，用于加标签(label)#####


    #######以下，local windows feature,首先选择目标残基对(i,j),注意i为从0到(Seqlen-7)，j为(i+7)到Seqlen;若窗口越界则填充0#######
    # f = open(r'E:\protein\rawdata\1atgA0\1atgA0.aln', 'r') #以下的目的，得到(i,j)两列的残基，从而计算pkl
    Temlist = []
    templist = []
    for n in range(Seqlen):
        for m in range(Seqnum):
            templist.append((mulsequence[m][n]))
        Temlist.append(templist)
        templist = []
    ##Temlist[i]存储了第i列的所有残基###

    # labelindexfeatures = []
    Tfeatures = []
    localwindows = []
    centralwindows = []

    potential_levitt = ['G', 'A', 'V', 'L', 'I', 'P', 'D', 'E', 'N', 'Q', 'K', 'R', 'S', 'T', 'M', 'C', 'Y', 'W', 'H', 'F']  # 用于levitt和Jernigan potential
    potential_braun = ['G', 'A', 'V', 'L', 'I', 'F', 'Y', 'W', 'M', 'C', 'P', 'S', 'T', 'N', 'Q', 'H', 'K', 'R', 'D', 'E']  # 用于Braun potential

    ####下面的这几个list用于记录levitt potential####
    Levittmap = []
    Levittmap.append([0.1, 0.7, 0.1, 0.1, 0, 0.5, 0.4, 0.6, 0.1, 0, 0.4, -0.1, 0.4, 0.2, -0.1, -0.1, -0.4, -0.7, 0, -0.3])
    Levittmap.append([0.5, -0.3, -0.4, -0.4, 0.6, 0.3, 0.6, 0.3, 0, 1.0, 0.2, 0.5, 0, -0.5, 0.3, -0.7, -0.8, 0, -0.8])
    Levittmap.append([-1.1, -1.2, -1.2, 0, 0.4, 0, 0, -0.4, 0.1, -0.5, 0, -0.3, -1.0, -0.5, -1.2, -1.6, -0.5, -1.5])
    Levittmap.append([-1.4, -1.4, -0.1, 0, -0.1, -0.1, -0.6, 0.1, -0.6, 0, -0.3, -1.3, -0.8, -1.4, -1.7, -0.7, -1.6])
    Levittmap.append([-1.5, -0.1, 0, -0.2, -0.1, -0.4, 0, -0.7, -0.1, -0.6, -1.4, -0.8, -1.4, -1.8, -0.8, -1.7])
    Levittmap.append([0.1, 0.1, 0.1, -0.1, -0.3, 0.6, -0.2, 0.2, 0, -0.5, 0, -1.0, -1.3, -0.4, -0.7])
    Levittmap.append([0, 0, -0.6, -0.3, -1.0, -1.4, -0.3, -0.3, 0.1, 0, -1.0, -0.6, -1.1, -0.3])
    Levittmap.append([0.1, -0.6, -0.4, -1.1, -1.5, -0.2, -0.3, -0.3, 0.1, -1.0, -0.8, -1.0, -0.5])
    Levittmap.append([-0.7, -0.7, -0.3, -0.8, -0.1, -0.4, -0.3, 0, -0.8, -0.8, -0.8, -0.6])
    Levittmap.append([-0.5, -0.4, -0.9, 0, -0.5, -0.6, -0.2, -1.1, -1.0, -0.5, -0.8])
    Levittmap.append([0.7, 0.1, 0.1, 0, -0.1, 0.5, -1.0, -0.8, 0, -0.4])
    Levittmap.append([-0.9, -0.4, -0.6, -0.5, 0, -1.4, -1.3, -1.0, -0.9])
    Levittmap.append([0, -0.2, -0.1, -0.1, -0.6, -0.6, -0.6, -0.4])
    Levittmap.append([-0.5, -0.6, -0.3, -0.8, -0.9, -0.7, -0.7])
    Levittmap.append([-1.5, -0.8, -1.5, -2.0, -0.9, -1.9])
    Levittmap.append([-2.7, -0.8, -1.3, -0.6, -1.2])
    Levittmap.append([-1.6, -1.8, -1.5, -1.7])
    Levittmap.append([-2.2, -1.5, -2.0])
    Levittmap.append([-1.6, -1.2])
    Levittmap.append([-2.0])

    ####下面的这几个list用于记录Jernigan potential####


    Jerniganmap = []
    Jerniganmap.append([-2.1, -2.2, -3.0, -2.5, -2.7, -1.8, -1.9, -1.3, -2.4, -2.0, -1.9, -2.2, -1.9, -2.4, -2.8, -3.0, -2.8, -3.1, -2.1, -2.6])
    Jerniganmap.append([-2.9, -4.1, -3.7, -3.9, -2.3, -2.6, -1.9, -2.8, -2.7, -1.9, -2.4, -2.4, -3.2, -3.8, -3.1, -3.7, -3.8, -2.6, -3.7])
    Jerniganmap.append([-5.1, -4.7, -5.0, -3.2, -2.8, -2.8, -3.5, -3.4, -3.1, -3.5, -3.1, -3.9, -4.7, -4.3, -4.5, -4.8, -3.5, -4.6])
    Jerniganmap.append([-4.3, -4.6, -2.8, -2.7, -2.4, -3.0, -3.1, -2.5, -3.0, -2.6, -3.3, -4.4, -4.0, -4.1, -4.4, -3.1, -4.2])
    Jerniganmap.append([-4.9, -3.0, -2.9, -2.7, -3.2, -3.2, -2.9, -3.3, -3.0, -3.8, -4.7, -4.2, -4.4, -4.7, -3.4, -4.5])
    Jerniganmap.append([-2.2, -2.3, -1.8, -2.7, -2.5, -1.8, -2.4, -2.2, -2.8, -3.4, -2.8, -3.5, -3.7, -2.6, -3.1])
    Jerniganmap.append([-2.6, -2.0, -3.3, -2.7, -3.5, -3.7, -2.8, -3.2, -2.7, -2.9, -3.5, -3.2, -3.3, -2.7])
    Jerniganmap.append([-1.5, -2.8, -2.2, -3.2, -3.2, -2.3, -2.7, -2.7, -2.4, -3.0, -2.8, -2.8, -2.4])
    Jerniganmap.append([-3.6, -3.2, -3.0, -3.2, -2.8, -3.4, -3.3, -3.2, -3.5, -3.5, -3.2, -3.2])
    Jerniganmap.append([-2.6, -2.7, -2.9, -2.3, -3.1, -3.3, -2.9, -3.4, -3.3, -2.5, -3.0])
    Jerniganmap.append([-1.7, -2.0, -2.3, -2.8, -2.9, -2.4, -3.5, -3.3, -2.2, -2.8])
    Jerniganmap.append([-2.9, -2.6, -3.1, -3.1, -2.6, -3.6, -3.5, -2.9, -3.0])
    Jerniganmap.append([-2.4, -2.9, -3.0, -3.0, -3.1, -3.1, -2.7, -2.8])
    Jerniganmap.append([-3.5, -3.8, -3.5, -3.6, -3.7, -3.2, -3.4])
    Jerniganmap.append([-4.8, -4.2, -4.4, -4.8, -3.5, -4.6])
    Jerniganmap.append([-6.1, -3.8, -4.3, -3.3, -4.1])
    Jerniganmap.append([-4.1, -4.3, -3.7, -4.1])
    Jerniganmap.append([-4.7, -3.6, -4.4])
    Jerniganmap.append([-3.5, -3.3])
    Jerniganmap.append([-4.3])

    ####下面的这几个list用于记录Braun potential####
    Braunmap = []
    Braunmap.append([-0.29])
    Braunmap.append([-0.14, -0.18])
    Braunmap.append([-0.10, -0.15, -0.48])
    Braunmap.append([-0.04, -0.24, -0.29, -0.43])
    Braunmap.append([0.27, -0.25, -0.31, -0.45, -0.48])
    Braunmap.append([-0.09, -0.16, -0.31, -0.28, -0.05, -0.50])
    Braunmap.append([-0.21, -0.18, 0.00, -0.10, -0.34, -0.27, -0.11])
    Braunmap.append([-0.34, -0.01, 0.18, -0.18, -0.28, 0.16, -0.30, -0.53])
    Braunmap.append([0.25, -0.02, -0.02, -0.32, 0.21, -0.36, 0.01, -0.73, -0.75])
    Braunmap.append([-0.42, 0.08, 0.08, 0.36, -0.16, -0.28, 0.69, -0.74, 0.27, -1.77])
    Braunmap.append([0.06, 0.28, 0.76, 0.30, 0.99, 0.65, -0.02, 0.70, -0.78, 0.31, -0.78])
    Braunmap.append([0.04, 0.38, 0.18, 0.30, 0.57, 0.15, -0.03, 0.44, 0.00, 0.12, 0.21, -0.68])
    Braunmap.append([0.28, 0.06, 0.19, 0.57, 0.34, 0.25, 0.23, 0.74, 0.43, 0.28, 0.04, -0.23, -0.58])
    Braunmap.append([0.49, -0.04, 0.48, 0.25, 1.45, 0.12, -0.14, 0.46, -0.52, 0.07, 0.59, -0.21, -0.06, -0.45])
    Braunmap.append([0.54, 0.35, 0.41, 0.35, 0.44, -0.04, -0.06, -0.09, 0.07, 0.39, 0.73, 0.19, -0.31, 0.20, -0.17])
    Braunmap.append([-0.09, 0.44, 0.37, 0.10, 0.24, 0.25, 0.33, -0.34, 1.07, -0.45, -0.21, -0.13, -0.22, -0.56, 0.28, -0.15])
    Braunmap.append([0.56, 0.28, 0.53, 0.37, -0.00, 0.75, -0.00, 0.02, 0.44, 0.68, 0.26, -0.05, -0.26, -0.27, 0.05, 0.57, 0.21])
    Braunmap.append([0.40, 0.59, 0.43, 0.37, 0.05, 0.31, 0.03, -0.20, 0.53, 0.92, 0.34, 0.24, -0.31, -0.00, 0.56, -0.11, 0.58, -0.03])
    Braunmap.append([-0.26, 0.24, 0.51, 0.80, 0.26, 0.33, 0.61, 0.74, 0.21, 0.53, 0.87, -0.03, 0.32, -0.43, -0.03, -0.61, -0.43, -0.79, 0.11])
    Braunmap.append([0.21, 0.53, 0.37, 0.51, 0.53, 0.38, 0.25, 1.37, 0.44, 0.17, 0.41, 0.10, -0.27, 0.76, -0.20, -0.14, -1.12, -0.85, 0.86, 0.58])
    f = open(path5.format(dir=name), 'w')
    g = open(path6.format(dir=name), 'w')
    ####大循环开始了，循环结束后，得到所有满足距离要求的残基对(i,j)的特征
    for i in range(Seqlen - 7):
        if mulsequence[0][i] != '-':
            for j in range(i + 7, Seqlen):  # 所有可能的成对的情况（sep>=6)
                if mulsequence[0][j] != '-':
                    ###下面提取以i为中心的9个窗口，注意左端越界的情况###
                    Ilocalwindows = []
                    Ilocalrightwindows = Tlist[i] + Tlist[i + 1] + Tlist[i + 2] + Tlist[i + 3] + Tlist[i + 4]
                    Ilocalleftwindows = []
                    if i == 0:
                        Ilocalleftwindows = [0] * 108
                    elif i == 1:
                        Ilocalleftwindows = ([0] * 81) + Tlist[0]
                    elif i == 2:
                        Ilocalleftwindows = ([0] * 54) + Tlist[0] + Tlist[1]
                    elif i == 3:
                        Ilocalleftwindows = ([0] * 27) + Tlist[0] + Tlist[1] + Tlist[2]
                    else:
                        Ilocalleftwindows = Tlist[i - 4] + Tlist[i - 3] + Tlist[i - 2] + Tlist[i - 1]
                    Ilocalwindows = Ilocalleftwindows + Ilocalrightwindows

                    ###以i为中心的9个窗口提取完毕，下面是以j为中心的9个窗口，此时注意右端越界的情况###
                    Jlocalwindows = []
                    Jlocalleftwindows = Tlist[j - 4] + Tlist[j - 3] + Tlist[j - 2] + Tlist[j - 1]
                    Jlocalrightwindows = []
                    if j <= (Seqlen - 5):
                        Jlocalrightwindows = Tlist[j] + Tlist[j + 1] + Tlist[j + 2] + Tlist[j + 3] + Tlist[j + 4]
                    elif j == (Seqlen - 4):
                        Jlocalrightwindows = Tlist[Seqlen - 4] + Tlist[Seqlen - 3] + Tlist[Seqlen - 2] + Tlist[
                            Seqlen - 1] + (
                                                 [0] * 27)
                    elif j == (Seqlen - 3):
                        Jlocalrightwindows = Tlist[Seqlen - 3] + Tlist[Seqlen - 2] + Tlist[Seqlen - 1] + ([0] * 54)
                    elif j == (Seqlen - 2):
                        Jlocalrightwindows = Tlist[Seqlen - 2] + Tlist[Seqlen - 1] + ([0] * 81)
                    else:
                        Jlocalrightwindows = Tlist[j] + ([0] * 108)

                    Jlocalwindows = Jlocalleftwindows + Jlocalrightwindows

                    if (i, j) in Conlist:
                        localwindows = Ilocalwindows + Jlocalwindows
                    else:
                        localwindows = Ilocalwindows + Jlocalwindows
                    ###提取出了以j为中心的9个窗口###

                    ###localwindows中保存了当前(i,j)对的local window feature，格式为[.......(486位特征，2*9*27)..]####


                    ###labelindexfeatures.append(localwindows)，用不到
                    ###labelindexfeatures[i]现在表示某一对(i,j)的特征，用不到
                    ###以下，仍在循环中，求所有(i,j)对的central segment window features,并保存在centralwindows中####
                    central = int((i + i) / 2)
                    centralwindows = Tlist[central - 2] + Tlist[central - 1] + Tlist[central] + Tlist[central + 1] + Tlist[central + 2]
                    sepnum = j - i + 1
                    sep = []
                    if sepnum < 6:
                        sep = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    elif sepnum == 6:
                        sep = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    elif sepnum == 7:
                        sep = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    elif sepnum == 8:
                        sep = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    elif sepnum == 9:
                        sep = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    elif sepnum == 10:
                        sep = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    elif sepnum == 11:
                        sep = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    elif sepnum == 12:
                        sep = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
                    elif sepnum == 13:
                        sep = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
                    elif sepnum == 14:
                        sep = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
                    elif sepnum < 19:
                        sep = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
                    elif sepnum < 24:
                        sep = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
                    elif sepnum <= 29:
                        sep = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
                    elif sepnum <= 39:
                        sep = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
                    elif sepnum <= 49:
                        sep = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
                    else:
                        sep = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

                    ####central segment window feature 提取完毕####
                    ####以下为residue type feature,mulsequence[0]是目标残基序列，共有Seqlen个残基####
                    resitype = []
                    if mulsequence[0][i] in ['G', 'A', 'V', 'L', 'I', 'P', 'M', 'F', 'W'] and mulsequence[0][j] in ['G', 'A', 'V', 'L', 'I', 'P', 'M', 'F', 'W']:
                        resitype = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    elif (mulsequence[0][i] in ['G', 'A', 'V', 'L', 'I', 'P', 'M', 'F', 'W'] and mulsequence[0][j] in ['S', 'T', 'N', 'Q', 'C', 'Y']) or (
                                    mulsequence[0][j] in ['G', 'A', 'V', 'L', 'I', 'P', 'M', 'F', 'W'] and mulsequence[0][i] in ['S', 'T', 'N', 'Q', 'C', 'Y']):
                        resitype = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
                    elif (mulsequence[0][i] in ['G', 'A', 'V', 'L', 'I', 'P', 'M', 'F', 'W'] and mulsequence[0][j] in ['D', 'E']) or (
                                    mulsequence[0][j] in ['G', 'A', 'V', 'L', 'I', 'P', 'M', 'F', 'W'] and mulsequence[0][i] in ['D', 'E']):
                        resitype = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
                    elif (mulsequence[0][i] in ['G', 'A', 'V', 'L', 'I', 'P', 'M', 'F', 'W'] and mulsequence[0][j] in ['K', 'R', 'H']) or (
                                    mulsequence[0][j] in ['G', 'A', 'V', 'L', 'I', 'P', 'M', 'F', 'W'] and mulsequence[0][i] in ['K', 'R', 'H']):
                        resitype = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
                    elif mulsequence[0][i] in ['S', 'T', 'N', 'Q', 'C', 'Y'] and mulsequence[0][j] in ['S', 'T', 'N', 'Q', 'C', 'Y']:
                        resitype = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
                    elif (mulsequence[0][i] in ['S', 'T', 'N', 'Q', 'C', 'Y'] and mulsequence[0][j] in ['D', 'E']) or (
                                    mulsequence[0][j] in ['S', 'T', 'N', 'Q', 'C', 'Y'] and mulsequence[0][i] in ['D', 'E']):
                        resitype = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
                    elif (mulsequence[0][i] in ['S', 'T', 'N', 'Q', 'C', 'Y'] and mulsequence[0][j] in ['K', 'R', 'H']) or (
                                    mulsequence[0][j] in ['S', 'T', 'N', 'Q', 'C', 'Y'] and mulsequence[0][i] in ['K', 'R', 'H']):
                        resitype = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
                    elif (mulsequence[0][i] in ['D', 'E'] and mulsequence[0][j] in ['D', 'E']) or (mulsequence[0][j] in ['D', 'E'] and mulsequence[0][i] in ['D', 'E']):
                        resitype = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
                    elif (mulsequence[0][i] in ['D', 'E'] and mulsequence[0][j] in ['K', 'R', 'H']) or (mulsequence[0][i] in ['D', 'E'] and mulsequence[0][j] in ['K', 'R', 'H']):
                        resitype = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
                    else:
                        resitype = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

                    ###residue type 提取完毕###
                    ###下面，pairwise information features###

                    pairentropy = 0
                    twolist = []
                    temptwolist = []
                    for m in range(Seqnum):
                        temptwolist = (Temlist[i][m], Temlist[j][m])
                        twolist.append(temptwolist)
                    c = Counter(twolist)
                    for element in list(c.elements()):
                        pkl = (c[element]) / Seqnum  # 得到了pkl
                        pk = Tlist[i][ProDict.index(element[0])]  # 第i列中，element[0]这一残基的概率
                        pl = Tlist[j][ProDict.index(element[1])]  # 第j列中，element[1]这一残基的概率
                        if (pk != 0) and (pl != 0):
                            pairentropy += pkl * (math.log(pkl / (pk * pl)))

                    x = np.array(Tlist[i][0:21])
                    y = np.array(Tlist[j][0:21])
                    lx = np.sqrt(x.dot(x))
                    ly = np.sqrt(y.dot(y))
                    cosine = x.dot(y) / (lx * ly)

                    r = stats.pearsonr(x, y)[0]

                    ###只剩下potential了####
                    ###使用了三种potential，分别为Levitt's contact potential, Jernigan's pairwise potential, 以及Braun's pairwise potential,
                    ###计算方法参考了svmcon原作者的script###

                    ####Levitt's contact potential####
                    Levittplotential = 0
                    id1 = potential_levitt.index(mulsequence[0][i])
                    id2 = potential_levitt.index(mulsequence[0][j])
                    if id1 > id2:
                        tmpid = id1
                        id1 = id2
                        id2 = tmpid
                    Levittplotential = Levittmap[id1][id2 - id1]
                    ####Jernigan's pairwise potential####
                    Jerniganpotential = 0
                    Jerniganpotential = Jerniganmap[id1][id2 - id1]
                    ####Braun's pairwise potential####
                    Braunpotential = 0
                    id1B = potential_braun.index(mulsequence[0][i])
                    id2B = potential_braun.index(mulsequence[0][j])
                    if id1B < id2B:
                        tmpidB = id1B
                        id1B = id2B
                        id2B = tmpidB
                    Braunpotential = Braunmap[id1B][id2B]
                    ####potential提取完毕####

                    ########所有特征提取完毕，下面汇总########



                    ##最后再加下面这两个，加在循环里面
                    features = localwindows + centralwindows + sep + resitype + [pairentropy, cosine, r] + [Levittplotential, Jerniganpotential,
                                                                                                            Braunpotential]  # 以上所有特征列表的加和，包括localwindows，centralwindows，sep，等等
                    if (i, j) in Conlist:
                        f.write(str(features))
                        f.write('\n')
                    else:
                        g.write(str(features))
                        g.write('\n')



    f.close()
    g.close()
    return


import os
names = os.listdir(r'E:\protein\rawdata\feature')  #names是个list


for name in names:
    features(name)
