def nBayesClassifier(traindata, trainlabel,testdata, testlabel, threshold):
    # 朴素贝叶斯分类器
    # threshold 为用于判断类别的后验概率的阈值。
    # 要求函数返回对测试数据的预测 ypred，以及通过与 ground label(真实标签)比较计算得到的性能指标 SP,SR 和 F。
    # ypred 与 SP,SR,F 以 tuple 形式返回。我们要求以表格的形式给出你的结果(其余两个分类器要求类似)并对分类器性能做出分析。
    # SP: 垃圾邮件识别准确率, Precision
    # SR: 垃圾邮件识别的查全率, Recall
    # F = SP*SR*2/(SP+SR)
    print("begin training...")
    wordNum = len(traindata)
    trainsetLen = len(trainlabel)
    testsetLen = len(testlabel)
    pIfSpam = [0 for i in range(wordNum)]
    pIfNotSpam = [0 for i in range(wordNum)]
    pWord = [0 for i in range(wordNum)]
    # 求先验概率
    countSpam = 0
    for i in range(trainsetLen):
        if trainlabel[i]==1:
            countSpam = countSpam + 1
    # countSpam为训练集中的垃圾邮件数
    pSpam = countSpam/(trainsetLen)
    countEasyHam = trainsetLen - countSpam
    pNotSpam = 1 - pSpam
    # 计算 P(wordi|Spam) 和 P(wordi|EasyHam) 和 P(wordi)
    for i in range(trainsetLen):
        #print(i)
        if(trainlabel[i]==1):
            # Spam
            for j in range(wordNum):
                if(traindata[j][i]>threshold):
                    pIfSpam[j] = pIfSpam[j] + 1
                    pWord[j] = pWord[j] + 1
        else:
            # Easy Ham
            for j in range(wordNum):
                if(traindata[j][i]>threshold):
                    pIfNotSpam[j] = pIfNotSpam[j] + 1 
                    pWord[j] = pWord[j] + 1 
    for i in range(wordNum):
        pIfNotSpam[i] = pIfNotSpam[i] / countEasyHam
        pIfSpam[i] = pIfSpam[i] / countSpam
        pWord[i] = pWord[i] / trainsetLen
    # pIfNotSpam[i]: P(wordi|Easy Ham)
    # pIfSpam[i]: p(wordi|Spam)
    print("begin testing...")
    # 测试
    nPam2Pam = 0
    nNormal2Pam = 0
    nPam = 0
    ypred = [0 for i in range(testsetLen)]
    for j in range(testsetLen):
        #print(j)
        p1 = pSpam
        p2 = pNotSpam
        for i in range(wordNum):
            if(testdata[i][j]>threshold):
                if(pWord[i]==0):
                    continue
                # Spam, have wordi
                p1 = p1 * pIfSpam[i]
                p1 = p1 / pWord[i]
                # Not Spam, have wordi
                p2 = p2 * pIfNotSpam[i]
                p2 = p2 / pWord[i]
            else:
                if(pWord[i]==1):
                    continue
                # Spam, not have wordi
                p1 = p1 * (1-pIfSpam[i])
                p1 = p1 / (1-pWord[i])
                # Not Spam, not have wordi
                p2 = p2 * (1-pIfNotSpam[i])
                p2 = p2 / (1-pWord[i])
        if(p1>=p2):
            ypred[j] = 1
            # predict spam
            if(testlabel[j]==1):
                # Spam
                nPam2Pam = nPam2Pam + 1
                nPam = nPam + 1
            else:
                #Normal
                nNormal2Pam = nNormal2Pam + 1
        else:
            if(testlabel[j]==1):
                nPam = nPam + 1
            ypred[j] = -1
    SP = nPam2Pam/(nPam2Pam+nNormal2Pam)
    SR = nPam2Pam/nPam
    F = SP*SR*2/(SP+SR)
    return ypred, SP, SR, F

import os
import re
import sys


def textParse(line, vocab):
    words = re.split(r'[_\W*]',line)
    #return [tok.lower() for tok in words if len(tok)>2]
    for i in words:
        i = i.lower()
        if(i.isalpha() and len(i)<=15):
            vocab.add(i)
    return vocab
        
vocab = set()
filepath = "C:\\Users\\weiq6\\Desktop\\20021010_easy_ham\\easy_ham"
pathDir =  os.listdir(filepath)
i = 0
for allDir in pathDir:
    if(i<250):
        i = i + 1
        continue
    path = os.path.join('%s\\%s' % (filepath, allDir))
    f = open(path,"r",encoding="windows-1252")
    #print(path)
    header = True
    for line in f:
        if(line=="\n"):
            header = False
        if(header==False):
            vocab = textParse(line,vocab)
    i = i + 1

filepath = "C:\\Users\\weiq6\\Desktop\\20021010_spam\\spam"
pathDir =  os.listdir(filepath)
i = 0
for allDir in pathDir:
    if(i<50):
        i = i + 1
        continue
    path = os.path.join('%s\\%s' % (filepath, allDir))
    f = open(path,"r",encoding="windows-1252")
    header = True
    #print(path)
    try:
        for line in f:
            if(line=="\n"):
                header = False
            if(header==False):
                vocab = textParse(line,vocab)
    except:
        print("error at "+path)
    i = i + 1


vocab = list(vocab)

# 设置200个EasyHam和50个Spam邮件作为测试集
wordNum = len(vocab)

filepath = "C:\\Users\\weiq6\\Desktop\\20021010_easy_ham\\easy_ham"
pathDir =  os.listdir(filepath)
len1 = len(pathDir)
filepath = "C:\\Users\\weiq6\\Desktop\\20021010_spam\\spam"
pathDir =  os.listdir(filepath)
len2 = len(pathDir)
testsetLen = 300
trainsetLen = len1 + len2 - testsetLen

testdata = [[0 for i in range(testsetLen)] for j in range(wordNum)]
testlabel = [1 if i<250 else -1 for i in range(testsetLen)]
traindata = [[0 for i in range(trainsetLen)] for j in range(wordNum)]
trainlabel = [1 if i<(len1-250) else -1 for i in range(trainsetLen)]
# 先easy ham，后spam

def Parse(line,i,testNum,base):
    words = re.split(r'[_\W*]',line)
    #return [tok.lower() for tok in words if len(tok)>2]
    for word in words:
        word = word.lower()
        if(word.isalpha() and len(word)<=15):
            if word in vocab:
                index = vocab.index(word)
                if(index>=0):
                    if(i<testNum):
                        testdata[index][base*250+i] += 1
                    else:
                        traindata[index][base*(len1-250)+i-testNum] += 1
    
print("processing easy ham set...")
filepath = "C:\\Users\\weiq6\\Desktop\\20021010_easy_ham\\easy_ham"
pathDir =  os.listdir(filepath)
i = 0
for allDir in pathDir:
    path = os.path.join('%s\\%s' % (filepath, allDir))
    f = open(path,"r",encoding="windows-1252")
    #print(path)
    header = True
    for line in f:
        if(line=="\n"):
            header = False
        if(header==False):
            Parse(line,i,250,0)
    i = i + 1

print("processing spam set...")
filepath = "C:\\Users\\weiq6\\Desktop\\20021010_spam\\spam"
pathDir =  os.listdir(filepath)
i = 0
for allDir in pathDir:
    path = os.path.join('%s\\%s' % (filepath, allDir))
    f = open(path,"r",encoding="windows-1252")
    header = True
    #print(path)
    try:
        for line in f:
            if(line=="\n"):
                header = False
            if(header==False):
                Parse(line,i,50,1)
    except:
        print("error at "+path)
    i = i + 1

print("Naive Bayes Model")
for i in range(10):
    ypred, SP, SR, F = nBayesClassifier(traindata, trainlabel, testdata, testlabel, i)
    print(SP, SR, F)

print("Least Square")

import numpy as np
def lsClassifier(traindata,trainlabel, testdata, testlabel, Lambda):
    wordNum = len(traindata)
    trainsetLen = len(trainlabel)
    testsetLen = len(testlabel)
    #w = np.zeros(wordNum)
    print("begin training...")
    x = np.mat(traindata)
    y = np.mat(trainlabel)
    y = y.T
    #temp = np.mat(np.eye(wordNum,wordNum,dtype=int))
    w = x * (x.T)
    for i in range(wordNum):
        w[i,i] += Lambda
    w = w.I
    w = w * x
    w = w * y
    w = w.T
    print("begin testing...")
    x = np.mat(testdata)
    y = np.array(testlabel)
    ypred = w * x
    ypred = np.array(ypred)
    ypred = ypred[0]
    nPam2Pam = 0
    nPam = 0
    nNormal2Pam = 0
    for i in range(testsetLen):
        if(ypred[i]>0):
            ypred[i] = 1
        else:
            ypred[i] = -1
        if(y[i]==1):
            nPam += 1
        if(y[i]==-1 and ypred[i]==1):
            nNormal2Pam += 1
        elif(y[i]==1 and ypred[i]==1):
            nPam2Pam += 1
    SP = nPam2Pam/(nPam2Pam+nNormal2Pam)
    SR = nPam2Pam/nPam
    F = SP*SR*2/(SP+SR)
    return ypred,SP,SR,F

# 测试最小二乘分类
for i in range(1,10):
    ypred,SP,SR,F = lsClassifier(traindata,trainlabel, testdata, testlabel, i)
    print(SP,SR,F)


import math
def Kernel(x1,x2,sigma):
    x1 = np.array(x1)
    x2 = np.array(x2)
    if(sigma==0):
        return x1*x2
    else:
        x1 = x1 - x2
        res = np.dot(x1, x2)
        res = - res / sigma / sigma
        res = math.exp(res)
        return res


# soft svm
def softsvm(traindata, trainlabel, testdata, testlabel, sigma, C):
    '''
    model = svm.SVC(kernel='linear', C=C, gamma=1) 
    if(sigma!=0):
        gamma = 1/2/sigma/sigma
        model = svm.SVC(C=C, kernel='rbf',gamma=gamma)
    X = np.array(traindata)
    y = np.array(trainlabel)
    X = np.transpose(X)
    model.fit(X, y)
    #Predict Output
    x_test = np.array(testdata)
    x_test = np.transpose(x_test)
    y_test = np.array(testlabel)
    ypred = model.predict(x_test)
    
    nPam2Pam = 0
    nNormal2Pam = 0
    nPam = 0
    for i in range(testsetLen):
        if(ypred[i]==1):
            if(y_test[i]==1):
                nPam2Pam += 1
            else:
                nNormal2Pam += 1
        if(y_test[i]==1):
            nPam += 1
    '''
    wordNum = len(traindata)
    trainsetLen = len(trainlabel)
    testsetLen = len(testlabel)
    # calculate alpha
    alpha = np.ones(trainsetLen)
    x = np.mat(traindata)
    x = x.T
    y = np.array(trainlabel)
    grad = np.ones(trainsetLen)
    
    b = 0
    count = 0
    Sum = 0
    for i in range(trainsetLen):
        if(alpha[i]!=0):
            count += 1
            b = y[i]
            for j in range(trainsetLen):
                b = b - alpha[j]*y[j]*Kernel(x[i],x[j])
            Sum += b
    b = Sum/count
    
    testx = np.array(testdata)
    testy = np.array(testlabel)
    testx = np.transpose(x)
    ypred = np.zeros(testsetLen)
    nPam2Pam = 0
    nNormal2Pam = 0
    nPam = 0
    for i in range(testsetLen):
        ypred[i] = b
        for j in range(trainsetLen):
            ypred[i] += alpha[j]*y[j]*Kernel(x[j],testx[i])
        if(ypred[i]>0):
            ypred[i] = 1
        else:
            ypred[i] = -1
        if(ypred[i]==1):
            if(testy[i]==1):
                nPam2Pam += 1
            else:
                nNormal2Pam += 1
        if(testy[i]==1):
            nPam += 1
    
    SP = nPam2Pam/(nPam2Pam+nNormal2Pam)
    SR = nPam2Pam/nPam
    F = SP*SR*2/(SP+SR)
    return ypred,SP,SR,F
    
# 测试soft svm
for C in [1,10,100]:
    for sigma in [0,0,1,10,100]:
        ypred,SP,SR,F = softsvm(traindata, trainlabel, testdata, testlabel, sigma, C)
        print(C, sigma, SP,SR,F)

# 交叉验证
wordNum = len(vocab)

filepath = "C:\\Users\\weiq6\\Desktop\\20021010_easy_ham\\easy_ham"
pathDir =  os.listdir(filepath)
len1 = len(pathDir)
filepath = "C:\\Users\\weiq6\\Desktop\\20021010_spam\\spam"
pathDir =  os.listdir(filepath)
len2 = len(pathDir)

EHdata = [[0 for i in range(len1)] for j in range(wordNum)]
Pamdata = [[0 for i in range(len2)] for j in range(wordNum)]

def parseline(line,i,spam):
    words = re.split(r'[_\W*]',line)
    for word in words:
        word = word.lower()
        if(word.isalpha() and len(word)<=15):
            if word in vocab:
                index = vocab.index(word)
                if(index>=0):
                    if(spam):
                        Pamdata[index][i] = Pamdata[index][i] + 1
                    else:
                        EHdata[index][i] = EHdata[index][i] + 1
print("processing easy ham set...")
filepath = "C:\\Users\\weiq6\\Desktop\\20021010_easy_ham\\easy_ham"
pathDir =  os.listdir(filepath)
i = 0
for allDir in pathDir:
    path = os.path.join('%s\\%s' % (filepath, allDir))
    f = open(path,"r",encoding="windows-1252")
    #print(path)
    header = True
    for line in f:
        if(line=="\n"):
            header = False
        if(header==False):
            parseline(line,i,False)
    i = i + 1
    f.close()

print("processing spam set...")
filepath = "C:\\Users\\weiq6\\Desktop\\20021010_spam\\spam"
pathDir =  os.listdir(filepath)
i = 0
for allDir in pathDir:
    path = os.path.join('%s\\%s' % (filepath, allDir))
    f = open(path,"r",encoding="windows-1252")
    header = True
    #print(path)
    try:
        for line in f:
            if(line=="\n"):
                header = False
            if(header==False):
                parseline(line,i,True)
    except:
        print("error at "+path)
    i = i + 1
    f.close()
EHdata = np.mat(EHdata)
Pamdata = np.mat(Pamdata)


def crossvalidation(threshold, Lambda, sigma, C, EHdata, Pamdata):
    testnum = int(len1/5)+int(len2/5)
    trainnum = len1 + len2 - testnum
    #train = np.zeros(wordNum,trainnum)
    #test = np.zeros(wordNum, testnum)
    testlabel = [1 if i < int(len1/5) else -1 for i in range(testnum)]
    trainlabel = [1 if i < len1-int(len1/5) else -1 for i in range(trainnum)]
    result1 = []
    result2 = []
    result3 = []
    for  i in range(5):
        print("fold ",i)
        begin = i*int(len1/5)
        test = EHdata[:,range(i*int(len1/5), (i+1)*int(len1/5))]
        temp = Pamdata[:,range(i*int(len2/5), (i+1)*int(len2/5))]
        test = np.hstack((test,temp))
        train = EHdata[:,range(i*int(len1/5))]
        temp = EHdata[:,range((i+1)*int(len1/5),len1)]
        train = np.hstack((train, temp))
        temp = Pamdata[:,range(i*int(len2/5))]
        train = np.hstack((train,temp))
        temp = Pamdata[:,range((i+1)*int(len2/5),len2)]
        train = np.hstack((train,temp))
        temp1 = []
        temp2 = []
        temp3 = []
        print("nBayes")
        for j in threshold:
            ypred,SP,SR,F = nBayesClassifier(train, trainlabel, test, testlabel, j)
            temp1.append(F)
            print(j,F)
        print("least square")
        for j in Lambda:
            ypred,SP,SR,F = lsClassifier(train,trainlabel, test, testlabel, j)
            temp2.append(F)
            print(j,F)
        print("soft svm")
        for j in sigma:
            for k in C:
                ypred,SP,SR,F = softsvm(train, trainlabel, test, testlabel, j, k)
                temp3.append(F)
                print(j,k,F)
        result1.append(temp1)
        result2.append(temp2)
        result3.append(temp3)
    return result1, result2, result3

r1, r2, r3 = crossvalidation([1,2,3], [0.01,0.5,1,10,100,5000], [0,100], [1,100,1000], EHdata, Pamdata)