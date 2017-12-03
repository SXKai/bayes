# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 21:39:21 2017

@author: Q
"""
import numpy as np
import re
import feedparser
import operator
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

def createVocabList(data):    #创建词向量
    returnList = set([])
    for subdata in data:
        returnList = returnList | set(subdata)
    return list(returnList)
    

def setofWords2Vec(vocabList,data):      #将文本转化为词条

    returnList = [0]*len(vocabList)
    for vocab in data:
        if vocab in vocabList:
            returnList[vocabList.index(vocab)] += 1
    return returnList

    
def trainNB0(trainMatrix,trainCategory):        #训练，得到分类概率
    pAbusive = sum(trainCategory)/len(trainCategory)
    p1num = np.ones(len(trainMatrix[0]))
    p0num = np.ones(len(trainMatrix[0]))
    p1Denom = 2
    p0Denom = 2
    for i in range(len(trainCategory)):
        if trainCategory[i] == 1:
            p1num = p1num + trainMatrix[i]
            p1Denom = p1Denom + sum(trainMatrix[i])
        else:
            p0num = p0num + trainMatrix[i]
            p0Denom = p0Denom + sum(trainMatrix[i])
    p1Vect = np.log(p1num/p1Denom)
    p0Vect = np.log(p0num/p0Denom)
    return p0Vect,p1Vect,pAbusive

    
def  classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):    #分类
    p0 = sum(vec2Classify*p0Vec)+np.log(1-pClass1)
    p1 = sum(vec2Classify*p1Vec)+np.log(pClass1)
    if p1 > p0:
        return 1
    else:
        return 0
def textParse(bigString):          #文本解析
    splitdata = re.split(r'\W+',bigString)
    splitdata = [token.lower() for token in splitdata if len(token) > 2]
    return splitdata
def spamTest():
    docList = []
    classList = []
    for i in range(1,26):
        with open('spam/%d.txt'%i) as f:
            doc = f.read()
        docList.append(doc)
        classList.append(1)
        with open('ham/%d.txt'%i) as f:
            doc = f.read()
        docList.append(doc)
        classList.append(0)
    vocalList = createVocabList(docList)
    trainList = list(range(50))
    testList = []
    for i in range(13):
        num = int(np.random.uniform(0,len(docList))-10)
        testList.append(trainList[num])
        del(trainList[num])
    docMatrix = []
    docClass = []
    for i in trainList:
        subVec = setofWords2Vec(vocalList,docList[i])
        docMatrix.append(subVec)
        docClass.append(classList[i])
    p0v,p1v,pAb = trainNB0(docMatrix,docClass)
    errorCount = 0
    for i in testList:
        subVec = setofWords2Vec(vocalList,docList[i])
        if classList[i] != classifyNB(subVec,p0v,p1v,pAb):
            errorCount += 1
    return errorCount/len(testList)

def calcMostFreq(vocabList,fullText):
    count = {}
    for vocab in vocabList:
        count[vocab] = fullText.count(vocab)
    sortedFreq = sorted(count.items(),key=operator.itemgetter(1),reverse=True)
    return sortedFreq[:30]

def localWords(feed1,feed0):
    docList = []
    classList = []
    fullText = []
    numList = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(numList):
        doc1 = feed1['entries'][i]['summary']
        docList.append(doc1)
        classList.append(1)
        fullText.extend(doc1)
        doc0 = feed0['entries'][i]['summary']
        docList.append(doc0)
        classList.append(0)
        fullText.extend(doc0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList,fullText)
    for word in top30Words:
        if word[0] in vocabList:
            vocabList.remove(word[0])
    trainingSet = list(range(2*numList))
    testSet = []
    for i in range(20):
        randnum = int(np.random.uniform(0,len(trainingSet)-5))
        testSet.append(trainingSet[randnum])
        del(trainingSet[randnum])
    trainMat = []
    trainClass = []
    for i in trainingSet:
        trainClass.append(classList[i])
        trainMat.append(setofWords2Vec(vocabList,docList[i]))
    p0V,p1V,pSpam = trainNB0(trainMat,trainClass)
    errCount = 0
    for i in testSet:
        testData = setofWords2Vec(vocabList,docList[i])
        if classList[i] != classifyNB(testData,p0V,p1V,pSpam):
            errCount += 1
    return errCount/len(testData)
if __name__=="__main__":
    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    print(localWords(ny,sf))
    


    