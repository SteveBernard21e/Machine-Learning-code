#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 16:05:32 2017

@author: SteveBernard
"""

import numpy as np
import math

#这里一律使用numpy数组，遇到特殊情况可以使用矩阵类型进行操作，比如对应项相乘再相加

def loadSimpData():
    dataSet = np.array([[ 1. ,  2.1],
                        [ 2. ,  1.1],
                        [ 1.3,  1. ],
                        [ 1. ,  1. ],
                        [ 2. ,  1. ]])
    classLabel = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataSet,classLabel



def weakLearner(dataSet, classLabel, D):  #dataset/weight distribution
    #观察数据我们发现属性值是连续型数值，因为我们要使用针对连续性数值的方法
    #注意，这里的最优划分属性的标准不再是 信息增益，而是 分类错误率    
    nRow, nColumn = dataSet.shape
    classLabel = np.array(classLabel)
    nStep = 10.0
    #
    best_error = math.inf
    best_attribute_index = 0
    best_threshold = 0
    best_label_value = 0
    best_labels = []
    for i in range(nRow):
        best_labels.append(0)
    best_labels = np.array(best_labels)
    #print(best_labels)
    #best_labels = np.zeros((1,nRow)) NONONO,shit!

    #print(errors)    
    for i in range(nColumn): # i=0,1,2,3,...(nColumn-1)
        attributeValue = [example[i] for example in dataSet] # return a list which contains all attribute values
        print(i)
        print(attributeValue)
        sortedValue = sorted(attributeValue) #retunrn a list
        minimum_value = sortedValue[0]
        maximum_value = sortedValue[-1]
        interval = (maximum_value - minimum_value) / nStep
        '''
        Label_predict = []
        for k in range(nRow):
            Label_predict.append(1)
        Label_predict = np.array(Label_predict)
        '''
        #
        for j in range(-1,int(nStep)+1): # j = 0,1,2,...,(nColumn-2)
            #threshold = (minimum_value + j * interval) #左到右，覆盖原范围
            threshold = (minimum_value + float(j) * interval)
            print('threshold=%f'%(threshold))
            '''
            for label_value in ['lt', 'gt']: #遍历所有可能性,小于的子集为1/-1
                 #每次都要初始化errors用于重新计算
                 errors = []
                 
                 Label_predict = []
                 for k in range(nRow):
                     Label_predict.append(1)
                 Label_predict = np.array(Label_predict)
                     
                 for i in range(nRow): 
                     errors.append(1.0)
                 errors = np.array(errors)
                 #print(errors)
                 #################################就他妈这段有问题，几把玩意
                 if label_value == 'lt':
                     for p in range(nRow):
                         if (attributeValue[p] <= threshold):
                             Label_predict[p] = -1.0
                     #print(Label_predict)
                 else: #label_value == 'gt':
                     for p in range(nRow):
                         if (attributeValue[p] > threshold):
                             Label_predict[p] = -1.0
                 '''
            for label_value in [1, -1]: #遍历所有可能性,小于的子集为1/-1
                 #每次都要初始化errors用于重新计算
                 errors = []
                 
                 Label_predict = []
                 for k in range(nRow):
                     Label_predict.append(1)
                 Label_predict = np.array(Label_predict)
                     
                 for i in range(nRow): 
                     errors.append(1.0)
                 errors = np.array(errors)
                 #print(errors)
                 #################################就他妈这段有问题，几把玩意
                 '''
                 if label_value == 'lt':
                     for p in range(nRow):
                         if (attributeValue[p] <= threshold):
                             Label_predict[p] = -1.0
                     #print(Label_predict)
                 else: #label_value == 'gt':
                     for p in range(nRow):
                         if (attributeValue[p] > threshold):
                             Label_predict[p] = -1.0
                 '''
                 Label_predict[attributeValue <= threshold] = label_value  #事实证明我还是对的
                 Label_predict[attributeValue > threshold] = (-1)*label_value
                 
 
                 #这么些没考虑样本权重问题 ：errorRate = sum(Label_deviné == classLabel)／nRow #对应项相等为1，否则为0,还需要结合样本权重
                 errors[Label_predict == classLabel] = 0 #三个数组的格式要一致，使errors数组中，Label_deviné == classLabel对应的index项赋值为0

                 weighted_error = np.mat(D) * np.mat(errors).T #利用矩阵乘法使每一项乘以对应的权重（为0的项相乘为0）求出,必须是matrix的数据类型

                 if weighted_error < best_error: #考虑使用字典一下子包含所有的变量

                     best_error = weighted_error
                     best_labels = Label_predict #存储分类结果,数组类型
                     best_attribute_index = i
                     best_threshold = threshold
                     best_label_value = label_value #存储符号，意义不大
    return best_error, best_attribute_index, best_threshold, best_label_value, best_labels #mdzz,这个函数好好的没问题
                  
def adaboost(dataSet, classLabel, T): #这里的T是人为规定的训练轮数
    nRow, nColumn = dataSet.shape
    D = []
    for i in range(nRow):
        D.append(1.0/nRow)
    D = np.array(D)
    print(D)

    Label_sum = []
    for k in range(nRow):
        Label_sum.append(0)
    Label_sum = np.array(Label_sum)
    #
    for t in range(T): # i=0,1,2,..,T-1
        best_error, best_attribute_index, best_threshold, best_label_value, best_labels = weakLearner(dataSet, classLabel, D)
        print(type(classLabel))
        '''
        if best_error > 0.5:
            break
        '''
        alpha = float(1/2 * math.log((1.0-best_error)/max(best_error,1e-16)))     
        for i in range(nRow): # i=0,1,2,3,..,nRow-1
            if best_labels[i]==classLabel[i]:
                D[i] = D[i] * math.exp(-alpha)
            else:
                D[i] = D[i] * math.exp(alpha)
        D = D / sum(D)  #分布归一化
        print('D权值分布')
        print(D)
        
        Label_sum = Label_sum + alpha*best_labels #all array
        Boost_label = sign(Label_sum)
        print(Boost_label)
        #print(type(classLabel))   ？？？？？？？？？？？？？？？？？？？？？？？我在前面的wakLeaner函数中传入了classLabel这个list，并在里面对其进行了数据类型的改变，然而这里又成了list类型？
        if sum(Boost_label - np.array(classLabel)) == 0:
            break 
    return  Label_sum 

def trail(lista,a):  #函数的内存调用规律是什么
    lista.append(100)
    lista = np.array(list)
    b = a
    return b  #事实证明数据类型是修改了的
    
    
    
def main(): #dataset的格式举例，不用书上给的，感觉很烦，回头写一个自动加载数据的loaddata函数
    dataSet,classLabel = loadSimpData()
    T = 4
    Label_sum = adaboost(dataSet, classLabel, T)
    print(Label_sum)
    
    print()
    b = trail(classLabel,1)
    print(type(classLabel))
    print(classLabel)
    

main()























