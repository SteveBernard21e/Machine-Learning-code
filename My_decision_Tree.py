#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 09:44:56 2017

@author: SteveBernard
"""

from math import log
#import operator

def Majority_D(D):
    count = {}
    for item in D:
        if item[-1] not in count.keys():
            count[item[-1]] = 0
            #count[str(item[-1])] = 0
        else:
            count[item[-1]] +=1
    List_count = sorted(count.items(),key = lambda item:item[1],reverse=True)
    #print(List_count[0][0])
    #List_count = sorted(count.items(),key = operator.itemgetter(1),reverse=True)
    return List_count[0][0]

def Entropy_D(D):
    count = {}
    for item in D:
        if item[-1] not in count.keys():
            count[item[-1]] = 1
            #count[str(item[-1])] = 1
        else:
            count[item[-1]] +=1
    List_count = sorted(count.items(),key = lambda item:item[1],reverse=True)
    #print(List_count)
    Ent_D = 0.0
    for tuple_item in List_count:
        p = float(tuple_item[-1])/len(D)
        Ent_D -= p * log(p,2)
    #print(Ent_D)
    return Ent_D

def Ensemble_split(axis,value,D):
    D_sub = []
    count = 0
    for item in D:
        if item[axis] == value:   #
            count+=1
            D_sub.append(item)
    return D_sub
    
def Best_div(D,A):
    best_gain = 0
    best_attribute_axis = 0
    for axis in range(len(A)): #不希望遍历所有的属性,A在主函数中将已经消耗的属性变为NaN“
        #print(axis)
        if axis is not 'NaN':
            axis_values = [example[axis] for example in D]
            #print(axis_values)
            unique_axis_values = set(axis_values) 
            #print(unique_axis_values)  #OK
            Sum_Ent = 0
            for value in unique_axis_values: #OK
                #print(value)
                D_sub = Ensemble_split(axis,value,D)
                Ent_D = Entropy_D(D_sub)
                #print(Ent_D)
                Sum_Ent = Sum_Ent + Ent_D*len(D_sub)/len(D)
                #print(len(D_sub)/len(D))
            Gain = Entropy_D(D) - Sum_Ent
            #print('Gain=%f'%(Gain))
            if Gain > best_gain:
                   best_gain = Gain
                   best_attribute_axis = axis
    #print('best_attribute_axis=%d'%(best_attribute_axis))
    return best_attribute_axis
            
def createTree(D,A):
    classList = [example[-1] for example in D]
    if len(set(classList)) == 1:
        print('cas 1')
        return classList[0]
    if (A.count('NaN') == len(A)):
        print('cas 2')
        return Majority_D(D)
    #重新写，如果子集中的每个元素的所有属性都相同，则标记节点为叶结点
    count = 0
    for i in range(len(A)):
        if A[i] != 'NaN':
            column = [example[i] for example in D]
            if column.count(column[0]) == len(column):
                print("该集合元素中该属性值全部相同哦")
                count += 1
    if count == len(A)-A.count('NaN'):
        return Majority_D(D)
    
    axis = Best_div(D,A)
    #print(axis)
    best_attribute = A[axis]
    #print(best_attribute)
    A[axis] = 'NaN'
    Tree = {best_attribute:{}}
    value_List = [example[axis] for example in D]
    #print(set(value_List))
    for value in set(value_List):
        D_sub= Ensemble_split(axis,value,D)
        sub_A = A[:]  ###########回来再思考
        #print(D_sub)
        #print(axis)
        #print(A)
        Tree[best_attribute][value] = createTree(D_sub,sub_A)
    return Tree

#试一试给出一个样本，我应该如何借助我的决策树返回它的分类呢，这样我就成功构建了一个分类器
#要求是 给出一组样本的时候，我不但能返回每一个样本的分类，还应该直接输出正确比率。
def classficateur(decisionTree, A, T_vector_list):  #A是属性名称，T是单一测试样本，返回一个分类结果；如果对测试集进行分类，外边使用for循环这样可以简化函数，接口稍微复杂了一点而已

    attribute_name = list(decisionTree.keys())[0]
    #print(attribute_name)
    axis = A.index(attribute_name)
    Value_dict = decisionTree[attribute_name]
    #print(Value_dict)
    #print(Value_dict[T_vector_list[axis]])
    if isinstance(Value_dict[T_vector_list[axis]], dict) is not True:
        return Value_dict[T_vector_list[axis]]  #return一个value代表value的值是函数的返回结果
    else:
        return classficateur(Value_dict[T_vector_list[axis]],A,T_vector_list)
        #如果这里不加return，我递归计算，在return Value_dict[T_vector_list[axis]] return了
        #，但是这个return是传给了classficateur(Value_dict[T_vector_list[axis]],A,T_vector_list)，并不是最终要的classficateur(decisionTree, A, T_vector_list)
        #因此需要额外一次return把这个值返回出来

def store(tree):
    import pickle
    with open("myTree",'wb') as fileobj:#以二进制写入方式打开文件myTree，如果没有则自动生成一个文件，得到用于接下来操作的文件对象fileobj
    #pickle存储方式默认是二进制方式
        pickle.dump(tree,fileobj) #把tree以二进制形式保存到这个文件对象中，然后文件关闭

def read(filename):
    import pickle
    with open(filename,'rb') as tree: #pickle默认是二进制操作写文件，因此在读入文件时也应该二进制形式读取
        #print(pickle.load(tree))  #用于检测效果，没毛病。
        
        return pickle.load(tree)

def main():
    '''
    D = [[1, 1, 'yes'], #training set
         [1, 1, 'yes'],
         [1, 0, 'no'],
         [0, 1, 'no'],
         [0, 1, 'no']] 
    A = ['no surfacing','flippers'] #attribute set
    '''
    with open("lenses.txt") as fr:
         lenses = [inst.strip().split('\t') for inst in fr.readlines()]#strip()函数去掉字符串首位的字符，如果为空，则去除空格字符
         lensesLabels = ['age','prescript','astigmatic','tearRate'] 
    #print(lenses)
    D = lenses
    #print(lenses)
    A = lensesLabels
    #'''
    tree = createTree(D,A)
    print(tree)
    #A = ['no surfacing','flippers'] #attribute set,因为我把A中的元素替换为了NaN，所以自然出错了。。以后写程序给的数据争取不要改变
    
    #print(classficateur(tree,A,[]))
    
    #store(tree)
    #read("myTree")
   
main()









