# conding:utf-8

import itertools
import pandas as pd
import numpy as np

def get_C1(data: pd.DataFrame):
    
    C1 = {}

    for col in data:
        for i in data[col]:
            C1.setdefault(i, 0)
            C1[i] = C1[i] + 1

    if np.nan in C1:
        del C1[np.nan]

    C1 = pd.DataFrame({'itemset': [set([s]) for s in list(C1.keys())], 'count': list(C1.values())}) #注意这一步将dict的key转化为set的做法
    
    return C1

def trim_C(C, min_sup):
    
    L =  C[C['count'] >= min_sup]
    L = L[['itemset','count']]
    return L


def _findsubsets(s,m):
    return set(itertools.combinations(s, m))


def _connect(L: pd.DataFrame):
    pre_C  = []
    Lkeys = list(L.itemset)
    
    # 需要限制可以进行连接的情况
    lengthOfL = len(Lkeys)
    for i in range(lengthOfL-1):
        listed_i = list(Lkeys[i])
        listed_i.sort()
        for j in range(i+1, lengthOfL):
            listed_j = list(Lkeys[j])
            listed_j.sort()
            if listed_i[:-1] == listed_j[:-1]:
                pre_C.append(Lkeys[i] | Lkeys[j])
    return pre_C


def _move_candidate_has_infrequent_subset(pre_C: list, k, L_: dict):
    for s in pre_C: 
        for subset in _findsubsets(s, k-1):
            if set(subset) not in list(L_[k-1].itemset):
                if pre_C != []: #防止pre_C已空的情况下,继续remove报错
                    pre_C.remove(s)
    return pre_C


def _count_support(pre_C, D):
    # 创建C的骨架(dataframe)
    C = pd.DataFrame(columns=['itemset', 'count'])
    for i in pre_C:
        C = C.append([{'itemset':i, 'count':0}], ignore_index=True)#非in-place操作,注意赋值回去给C
    
    for index, row in D.iterrows():
        row = list(row)
        for cb in pre_C:
            if cb <= set(row):
                C.loc[C.itemset==cb, 'count'] += 1
    return C


# 由L1生成C2
#apriori_gen
def L2C(L: pd.DataFrame, D: pd.DataFrame, k: int, L_: list):
    
    # 1. 连接
    pre_C = _connect(L)

    #2. 剪枝, 删除非频繁候选
    pre_C = _move_candidate_has_infrequent_subset(pre_C, k, L_)
     
    #3. 支持度计数
    C= _count_support(pre_C, D)
    
    return C

