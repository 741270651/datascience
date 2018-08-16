
### Aprior算法实现

算法实现要点：

* 通过pandas的`DataFrame`来实现对`项集（itemset）`的支持度计数
* 项集的集合`C`和`L`都是DataFrame对象

----------
#### Implementation start


```python
import pandas as pd
import numpy as np
```


```python
def get_C1(data: pd.DataFrame):
    
    C1 = {}
    for col in data:
        for i in data[col]:
            C1.setdefault(i, 0)
            C1[i] = C1[i] + 1

    if np.nan in C1:
        del C1[np.nan]
        
    C1 = pd.DataFrame({'itemset': [set([s]) for s in list(C1.keys())], 'count': list(C1.values())}) #注意这一步将dict的key转化为set的做法
    
    C1 = C1[['itemset','count']]
    
    return C1
```


```python
def trim_C(C, min_sup):
    
    L =  C[C['count'] >= min_sup]
    L = L[['itemset','count']]
    return L

import itertools

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


def _remove_candidate_has_infrequent_subset(pre_C: list, k, L_: dict):
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
    pre_C = _remove_candidate_has_infrequent_subset(pre_C, k, L_)
     
    #3. 支持度计数
    C= _count_support(pre_C, D)
    
    return C
```


```python
def confidence(L: pd.DataFrame, L_):
    itemset_ses = L['itemset']
    k = len(itemset_ses[0])
    conf_df = pd.DataFrame()
    for itemset in itemset_ses:        
        for ik in range(1, k):
            subsets = _findsubsets(itemset, ik)
            for subset in subsets:
                subset = set(subset)
                diffset = itemset - subset #求出差集
                c_itemset = L.loc[L['itemset'] == itemset, 'count'].values[0]
                _L = L_[ik]
                c_subset = _L.loc[_L['itemset'] == subset, 'count'].values[0]
                conf = c_itemset / c_subset
                conf_df= conf_df.append(
                    {'itemset': itemset, 'start': subset, 'subset_count': c_subset,  'end': diffset, 'itemset_count': c_itemset, 'conf': "{0:.0f}%".format(conf*100)},
                    ignore_index=True)
                # conf为numpy.float64对象
                # python内置round()函数针对numpy.float64不能正确工作，方法是先将numpy.float64转换为python原生的float
    conf_df = conf_df[['itemset', 'start','end', 'subset_count',  'itemset_count', 'conf']]
    return conf_df
```

#### Implementation end

--------------


```python
inputfile = '../data/AllElectronics_orders.xls'

data = pd.read_excel(inputfile, header=None)
```


```python
C1 = get_C1(data)
L1 = trim_C(C1,2)
#L1 = L1[['itemset','count']] #调整L1列的顺序, 顺序调整: count itemset -> itemset count
L_ = {1: L1}
C_ = {1: C1}
k = 1

while not L_[k].empty:
    #print(L_[k])
    C_[k+1] = L2C(L_[k], data, k+1, L_)
    L_[k+1] = trim_C(C_[k+1], 2) #指定最小支持度计数(min_sup)为2
    k += 1
```


```python
from IPython.display import display
for i in L_:
    display(L_[i])
```


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>itemset</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>{I1}</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>{I2}</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>{I4}</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>{I3}</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>{I5}</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>itemset</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>{I2, I1}</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>{I1, I3}</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>{I1, I5}</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>{I2, I4}</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>{I2, I3}</td>
      <td>4</td>
    </tr>
    <tr>
      <th>6</th>
      <td>{I2, I5}</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>itemset</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>{I2, I1, I3}</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>{I2, I1, I5}</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>itemset</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



```python
def sort_by_conf(df: pd.DataFrame):
    return df.reindex(index=df.conf.str.rstrip('%').astype(float).sort_values(ascending=False).index)


L3 = L_[3]
df = confidence(L3, L_)
df = sort_by_conf(df)
```


```python
df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>itemset</th>
      <th>start</th>
      <th>end</th>
      <th>subset_count</th>
      <th>itemset_count</th>
      <th>conf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11</th>
      <td>{I2, I1, I5}</td>
      <td>{I2, I5}</td>
      <td>{I1}</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>100%</td>
    </tr>
    <tr>
      <th>10</th>
      <td>{I2, I1, I5}</td>
      <td>{I1, I5}</td>
      <td>{I2}</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>100%</td>
    </tr>
    <tr>
      <th>6</th>
      <td>{I2, I1, I5}</td>
      <td>{I5}</td>
      <td>{I2, I1}</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>100%</td>
    </tr>
    <tr>
      <th>9</th>
      <td>{I2, I1, I5}</td>
      <td>{I2, I1}</td>
      <td>{I5}</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>50%</td>
    </tr>
    <tr>
      <th>5</th>
      <td>{I2, I1, I3}</td>
      <td>{I2, I3}</td>
      <td>{I1}</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>50%</td>
    </tr>
    <tr>
      <th>4</th>
      <td>{I2, I1, I3}</td>
      <td>{I1, I3}</td>
      <td>{I2}</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>50%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>{I2, I1, I3}</td>
      <td>{I2, I1}</td>
      <td>{I3}</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>50%</td>
    </tr>
    <tr>
      <th>8</th>
      <td>{I2, I1, I5}</td>
      <td>{I1}</td>
      <td>{I2, I5}</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>33%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>{I2, I1, I3}</td>
      <td>{I3}</td>
      <td>{I2, I1}</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>33%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>{I2, I1, I3}</td>
      <td>{I1}</td>
      <td>{I2, I3}</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>33%</td>
    </tr>
    <tr>
      <th>7</th>
      <td>{I2, I1, I5}</td>
      <td>{I2}</td>
      <td>{I1, I5}</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>29%</td>
    </tr>
    <tr>
      <th>0</th>
      <td>{I2, I1, I3}</td>
      <td>{I2}</td>
      <td>{I1, I3}</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>29%</td>
    </tr>
  </tbody>
</table>
</div>




```python
sort_by_conf(df[df['itemset'] == set(['I1','I2','I5'])])
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>itemset</th>
      <th>start</th>
      <th>end</th>
      <th>subset_count</th>
      <th>itemset_count</th>
      <th>conf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>{I2, I1, I5}</td>
      <td>{I5}</td>
      <td>{I2, I1}</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>100%</td>
    </tr>
    <tr>
      <th>10</th>
      <td>{I2, I1, I5}</td>
      <td>{I1, I5}</td>
      <td>{I2}</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>100%</td>
    </tr>
    <tr>
      <th>11</th>
      <td>{I2, I1, I5}</td>
      <td>{I2, I5}</td>
      <td>{I1}</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>100%</td>
    </tr>
    <tr>
      <th>9</th>
      <td>{I2, I1, I5}</td>
      <td>{I2, I1}</td>
      <td>{I5}</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>50%</td>
    </tr>
    <tr>
      <th>8</th>
      <td>{I2, I1, I5}</td>
      <td>{I1}</td>
      <td>{I2, I5}</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>33%</td>
    </tr>
    <tr>
      <th>7</th>
      <td>{I2, I1, I5}</td>
      <td>{I2}</td>
      <td>{I1, I5}</td>
      <td>7.0</td>
      <td>2.0</td>
      <td>29%</td>
    </tr>
  </tbody>
</table>
</div>


