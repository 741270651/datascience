
### FP-Growth算法实现

要点总结:

1. 设计节点的数据结构`treeNode`
2. 两次扫描事务数据库`D`,创建`FP tree`和`headerTable`
3. 寻找每个`item`的前缀路径,构成Conditional Pattern Bases(CPB), 把CPBs当做每个节点的local D,创建`item`的条件FP tree
4. 用条件FP tree的每条分支的节点的所有组合, 分别与后缀元素(item)求并集,形成频繁项集

形如*<数据挖掘: 概念与技术> P168表6.2*的挖掘结果保存在`resultTable`中.

credits: https://blog.csdn.net/gamer_gyt/article/details/51113753

#### Implementation start

------------


```python
inputfile = './data/AllElectronics_orders.xls'

data = pd.read_excel(inputfile, header=None)
```


```python
import pandas as pd
pd.options.display.max_colwidth = 300

import itertools
def _findsubsets(s,m):
    return set(itertools.combinations(s, m))
```


```python
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}
        self.tree = ''
 
    def inc(self, numOccur):
        self.count += numOccur
 
    def disp(self, ind=0): #ind: indentation
        
        print('*' * ind, self.name, ':', self.count)
        
        for child in self.children.values():
            child.disp(ind + 1)
            
    
    def stat(self, stat_list: list, ind=0): #ind: indentation
        
        if self.name != '{null}':
            stat_list.append(['*' * ind+self.name, self.count])
        
        for child in self.children.values():
            child.stat(stat_list, ind + 1)

        return stat_list
```


```python
def createTree(dataSet, minSup=2):
    ''' 创建FP树 '''

    headerTable = {}
    for trans, count in dataSet.items():
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + count

    for k in list(headerTable.keys()):
        if headerTable[k] < minSup:
            del(headerTable[k])
    
    freqItemSet = set(headerTable.keys())
    
    if len(freqItemSet) == 0:
        return None, None
    
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]
        
    retTree = treeNode('{null}', 1, None) 
     
    for tranSet, count in dataSet.items(): 
        for c in range(count): #这一步需要多注意
            localD = {}
            for item in tranSet: 
                if item in freqItemSet:
                    localD[item] = headerTable[item][0] 
            if len(localD) > 0:
                orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)] # 排序
                updateTree(orderedItems, retTree, headerTable, 1) # 更新FP树
    return retTree, headerTable
```


```python
def updateTree(orderedItems, inTree, headerTable, count):
    '''
    做的事情：
    把排序后列表[p|P]的第一项(p)添加到已有的FP tree上面，分两种情况：
        1. 根节点下面已经有p，则根节点下面的p节点计数增加1。
        2. 根节点下面没有p，那么在根节点下面创建新节点p。因为新创建该节点，需要更新headerTable中该项的nodeLink.
    对剩下的P递归调用上面的过程，直到P为空
    '''
    if orderedItems[0] in inTree.children:
        inTree.children[orderedItems[0]].inc(count)
    else:
        inTree.children[orderedItems[0]] = treeNode(orderedItems[0], count, inTree)
        if headerTable[orderedItems[0]][1] == None:
            headerTable[orderedItems[0]][1] = inTree.children[orderedItems[0]]
        else:
            updateHeader(headerTable[orderedItems[0]][1], inTree.children[orderedItems[0]])
            
    if len(orderedItems) > 1:
        updateTree(orderedItems[1:], inTree.children[orderedItems[0]], headerTable, count)
```


```python
def updateHeader(nodeToUpdate, targetNode):
    while (nodeToUpdate.nodeLink != None):
        nodeToUpdate = nodeToUpdate.nodeLink
    nodeToUpdate.nodeLink = targetNode
```


```python
def loadSimpDat(data):
    lines = []
    for r in data.iterrows():
        lines.append([i for i in r[1].dropna()])
    input_data = _pre_process(lines)
    return input_data
```


```python
def _pre_process(lines):
    input_data = {}
    for line in lines:
        s = frozenset(line)
        input_data[s] = input_data.get(s, 0) + 1
    return input_data
```


```python
simpDat = loadSimpDat(data)
myTree, myTable = createTree(simpDat, 2)
```


```python
def findPrefixPath(treeNode): #indPrefixPath is also "find conditional base"
    ''' 创建前缀路径 '''
    condPats = {} #条件基是一个字典, key是前缀路基(条件基), value是对应的计数
    while treeNode != None:
        prefixPath = []
        move_to_top_and_record(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        
        treeNode = treeNode.nodeLink

    return condPats
```


```python
def move_to_top_and_record(leafNode, prefixPath):
    if leafNode.parent != None: #当leftNode是根节点(null)时, 路径记录终止,记录不含根节点
        prefixPath.append(leafNode.name)
        move_to_top_and_record(leafNode.parent, prefixPath)
```


```python
def parse_tree_to_branches(inTree: list):
    
    tree = [[i[0].count('*') ,i[0].lstrip('*'), i[1]] for i in inTree]
    originTree = [[i[0].lstrip('*'), i[1]] for i in inTree]
    ct = 0
    new_tree = []
    for line  in tree:
        new_tree.append([ct] + line)
        ct += 1
    tree = new_tree
    
    length = len(tree)
    
    level_to_index_list = {}
    
    for line in tree:
        level_to_index_list.setdefault(line[1], [])
        level_to_index_list[line[1]].append(line[0])
    
    levelDescendList = sorted(level_to_index_list.keys(), reverse=True)
    
    lenOflDL = len(levelDescendList)
    
    flat_tree_index = []
    
    for levelNum in levelDescendList:
        for item in level_to_index_list[levelNum]:
            branch = [item]
            for beforeLevelNum in range(1, levelNum):
                localL = []
                for beforeItem in level_to_index_list[beforeLevelNum]:
                    
                    if beforeItem < branch[0]:
                        localL.append(beforeItem)                        
                branch = [max(localL)] + branch
            
            if flat_tree_index != []:
                flag = True
                for testItem in flat_tree_index:
                    if  set(branch) <= set(testItem):
                        flag = False
                if flag:
                    flat_tree_index.append(branch)
            else:
                flat_tree_index.append(branch)
    
    flat_tree = []
    for line in flat_tree_index:
        lineL  = []
        for item in line:
            lineL.append(originTree[item])
        flat_tree.append(lineL)
        
    return flat_tree
```


```python
def mineTree(inTree, headerTable, minSup, preFix):

    resultTable = pd.DataFrame(columns=['item', 'CPB', 'Contional FP-tree', 'Frequent Pattern'])

    global_L = [item[0] for item in sorted(list(headerTable.items()), key=lambda p: p[1][0])] #注意headerTable的构造
    # gloabl_L的排序根据支持度从小到大排序

    freqItemDict = {}
    
    for element in global_L: #1 element
        newFreqSet = preFix.copy()
        newFreqSet.add(element)
        
        condPattBases = findPrefixPath(headerTable[element][1]) #2 conditional base

        myCondTree, myHead = createTree(condPattBases, minSup) #创建一个条件FP树 #D的格式问题这里需要解决

        
        if myHead != None: #当element为I2时, myHead为None
            s = []
            myCondTree.stat(stat_list=s)
            flat_tree = parse_tree_to_branches(s)
        
            FP = {}
            for branch in flat_tree: #3. conditional tree
                # [['I1', 4], ['I2', 2]] branch
                localDict = dict(branch)
                baseSet  = set(localDict.keys())
                # baseSet = {'I1','I2'}
                lenOfbaseSet = len(baseSet)
                
                for ik in range(1, lenOfbaseSet+1):
                    subSets = _findsubsets(baseSet, ik)
                    for subSet in subSets:
                        fp = frozenset(set(subSet) | set([element]))
                        FP[fp] = FP.get(fp, 0) + min([localDict[ifk] for ifk in subSet])           
            
            freqItemDict[element]  = FP #4. Frequent Pattern

            #警告: append非原地操作
            resultTable = resultTable.append({'item': element, 'CPB': condPattBases, 'Contional FP-tree':  flat_tree, 'Frequent Pattern': FP}, ignore_index=True)

    return freqItemDict, resultTable
    
```


```python
def fpGrowth(dataSet, minSup=2):
    myFPtree, myHeaderTab = createTree(dataSet, minSup)
    freqItemDict, resultTable = mineTree(myFPtree, myHeaderTab, minSup, set([]))
    return freqItemDict,resultTable
```


```python
dataSet = loadSimpDat(data)
freqItems,resultTable = fpGrowth(dataSet)
```


```python
resultTable
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>item</th>
      <th>CPB</th>
      <th>Contional FP-tree</th>
      <th>Frequent Pattern</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>I5</td>
      <td>{('I1', 'I2'): 1, ('I3', 'I1', 'I2'): 1}</td>
      <td>[[[I1, 2], [I2, 2]]]</td>
      <td>{('I1', 'I5'): 2, ('I5', 'I2'): 2, ('I1', 'I5', 'I2'): 2}</td>
    </tr>
    <tr>
      <th>1</th>
      <td>I4</td>
      <td>{('I2'): 1, ('I1', 'I2'): 1}</td>
      <td>[[[I2, 2]]]</td>
      <td>{('I2', 'I4'): 2}</td>
    </tr>
    <tr>
      <th>2</th>
      <td>I1</td>
      <td>{('I2'): 2, ('I3'): 2, ('I3', 'I2'): 2}</td>
      <td>[[[I3, 4], [I2, 2]], [[I2, 2]]]</td>
      <td>{('I3', 'I1'): 4, ('I1', 'I2'): 4, ('I3', 'I1', 'I2'): 2}</td>
    </tr>
    <tr>
      <th>3</th>
      <td>I3</td>
      <td>{('I2'): 4}</td>
      <td>[[[I2, 4]]]</td>
      <td>{('I3', 'I2'): 4}</td>
    </tr>
  </tbody>
</table>
</div>




```python
from pprint import pformat
```


```python
print(pformat(freqItems))
```

    {'I1': {frozenset({'I3', 'I1'}): 4,
            frozenset({'I1', 'I2'}): 4,
            frozenset({'I3', 'I1', 'I2'}): 2},
     'I3': {frozenset({'I3', 'I2'}): 4},
     'I4': {frozenset({'I2', 'I4'}): 2},
     'I5': {frozenset({'I1', 'I5'}): 2,
            frozenset({'I5', 'I2'}): 2,
            frozenset({'I1', 'I5', 'I2'}): 2}}
    

#### Implementation end

------------
