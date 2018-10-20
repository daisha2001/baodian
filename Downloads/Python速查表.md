
## Python速查宝典
###  整理机器学习建模过程中常用的一些方法，方便日后工作更加高效。——by Sam

# 一、导入数据


```python
# 导入常用库numpy和pandas
import numpy as np
import pandas as pd
```


```python
# 常用的两种方式来查看一些陌生的方法
# np.info(np.ndarray.dtype)
# help(pd.read_csv)
```

## 1.1 文本文件

### 纯文本文件


```python
filename = 'xxx.text'
file = open(filename,mode='r')
text = file.read()
print(file.closed)
file.close()
print(text)
# 以只读方式读取文件，查看文件是否已经关闭，关闭文件
# open的mode方式还有很多，见 https://blog.csdn.net/pengyangyan/article/details/79966297
```


```python
# 使用上下文管理器 with
with open('xxx.txt','r') as file:
    print(file.readline())
    print(file.readline())
    print(file.readline())
    # 读取一行
```

### 表格数据


```python
# 单数据类型文件
filename = 'xxx.txt'
data = np.loadtxt(filename,
                 delimiter=',', # 用于分割各列值得字符
                 skiprows=2, # 用于跳过前2行
                 usecols=[0,2], # 读取并使用第1列和第3列
                 dtype=str) # 使用的数据类型
```


```python
# 多数据类型文件
filename = 'xxx.txt'
data = np.genfromtxt(filename,
                    delimiter=',',
                    names=True, # 导入时查找列名
                    dtype=None)
```


```python
# 使用pandas导入文本文件
filename = 'xxx.csv'
data = pd.read_csv(filename,
                  nrows=5, # 读取的行数
                  header=None, # 用哪行做列名，默认首行
                  sep='\t', # 用于分割各列的字符
                  comment='#', # 用于分割注释的字符
                  na_values=("") # 读取时，哪些值为NA/NaN)
```

## 1.2 Excel表


```python
file = 'xxx.xlsx'
data = pd.ExcelFile(file)
df_sheet2 = data.parse('1960-1966',
                      skiprow=[0],
                      names=['Country',
                             'AAM:War(2002)'])
```


```python
# 使用sheet_name属性访问表单名称
data.sheet_names
```

## 1.3 SAS 文件


```python
from sas7bdat import SAS7BDAT
with SAS7BDAT('xxx,sas7bdat') as file:
    df_sas = file.to_data_frame()
```

## 1.4 探索数据

### Numpy数组


```python
# 查看数据元素的数据类型
data_array.dtype
# 查看数组维度
data_array.shape
# 查看数据长度
len(data_array)
```

### Pandas数据框


```python
# 返回数据框的前几行，默认5行
df.head()
# 返回数据框的后几行，默认5行
df.tail()
# 查看数据框的索引
df.index
# 查看数据框的列名
df.columns
# 查看数据框各列的信息
df.info()
# 将数据框转为Numpy数组
data_array = data.values
```

### 探索字典


```python
# 输出字典的键值
print(mat.keys())
# 输出字典的键值
for key in data.keys():
    print(key)
# 输出字典的值
pickled_data.values()
# 返回由元祖构成字典键值对列表
print(mat.items())
```

### 探索文件系统


```python
# 列出当前目录里的文件和文件夹
!ls
```

    Day 1-7.ipynb                      US-Baby-Names-1880-2010-master.zip
    Python速查表.ipynb                 [34mml-1m[m[m
    [34mUS-Baby-Names-1880-2010-master[m[m     ml-1m.zip



```python
# 改变当前工作目录
%cd .
```


```python
# 返回当前工作目录的路径
%pwd
```




    '/Users/yongsenlin/利用python进行数据分析'




```python
# OS库的一些操作
import OS
path = '/usr/tmp' 
wd = os.getcwd() # 将当前工作目录存为字符串
os.listdir(wd) # 将目录里的内容输出为列表
os.chdir(path) # 改变当前工作目录
os.rename('t1.txt',
          't2.txt') # 重命名文件
os.remove('t2.txt') # 删除文件
os.mkdir('newdir') # 新建文件夹
```

# 二、numpy库


```python
import numpy as np
```

## 创建数组


```python
a = np.array([1,2,3])
a
```




    array([1, 2, 3])




```python
b = np.array([(1.5,2,3),(4,5,6)],dtype=float)
b
```




    array([[ 1.5,  2. ,  3. ],
           [ 4. ,  5. ,  6. ]])




```python
c = np.array([[(1.3,2,3),(2.2,3.4,2)],[(1,2,3),(4,5,6)]],dtype=float)
c
```




    array([[[ 1.3,  2. ,  3. ],
            [ 2.2,  3.4,  2. ]],
    
           [[ 1. ,  2. ,  3. ],
            [ 4. ,  5. ,  6. ]]])



## 初始化占位符


```python
# 创建值为0的数组
np.zeros((3,4))
# 创建值为1的数组
np.ones((2,3,4),dtype=np.int16)
# 创建均匀间隔的数组（步进值）
d = np.arange(10,25,5)
# 创建均匀间隔的数组（样本数）
np.linspace(0,2,9)
# 创建常数数组
e = np.full((2,2),7)
# 创建2X2单位矩阵
f = np.eye(2)
# 创建随机值的数组
np.random.random((2,2))
```

## 输入/输出


```python
# 将数组保存到磁盘中
np.save('my_array',a)
# 多个数组保存到一个文件中
np.savez('array.npz',a,b)
# 加载数组文件
np.load('my_array.npy')

# 保存与载入文本文件
np.loadtxt('myfile.txt')
np.genfromtxt('my_file.csv',delimiter=',')
np.savetxt('myarray.txt',a,delimiter=" ")
```

## 数据类型


```python
# 带符号的64位整数
np.int64
# 标准双精度浮点数
np.float32
# 显示为128位浮点数的复数
np.complex
# 布尔值
np.bool
# Python对象
np.object
# 固定长度字符串
np.string_
# 固定长度Unicode
np.unicode_
```

## 数组信息


```python
# 数组形状，几行几列
a.shape
# 数组长度
len(a)
# 几维数组
a.ndim
# 数组有多少元素
a.size
# 数据类型
a.dtype
# 数据类型的名字
a.dtype.name
# 数据类型转换
a.astype(int)
```

## 数组计算

### 算术计算


```python
# 加法
a+b
np.add(a,b)
# 减法
a-b
np.substract(a,b)
# 乘法
a*b
np.multiply(a,b)
# 除法
a/b
np.divide(a,b)
# 幂
np.exp(a)
# 平方根
np.sqrt(a)
# 正弦
np.sin(a)
# 余弦
np.cos(a)
# 自然对数
np.log(a)
# 点积
np.dot(a)
```

### 比较计算


```python
a = np.array([(1,2,3),(4,5,6)],dtype=float)
b = np.array([(0.5,2,3),(1,5,6)],dtype=float)
```


```python
a == b
```




    array([[False,  True,  True],
           [False,  True,  True]], dtype=bool)




```python
a < 2
```




    array([[ True, False, False],
           [False, False, False]], dtype=bool)




```python
np.array_equal(a,b)
```




    False



### 聚合函数


```python
a = np.array([(1,2,3),(4,5,6)],dtype=float)
a
```




    array([[ 1.,  2.,  3.],
           [ 4.,  5.,  6.]])




```python
# 数组汇总
a.sum()
```




    21.0




```python
# 数组最小值
a.min()
```




    1.0




```python
# 数组最大值
print(a.max(axis=0))
print(a.max(axis=1))
```

    [ 4.  5.  6.]
    [ 3.  6.]



```python
# 数组元素的累加值
print(a.cumsum(axis=0))
print(a.cumsum(axis=1))
```

    [[ 1.  2.  3.]
     [ 5.  7.  9.]]
    [[  1.   3.   6.]
     [  4.   9.  15.]]



```python
# 平均数
a.mean()
```




    3.5




```python
# 标准差
np.std(a)
```




    1.707825127659933



### 数组排序


```python
a = np.array([(1,14,3),(4,9,6)],dtype=float)
a
```




    array([[  1.,  14.,   3.],
           [  4.,   9.,   6.]])




```python
a.sort()
a
```




    array([[  1.,   3.,  14.],
           [  4.,   6.,   9.]])




```python
a.sort(axis=0)
a
```




    array([[  1.,   3.,   9.],
           [  4.,   6.,  14.]])



## 子集、切片、索引


```python
a = np.array([(1.5,2,3),(4,5,6),(7.7,8,9)],dtype=float)
a
```




    array([[ 1.5,  2. ,  3. ],
           [ 4. ,  5. ,  6. ],
           [ 7.7,  8. ,  9. ]])




```python
a[0:2,1]
```




    array([ 2.,  5.])




```python
a[:1]
# 等同 a[0:1,:]
```




    array([[ 1.5,  2. ,  3. ]])




```python
a[1:]
# 等同 a[1:,:]
```




    array([[ 4. ,  5. ,  6. ],
           [ 7.7,  8. ,  9. ]])




```python
a[2,...]
```




    array([ 7.7,  8. ,  9. ])




```python
# 条件索引
a[a<5]
```




    array([ 1.5,  2. ,  3. ,  4. ])



## 数组操作


```python
a = np.array([(1.5,2,3),(4,5,6),(7.7,8,9)],dtype=float)
a
```




    array([[ 1.5,  2. ,  3. ],
           [ 4. ,  5. ,  6. ],
           [ 7.7,  8. ,  9. ]])




```python
# 转置数组
i = np.transpose(a)
i.T
```




    array([[ 1.5,  2. ,  3. ],
           [ 4. ,  5. ,  6. ],
           [ 7.7,  8. ,  9. ]])




```python
# 改变数组形状
a.ravel()
a.reshape(1,9)
```




    array([[ 1.5,  2. ,  3. ,  4. ,  5. ,  6. ,  7.7,  8. ,  9. ]])



# 三、Pandas库


```python
import pandas as pd
```


```python
# Series 序列
# 存储任意类型数据的一维数组
s = pd.Series([3,-5,7,4],index=['a','b','c','d'])
s
```




    a    3
    b   -5
    c    7
    d    4
    dtype: int64




```python
# DataFrame - 数据框
data = {'Country':['Belgium','India','Brazil'],
       'Capital':['Brussels','New Delhi','Brasilia'],
       'Population':[11190846,1303171035,207847528]}
df = pd.DataFrame(data,
                  columns=['Country','Capital','Population'])
df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Capital</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Belgium</td>
      <td>Brussels</td>
      <td>11190846</td>
    </tr>
    <tr>
      <th>1</th>
      <td>India</td>
      <td>New Delhi</td>
      <td>1303171035</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Brazil</td>
      <td>Brasilia</td>
      <td>207847528</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 按位置
print(df.iloc[[0],[0]])
print('\n')
# 按标签
print(df.loc[[0],['Country']])
print('\n')
# 布尔索引
print(s[~(s>1)])
```

       Country
    0  Belgium
    
    
       Country
    0  Belgium
    
    
    b   -5
    dtype: int64



```python
# 删除数据
# 按索引删除序列的值
s1 = s.drop(['a','c'])
print(s1)
print('\n')

# 按列名删除数据框的列
df1 = df.drop('Country',axis=1)
print(df1)
```

    b   -5
    d    4
    dtype: int64
    
    
         Capital  Population
    0   Brussels    11190846
    1  New Delhi  1303171035
    2   Brasilia   207847528



```python
# 排序和排名
# 按索引排序
df.sort_index()
# 按某列的值排序
df.sort_values(by='Country')
# 数据框排名，默认升序
df.rank()
```


```python
df.rank()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Capital</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>



## 输入/输出


```python
# 读取/写入CSV
pd.read_csv('file.csv',header=None,nrows=5)
df.to_csv('MyDataFrame.csv')
```


```python
# 读取/写入Excel
pd.read_excel('file.xlsx')
pd.to_excel('dir/myDataFrame.xlsx',sheet_name='Sheet1')
```


```python
# 读取内含多个表的Excel
xlsx = pd.ExcelFile('file.xls')
df = pd.read_excel(xlsx,'Sheet1')
```

## 查询序列与数据框的信息

### 基础信息查询


```python
# （行，列）
df.shape
# 获取索引
df.index
# 获取列名
df.columns
# 获取数据框基本信息
df.info()
# 非NA值的数量
df.count()
# 合计
df.sum()
# 累计
df.cumsum()
# 最小值除以最大值
df.min()/df.max()
# 索引最小值除以索引最大值
df.idxmin()/df.idxmax()
# 基础统计数据
df.describe()
# 平均值
df.mean()
# 中位数
df.median()
```

### 应用函数


```python
df
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Capital</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Belgium</td>
      <td>Brussels</td>
      <td>11190846</td>
    </tr>
    <tr>
      <th>1</th>
      <td>India</td>
      <td>New Delhi</td>
      <td>1303171035</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Brazil</td>
      <td>Brasilia</td>
      <td>207847528</td>
    </tr>
  </tbody>
</table>
</div>




```python
f = lambda x:x*2 # 匿名函数lambda
df1 = df.apply(f) # 应用函数
df1
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Capital</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BelgiumBelgium</td>
      <td>BrusselsBrussels</td>
      <td>22381692</td>
    </tr>
    <tr>
      <th>1</th>
      <td>IndiaIndia</td>
      <td>New DelhiNew Delhi</td>
      <td>2606342070</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BrazilBrazil</td>
      <td>BrasiliaBrasilia</td>
      <td>415695056</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2 = df.applymap(f) # 对每个单元格应用函数
df2
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Capital</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BelgiumBelgium</td>
      <td>BrusselsBrussels</td>
      <td>22381692</td>
    </tr>
    <tr>
      <th>1</th>
      <td>IndiaIndia</td>
      <td>New DelhiNew Delhi</td>
      <td>2606342070</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BrazilBrazil</td>
      <td>BrasiliaBrasilia</td>
      <td>415695056</td>
    </tr>
  </tbody>
</table>
</div>



## 数据重塑

### 数据透视


```python
data2 = {'Date':['2016-03-01','2016-03-02','2016-03-01','2016-03-03','2016-03-02','2016-03-03'],
        'Type':['a','b','c','a','a','c'],
        'Value':[11.432,13.031,20.784,99.906,1.303,20.784]}
df2 = pd.DataFrame(data2,
                   columns=['Date','Type','Value'])
df2
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Type</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-03-01</td>
      <td>a</td>
      <td>11.432</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-03-02</td>
      <td>b</td>
      <td>13.031</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-03-01</td>
      <td>c</td>
      <td>20.784</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-03-03</td>
      <td>a</td>
      <td>99.906</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-03-02</td>
      <td>a</td>
      <td>1.303</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2016-03-03</td>
      <td>c</td>
      <td>20.784</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 行变列,and Index could not contains duplicate entries
df3 = df2.pivot(index='Date',
                columns='Type',
                values='Value')
df3
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Type</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-03-01</th>
      <td>11.432</td>
      <td>NaN</td>
      <td>20.784</td>
    </tr>
    <tr>
      <th>2016-03-02</th>
      <td>1.303</td>
      <td>13.031</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2016-03-03</th>
      <td>99.906</td>
      <td>NaN</td>
      <td>20.784</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 数据透视
df4 = pd.pivot_table(df2,
                     values='Value',
                     index='Date',
                     columns='Type',
                     aggfunc='count') # np.sum np.mean
df4
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Type</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-03-01</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2016-03-02</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2016-03-03</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



### 融合


```python
df5 = pd.melt(df2,
              id_vars=['Date'],
              value_vars=['Type','Value'],
              value_name='Observations')
df5
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>variable</th>
      <th>Observations</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-03-01</td>
      <td>Type</td>
      <td>a</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-03-02</td>
      <td>Type</td>
      <td>b</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-03-01</td>
      <td>Type</td>
      <td>c</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-03-03</td>
      <td>Type</td>
      <td>a</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-03-02</td>
      <td>Type</td>
      <td>a</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2016-03-03</td>
      <td>Type</td>
      <td>c</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2016-03-01</td>
      <td>Value</td>
      <td>11.432</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2016-03-02</td>
      <td>Value</td>
      <td>13.031</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2016-03-01</td>
      <td>Value</td>
      <td>20.784</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2016-03-03</td>
      <td>Value</td>
      <td>99.906</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2016-03-02</td>
      <td>Value</td>
      <td>1.303</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2016-03-03</td>
      <td>Value</td>
      <td>20.784</td>
    </tr>
  </tbody>
</table>
</div>



### 高级索引


```python
df3
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Type</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-03-01</th>
      <td>11.432</td>
      <td>NaN</td>
      <td>20.784</td>
    </tr>
    <tr>
      <th>2016-03-02</th>
      <td>1.303</td>
      <td>13.031</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2016-03-03</th>
      <td>99.906</td>
      <td>NaN</td>
      <td>20.784</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 选择任一值大于1的列
dft = df3.loc[:,(df3>1).any()]
dft
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Type</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-03-01</th>
      <td>11.432</td>
      <td>NaN</td>
      <td>20.784</td>
    </tr>
    <tr>
      <th>2016-03-02</th>
      <td>1.303</td>
      <td>13.031</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2016-03-03</th>
      <td>99.906</td>
      <td>NaN</td>
      <td>20.784</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 选择所有值都大于1的列
dft = df3.loc[:,(df3>1).all()]
dft
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Type</th>
      <th>a</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-03-01</th>
      <td>11.432</td>
    </tr>
    <tr>
      <th>2016-03-02</th>
      <td>1.303</td>
    </tr>
    <tr>
      <th>2016-03-03</th>
      <td>99.906</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 选择含Nan值的列
dft = df3.loc[:,df3.isnull().any()]
dft
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Type</th>
      <th>b</th>
      <th>c</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-03-01</th>
      <td>NaN</td>
      <td>20.784</td>
    </tr>
    <tr>
      <th>2016-03-02</th>
      <td>13.031</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2016-03-03</th>
      <td>NaN</td>
      <td>20.784</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 选择不含Nan值的列
dft = df3.loc[:,df3.notnull().all()]
dft
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Type</th>
      <th>a</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-03-01</th>
      <td>11.432</td>
    </tr>
    <tr>
      <th>2016-03-02</th>
      <td>1.303</td>
    </tr>
    <tr>
      <th>2016-03-03</th>
      <td>99.906</td>
    </tr>
  </tbody>
</table>
</div>



### 索引


```python
# 设置索引
dft = df.set_index('Country')
dft
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Capital</th>
      <th>Population</th>
    </tr>
    <tr>
      <th>Country</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Belgium</th>
      <td>Brussels</td>
      <td>11190846</td>
    </tr>
    <tr>
      <th>India</th>
      <td>New Delhi</td>
      <td>1303171035</td>
    </tr>
    <tr>
      <th>Brazil</th>
      <td>Brasilia</td>
      <td>207847528</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 取消索引
df4 = dft.reset_index()
df4
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Capital</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Belgium</td>
      <td>Brussels</td>
      <td>11190846</td>
    </tr>
    <tr>
      <th>1</th>
      <td>India</td>
      <td>New Delhi</td>
      <td>1303171035</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Brazil</td>
      <td>Brasilia</td>
      <td>207847528</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 重命名DataFrame列名
df5 = df.rename(index=str,
                columns={'Country':'cntry',
                         'Capital':'cptl',
                         'Population':'ppltn'})
df5
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cntry</th>
      <th>cptl</th>
      <th>ppltn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Belgium</td>
      <td>Brussels</td>
      <td>11190846</td>
    </tr>
    <tr>
      <th>1</th>
      <td>India</td>
      <td>New Delhi</td>
      <td>1303171035</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Brazil</td>
      <td>Brasilia</td>
      <td>207847528</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 重置索引
s2 = s.reindex(['a','c','d','e','b'])
s2
```




    a    3.0
    c    7.0
    d    4.0
    e    NaN
    b   -5.0
    dtype: float64




```python
# 前向填充
df6 = df.reindex(range(8),method='ffill')
df6
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Capital</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Belgium</td>
      <td>Brussels</td>
      <td>11190846</td>
    </tr>
    <tr>
      <th>1</th>
      <td>India</td>
      <td>New Delhi</td>
      <td>1303171035</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Brazil</td>
      <td>Brasilia</td>
      <td>207847528</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Brazil</td>
      <td>Brasilia</td>
      <td>207847528</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Brazil</td>
      <td>Brasilia</td>
      <td>207847528</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Brazil</td>
      <td>Brasilia</td>
      <td>207847528</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Brazil</td>
      <td>Brasilia</td>
      <td>207847528</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Brazil</td>
      <td>Brasilia</td>
      <td>207847528</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 后向填充
df6 = df.reindex(range(4),method='bfill')
df6
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Capital</th>
      <th>Population</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Belgium</td>
      <td>Brussels</td>
      <td>1.119085e+07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>India</td>
      <td>New Delhi</td>
      <td>1.303171e+09</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Brazil</td>
      <td>Brasilia</td>
      <td>2.078475e+08</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 返回唯一值
s.unique()
```




    array([ 3, -5,  7,  4])




```python
# 查找重复值
print(df2)
print('\n')
print(df2.duplicated('Type',keep=False))
# keep=‘frist’：除了第一次出现外，其余相同的被标记为重复
# keep='last'：除了最后一次出现外，其余相同的被标记为重复
```

             Date Type   Value
    0  2016-03-01    a  11.432
    1  2016-03-02    b  13.031
    2  2016-03-01    c  20.784
    3  2016-03-03    a  99.906
    4  2016-03-02    a   1.303
    5  2016-03-03    c  20.784
    
    
    0     True
    1    False
    2     True
    3     True
    4     True
    5     True
    dtype: bool



```python
# 去除重复值,保留最后一条记录
df7 = df2.drop_duplicates('Type',keep='last')
df7
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Type</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>2016-03-02</td>
      <td>b</td>
      <td>13.031</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-03-02</td>
      <td>a</td>
      <td>1.303</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2016-03-03</td>
      <td>c</td>
      <td>20.784</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 查找重复索引
df.index.duplicated()
```




    array([False, False, False], dtype=bool)



### 数据分组


```python
data2 = {'Date':['2016-03-01','2016-03-02','2016-03-01','2016-03-03','2016-03-02','2016-03-03'],
        'Type':['a','b','c','a','a','c'],
        'Value':[11.432,13.031,20.784,99.906,1.303,20.784]}
df2 = pd.DataFrame(data2,
                   columns=['Date','Type','Value'])
df2
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Type</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-03-01</td>
      <td>a</td>
      <td>11.432</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-03-02</td>
      <td>b</td>
      <td>13.031</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-03-01</td>
      <td>c</td>
      <td>20.784</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-03-03</td>
      <td>a</td>
      <td>99.906</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-03-02</td>
      <td>a</td>
      <td>1.303</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2016-03-03</td>
      <td>c</td>
      <td>20.784</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 聚合
a = df2.groupby(by=['Date','Type']).mean()
a
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Value</th>
    </tr>
    <tr>
      <th>Date</th>
      <th>Type</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">2016-03-01</th>
      <th>a</th>
      <td>11.432</td>
    </tr>
    <tr>
      <th>c</th>
      <td>20.784</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">2016-03-02</th>
      <th>a</th>
      <td>1.303</td>
    </tr>
    <tr>
      <th>b</th>
      <td>13.031</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">2016-03-03</th>
      <th>a</th>
      <td>99.906</td>
    </tr>
    <tr>
      <th>c</th>
      <td>20.784</td>
    </tr>
  </tbody>
</table>
</div>




```python
df4 = pd.pivot_table(df2,
                     values='Value',
                     index='Date',
                     columns='Type',
                     aggfunc='count') # np.sum np.mean
df4
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Type</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-03-01</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2016-03-02</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2016-03-03</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
customsum = lambda x:(x+x%2)
a = df4.groupby(level=0).transform(customsum)
a
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Type</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-03-01</th>
      <td>2.0</td>
      <td>NaN</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2016-03-02</th>
      <td>2.0</td>
      <td>2.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2016-03-03</th>
      <td>2.0</td>
      <td>NaN</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>



### 缺失值


```python
# 去除缺失值NAN
df.dropna() 
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-13-11a1fe8da150> in <module>()
          1 # 去除缺失值NAN
    ----> 2 df.dropna()
          3 # 用预设值填充缺失值NaN
          4 a = df4.fillna(df4.mean())
          5 a


    NameError: name 'df' is not defined



```python
# 用预设值填充缺失值NaN
a = df4.fillna(df4.sum())
a
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Type</th>
      <th>a</th>
      <th>b</th>
      <th>c</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-03-01</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2016-03-02</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2016-03-03</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 用一个值替换另一个值
print("*"*25)
print(df2)
a= df2.replace("a","f")
print("*"*25)
print(a)
```

    *************************
             Date Type   Value
    0  2016-03-01    a  11.432
    1  2016-03-02    b  13.031
    2  2016-03-01    c  20.784
    3  2016-03-03    a  99.906
    4  2016-03-02    a   1.303
    5  2016-03-03    c  20.784
    *************************
             Date Type   Value
    0  2016-03-01    f  11.432
    1  2016-03-02    b  13.031
    2  2016-03-01    c  20.784
    3  2016-03-03    f  99.906
    4  2016-03-02    f   1.303
    5  2016-03-03    c  20.784



```python
data2 = {'Date':['2016-03-01','2016-03-02','2016-03-01','2016-03-03','2016-03-02','2016-03-03'],
        'Type':['a','b','c','a','a','c'],
        'Value':[11.432,13.031,20.784,99.906,1.303,20.784]}
df2 = pd.DataFrame(data2,
                   columns=['Date','Type','Value'])
df2
```

### 合并数据


```python
data1 = {'X1':['a','b','c'],
        'X2':[11.432,1.303,99.906]}
data1 = pd.DataFrame(data1,columns=['X1','X2'])
data1
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X1</th>
      <th>X2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>11.432</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1.303</td>
    </tr>
    <tr>
      <th>2</th>
      <td>c</td>
      <td>99.906</td>
    </tr>
  </tbody>
</table>
</div>




```python
data2 = {'X1':['a','b','d'],
        'X2':[20.784,'NaN',20.784]}
data2 = pd.DataFrame(data2,columns=['X1','X2'])
data2
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>X1</th>
      <th>X2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>a</td>
      <td>20.784</td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>d</td>
      <td>20.784</td>
    </tr>
  </tbody>
</table>
</div>




```python
a1 = pd.merge(data1,
              data2,
              how='left',
              on='X1')
a2 = pd.merge(data1,
              data2,
              how='right',
              on='X1')
a3 = pd.merge(data1,
              data2,
              how='inner',
              on='X1')
a4 = pd.merge(data1,
              data2,
              how='outer',
              on='X1')
print(a1)
print("*"*30 + '\n')
print(a2)
print("*"*30 + '\n')
print(a3)
print("*"*30 + '\n')
print(a4)
print("*"*30 + '\n')
```

      X1    X2_x    X2_y
    0  a  11.432  20.784
    1  b   1.303     NaN
    2  c  99.906     NaN
    ******************************
    
      X1    X2_x    X2_y
    0  a  11.432  20.784
    1  b   1.303     NaN
    2  d     NaN  20.784
    ******************************
    
      X1    X2_x    X2_y
    0  a  11.432  20.784
    1  b   1.303     NaN
    ******************************
    
      X1    X2_x    X2_y
    0  a  11.432  20.784
    1  b   1.303     NaN
    2  c  99.906     NaN
    3  d     NaN  20.784
    ******************************
    


### 拼接-concatenate 


```python
# 纵向
s.append(s2)
# 横向
pd.concat([s,s2],axis=1,keys=['One','Two'])
pd.concat([data1,data2],axis=1,join='inner')
```

### 日期


```python
df2.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6 entries, 0 to 5
    Data columns (total 3 columns):
    Date     6 non-null object
    Type     6 non-null object
    Value    6 non-null float64
    dtypes: float64(1), object(2)
    memory usage: 224.0+ bytes



```python
df2['Date']=pd.to_datetime(df2['Date'])
df2.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6 entries, 0 to 5
    Data columns (total 3 columns):
    Date     6 non-null datetime64[ns]
    Type     6 non-null object
    Value    6 non-null float64
    dtypes: datetime64[ns](1), float64(1), object(1)
    memory usage: 224.0+ bytes


# 四、Scikit-learn

### Scikit-learn是开源的Python库，通过统一的界面实现机器学习、预处理、交叉验证及可视化算法。


## 加载数据

### Scikit-learn处理的数据是储存在Numpy数据或者SciPy稀疏矩阵的数字，还支持Pandas数据框等可转换为数据数组的其他数据类型。


```python
import numpy as np
X = np.random.random((10,5))
y = np.array(['M','M','F','F','M','F','M','M','F','F'])
X
```




    array([[ 0.59860167,  0.27634186,  0.96705409,  0.13845522,  0.79911469],
           [ 0.04612609,  0.92248277,  0.2336466 ,  0.62711518,  0.55200292],
           [ 0.67953002,  0.0535348 ,  0.29133439,  0.65803389,  0.1779828 ],
           [ 0.46532977,  0.61473673,  0.55722408,  0.34851012,  0.41344635],
           [ 0.57241382,  0.24572994,  0.62927885,  0.79895657,  0.69766902],
           [ 0.89019382,  0.27731088,  0.31132927,  0.73827061,  0.98169583],
           [ 0.69930699,  0.72951348,  0.55529067,  0.42271121,  0.11885605],
           [ 0.70920953,  0.78690679,  0.04769703,  0.76326608,  0.56393779],
           [ 0.16243085,  0.29764679,  0.96881633,  0.40524756,  0.19952772],
           [ 0.22004726,  0.83805926,  0.52459605,  0.80126906,  0.97001867]])




```python
X[X<0.7]=0
X
```




    array([[ 0.        ,  0.        ,  0.96705409,  0.        ,  0.79911469],
           [ 0.        ,  0.92248277,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.79895657,  0.        ],
           [ 0.89019382,  0.        ,  0.        ,  0.73827061,  0.98169583],
           [ 0.        ,  0.72951348,  0.        ,  0.        ,  0.        ],
           [ 0.70920953,  0.78690679,  0.        ,  0.76326608,  0.        ],
           [ 0.        ,  0.        ,  0.96881633,  0.        ,  0.        ],
           [ 0.        ,  0.83805926,  0.        ,  0.80126906,  0.97001867]])



### 划分训练验证


```python
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)
```

## 数据预处理

### 标准化


```python
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler().fit(X_train)
standardized_X = scaler.transform(X_train)
standardized_X_test = scaler.transform(X_test)
```


```python
scaler
```




    StandardScaler(copy=True, with_mean=True, with_std=True)



### 归一化


```python
from sklearn.preprocessing import Normalizer
scaler = Normalizer().fit(X_train)
normalized_X = scaler.transform(X_train)
normalized_X_test = scaler.transform(X_test)
```


```python
scaler
```




    Normalizer(copy=True, norm='l2')



### 二值化


```python
from sklearn.preprocessing import Binarizer
binarizer = Binarizer(threshold=0.0).fit(X)
binary_X = binarizer.transform(X)
```


```python
X
```




    array([[ 0.        ,  0.        ,  0.96705409,  0.        ,  0.79911469],
           [ 0.        ,  0.92248277,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.79895657,  0.        ],
           [ 0.89019382,  0.        ,  0.        ,  0.73827061,  0.98169583],
           [ 0.        ,  0.72951348,  0.        ,  0.        ,  0.        ],
           [ 0.70920953,  0.78690679,  0.        ,  0.76326608,  0.        ],
           [ 0.        ,  0.        ,  0.96881633,  0.        ,  0.        ],
           [ 0.        ,  0.83805926,  0.        ,  0.80126906,  0.97001867]])




```python
binary_X
```




    array([[ 0.,  0.,  1.,  0.,  1.],
           [ 0.,  1.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  1.,  0.],
           [ 1.,  0.,  0.,  1.,  1.],
           [ 0.,  1.,  0.,  0.,  0.],
           [ 1.,  1.,  0.,  1.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  0.,  1.,  1.]])



### 编码分类特征


```python
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
y = enc.fit_transform(y)
y
```




    array([1, 1, 0, 0, 1, 0, 1, 1, 0, 0])



### 输入缺失值


```python
X_train
```




    array([[ 0.        ,  0.83805926,  0.        ,  0.80126906,  0.97001867],
           [ 0.        ,  0.92248277,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.72951348,  0.        ,  0.        ,  0.        ],
           [ 0.70920953,  0.78690679,  0.        ,  0.76326608,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.96705409,  0.        ,  0.79911469],
           [ 0.89019382,  0.        ,  0.        ,  0.73827061,  0.98169583]])




```python
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values=0,strategy='mean',axis=0)
imp.fit_transform(X_train)
```




    array([[ 0.79970167,  0.83805926,  0.96705409,  0.80126906,  0.97001867],
           [ 0.79970167,  0.92248277,  0.96705409,  0.76760192,  0.91694306],
           [ 0.79970167,  0.72951348,  0.96705409,  0.76760192,  0.91694306],
           [ 0.70920953,  0.78690679,  0.96705409,  0.76326608,  0.91694306],
           [ 0.79970167,  0.81924057,  0.96705409,  0.76760192,  0.91694306],
           [ 0.79970167,  0.81924057,  0.96705409,  0.76760192,  0.79911469],
           [ 0.89019382,  0.81924057,  0.96705409,  0.73827061,  0.98169583]])



### 生成多项式特征


```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(5)
poly.fit_transform(X)
```




    array([[ 1.        ,  0.        ,  0.        , ...,  0.        ,
             0.        ,  0.32587089],
           [ 1.        ,  0.        ,  0.92248277, ...,  0.        ,
             0.        ,  0.        ],
           [ 1.        ,  0.        ,  0.        , ...,  0.        ,
             0.        ,  0.        ],
           ..., 
           [ 1.        ,  0.70920953,  0.78690679, ...,  0.        ,
             0.        ,  0.        ],
           [ 1.        ,  0.        ,  0.        , ...,  0.        ,
             0.        ,  0.        ],
           [ 1.        ,  0.        ,  0.83805926, ...,  0.5859992 ,
             0.70941234,  0.85881666]])




```python
X
```




    array([[ 0.        ,  0.        ,  0.96705409,  0.        ,  0.79911469],
           [ 0.        ,  0.92248277,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        ,  0.79895657,  0.        ],
           [ 0.89019382,  0.        ,  0.        ,  0.73827061,  0.98169583],
           [ 0.        ,  0.72951348,  0.        ,  0.        ,  0.        ],
           [ 0.70920953,  0.78690679,  0.        ,  0.76326608,  0.        ],
           [ 0.        ,  0.        ,  0.96881633,  0.        ,  0.        ],
           [ 0.        ,  0.83805926,  0.        ,  0.80126906,  0.97001867]])



## 创建模型

### 有监督学习评估器


```python
# 线性回归
from sklearn.linear_model import LinearRegression
lr = LinearRegression(normalize=True)

# 支持向量机（SVM）
from sklearn.svm import SVC
svc = SVC(kernel='linear')

# 朴素贝叶斯
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

# KNN
from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
```

### 无监督学习评估器


```python
# 主成分分析（PCA）
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)

# K Means
from sklearn.cluster import KMeans
k_means = KMeans(n_clusters=3,random_state=0)
```

## 模型拟合


```python
# 有监督学习
lr.fit(X,y)
knn.fit(X,y)
svc.fit(X,y)

# 无监督学习
k_means.fit(X_train)
pca_model = pca.fit_transform(X_train)
```

## 预测


```python
# 有监督评估器
y_pred = svc.predict(np.random.random((2,5)))
y_pred = lr.predict(X_test)
y_pred = knn.predict_proba(X_test)

# 无监督评估器
y_pred = k_means.predict(X_test)
```

## 评估模型性能

### 分类指标


```python
# 准确率
knn.score(X_test,y_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

# 分类预估评价函数(精确度、召回率、F1分数及支持率)
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

# 混淆矩阵
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))
```

### 回归指标


```python
# 平均绝对误差
from sklearn.metrics import mean_absolute_error
y_true=[2,-0.2,2]
y_pred=[3,-0.2,2]
mean_absolute_error(y_true,y_pred)

# 均方误差
from sklearn.metrics import mean_squared_error
mean_squared_error(y_true,y_pred)

# R2评分
from sklearn.metrics import r2_score
r2_score(y_true,y_pred)
```




    0.33333333333333331



### 交叉验证


```python
from sklearn.cross_validation import cross_val_score
print(cross_val_score(knn,X_train,y_train,cv=4))
print(cross_val_score(lr,X,y,cv=2))
```

### 模型调参


```python
# 网格搜索
from sklearn.grid_search import GridSearchCV
params = {'n_neighbors':np.arange(1,3),
        'metric':['euclidean','cityblock']}
grid = GridSearchCV(estimator=knn,
                    param_grid=params)
grid.fit(X_train,y_train)
print(grid.best_score_)
print(grid.best_estimator_.n_neighbors)

# 随机参数优化
from sklearn.grid_search import RandomizedSearchCV
params = {'n_neighbors':range(1,5),
          'weights':['uniform','distance']}
rsearch = RandomizeSearchCV(estimator=knn,
                            param_distributions=params,
                            cv=4,
                            n_iter=8,
                            random_state=5)
```
