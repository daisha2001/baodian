
# coding: utf-8

# ## Python速查宝典
# ###  整理机器学习建模过程中常用的一些方法，方便日后工作更加高效。——by Sam

# # 一、导入数据

# In[1]:


# 导入常用库numpy和pandas
import numpy as np
import pandas as pd


# In[5]:


# 常用的两种方式来查看一些陌生的方法
# np.info(np.ndarray.dtype)
# help(pd.read_csv)


# ## 1.1 文本文件

# ### 纯文本文件

# In[ ]:


filename = 'xxx.text'
file = open(filename,mode='r')
text = file.read()
print(file.closed)
file.close()
print(text)
# 以只读方式读取文件，查看文件是否已经关闭，关闭文件
# open的mode方式还有很多，见 https://blog.csdn.net/pengyangyan/article/details/79966297


# In[ ]:


# 使用上下文管理器 with
with open('xxx.txt','r') as file:
    print(file.readline())
    print(file.readline())
    print(file.readline())
    # 读取一行


# ### 表格数据

# In[ ]:


# 单数据类型文件
filename = 'xxx.txt'
data = np.loadtxt(filename,
                 delimiter=',', # 用于分割各列值得字符
                 skiprows=2, # 用于跳过前2行
                 usecols=[0,2], # 读取并使用第1列和第3列
                 dtype=str) # 使用的数据类型


# In[ ]:


# 多数据类型文件
filename = 'xxx.txt'
data = np.genfromtxt(filename,
                    delimiter=',',
                    names=True, # 导入时查找列名
                    dtype=None)


# In[ ]:


# 使用pandas导入文本文件
filename = 'xxx.csv'
data = pd.read_csv(filename,
                  nrows=5, # 读取的行数
                  header=None, # 用哪行做列名，默认首行
                  sep='\t', # 用于分割各列的字符
                  comment='#', # 用于分割注释的字符
                  na_values=("") # 读取时，哪些值为NA/NaN)


# ## 1.2 Excel表

# In[ ]:


file = 'xxx.xlsx'
data = pd.ExcelFile(file)
df_sheet2 = data.parse('1960-1966',
                      skiprow=[0],
                      names=['Country',
                             'AAM:War(2002)'])


# In[ ]:


# 使用sheet_name属性访问表单名称
data.sheet_names


# ## 1.3 SAS 文件

# In[ ]:


from sas7bdat import SAS7BDAT
with SAS7BDAT('xxx,sas7bdat') as file:
    df_sas = file.to_data_frame()


# ## 1.4 探索数据

# ### Numpy数组

# In[ ]:


# 查看数据元素的数据类型
data_array.dtype
# 查看数组维度
data_array.shape
# 查看数据长度
len(data_array)


# ### Pandas数据框

# In[ ]:


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


# ### 探索字典

# In[ ]:


# 输出字典的键值
print(mat.keys())
# 输出字典的键值
for key in data.keys():
    print(key)
# 输出字典的值
pickled_data.values()
# 返回由元祖构成字典键值对列表
print(mat.items())


# ### 探索文件系统

# In[6]:


# 列出当前目录里的文件和文件夹
get_ipython().system('ls')


# In[ ]:


# 改变当前工作目录
get_ipython().magic('cd .')


# In[7]:


# 返回当前工作目录的路径
get_ipython().magic('pwd')


# In[ ]:


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


# # 二、numpy库

# In[2]:


import numpy as np


# ## 创建数组

# In[14]:


a = np.array([1,2,3])
a


# In[15]:


b = np.array([(1.5,2,3),(4,5,6)],dtype=float)
b


# In[16]:


c = np.array([[(1.3,2,3),(2.2,3.4,2)],[(1,2,3),(4,5,6)]],dtype=float)
c


# ## 初始化占位符

# In[ ]:


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


# ## 输入/输出

# In[ ]:


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


# ## 数据类型

# In[ ]:


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


# ## 数组信息

# In[ ]:


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


# ## 数组计算

# ### 算术计算

# In[ ]:


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


# ### 比较计算

# In[8]:


a = np.array([(1,2,3),(4,5,6)],dtype=float)
b = np.array([(0.5,2,3),(1,5,6)],dtype=float)


# In[4]:


a == b


# In[5]:


a < 2


# In[6]:


np.array_equal(a,b)


# ### 聚合函数

# In[14]:


a = np.array([(1,2,3),(4,5,6)],dtype=float)
a


# In[10]:


# 数组汇总
a.sum()


# In[11]:


# 数组最小值
a.min()


# In[12]:


# 数组最大值
print(a.max(axis=0))
print(a.max(axis=1))


# In[15]:


# 数组元素的累加值
print(a.cumsum(axis=0))
print(a.cumsum(axis=1))


# In[16]:


# 平均数
a.mean()


# In[22]:


# 标准差
np.std(a)


# ### 数组排序

# In[30]:


a = np.array([(1,14,3),(4,9,6)],dtype=float)
a


# In[31]:


a.sort()
a


# In[33]:


a.sort(axis=0)
a


# ## 子集、切片、索引

# In[38]:


a = np.array([(1.5,2,3),(4,5,6),(7.7,8,9)],dtype=float)
a


# In[39]:


a[0:2,1]


# In[40]:


a[:1]
# 等同 a[0:1,:]


# In[41]:


a[1:]
# 等同 a[1:,:]


# In[45]:


a[2,...]


# In[47]:


# 条件索引
a[a<5]


# ## 数组操作

# In[49]:


a = np.array([(1.5,2,3),(4,5,6),(7.7,8,9)],dtype=float)
a


# In[48]:


# 转置数组
i = np.transpose(a)
i.T


# In[52]:


# 改变数组形状
a.ravel()
a.reshape(1,9)


# # 三、Pandas库

# In[3]:


import pandas as pd


# In[58]:


# Series 序列
# 存储任意类型数据的一维数组
s = pd.Series([3,-5,7,4],index=['a','b','c','d'])
s


# In[114]:


# DataFrame - 数据框
data = {'Country':['Belgium','India','Brazil'],
       'Capital':['Brussels','New Delhi','Brasilia'],
       'Population':[11190846,1303171035,207847528]}
df = pd.DataFrame(data,
                  columns=['Country','Capital','Population'])
df


# In[79]:


# 按位置
print(df.iloc[[0],[0]])
print('\n')
# 按标签
print(df.loc[[0],['Country']])
print('\n')
# 布尔索引
print(s[~(s>1)])


# In[82]:


# 删除数据
# 按索引删除序列的值
s1 = s.drop(['a','c'])
print(s1)
print('\n')

# 按列名删除数据框的列
df1 = df.drop('Country',axis=1)
print(df1)


# In[ ]:


# 排序和排名
# 按索引排序
df.sort_index()
# 按某列的值排序
df.sort_values(by='Country')
# 数据框排名，默认升序
df.rank()


# In[83]:


df.rank()


# ## 输入/输出

# In[ ]:


# 读取/写入CSV
pd.read_csv('file.csv',header=None,nrows=5)
df.to_csv('MyDataFrame.csv')


# In[ ]:


# 读取/写入Excel
pd.read_excel('file.xlsx')
pd.to_excel('dir/myDataFrame.xlsx',sheet_name='Sheet1')


# In[ ]:


# 读取内含多个表的Excel
xlsx = pd.ExcelFile('file.xls')
df = pd.read_excel(xlsx,'Sheet1')


# ## 查询序列与数据框的信息

# ### 基础信息查询

# In[ ]:


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


# ### 应用函数

# In[84]:


df


# In[86]:


f = lambda x:x*2 # 匿名函数lambda
df1 = df.apply(f) # 应用函数
df1


# In[87]:


df2 = df.applymap(f) # 对每个单元格应用函数
df2


# ## 数据重塑

# ### 数据透视

# In[93]:


data2 = {'Date':['2016-03-01','2016-03-02','2016-03-01','2016-03-03','2016-03-02','2016-03-03'],
        'Type':['a','b','c','a','a','c'],
        'Value':[11.432,13.031,20.784,99.906,1.303,20.784]}
df2 = pd.DataFrame(data2,
                   columns=['Date','Type','Value'])
df2


# In[94]:


# 行变列,and Index could not contains duplicate entries
df3 = df2.pivot(index='Date',
                columns='Type',
                values='Value')
df3


# In[105]:


# 数据透视
df4 = pd.pivot_table(df2,
                     values='Value',
                     index='Date',
                     columns='Type',
                     aggfunc='count') # np.sum np.mean
df4


# ### 融合

# In[107]:


df5 = pd.melt(df2,
              id_vars=['Date'],
              value_vars=['Type','Value'],
              value_name='Observations')
df5


# ### 高级索引

# In[108]:


df3


# In[109]:


# 选择任一值大于1的列
dft = df3.loc[:,(df3>1).any()]
dft


# In[111]:


# 选择所有值都大于1的列
dft = df3.loc[:,(df3>1).all()]
dft


# In[112]:


# 选择含Nan值的列
dft = df3.loc[:,df3.isnull().any()]
dft


# In[113]:


# 选择不含Nan值的列
dft = df3.loc[:,df3.notnull().all()]
dft


# ### 索引

# In[117]:


# 设置索引
dft = df.set_index('Country')
dft


# In[118]:


# 取消索引
df4 = dft.reset_index()
df4


# In[119]:


# 重命名DataFrame列名
df5 = df.rename(index=str,
                columns={'Country':'cntry',
                         'Capital':'cptl',
                         'Population':'ppltn'})
df5


# In[126]:


# 重置索引
s2 = s.reindex(['a','c','d','e','b'])
s2


# In[121]:


# 前向填充
df6 = df.reindex(range(8),method='ffill')
df6


# In[125]:


# 后向填充
df6 = df.reindex(range(4),method='bfill')
df6


# In[129]:


# 返回唯一值
s.unique()


# In[134]:


# 查找重复值
print(df2)
print('\n')
print(df2.duplicated('Type',keep=False))
# keep=‘frist’：除了第一次出现外，其余相同的被标记为重复
# keep='last'：除了最后一次出现外，其余相同的被标记为重复


# In[135]:


# 去除重复值,保留最后一条记录
df7 = df2.drop_duplicates('Type',keep='last')
df7


# In[137]:


# 查找重复索引
df.index.duplicated()


# ### 数据分组

# In[4]:


data2 = {'Date':['2016-03-01','2016-03-02','2016-03-01','2016-03-03','2016-03-02','2016-03-03'],
        'Type':['a','b','c','a','a','c'],
        'Value':[11.432,13.031,20.784,99.906,1.303,20.784]}
df2 = pd.DataFrame(data2,
                   columns=['Date','Type','Value'])
df2


# In[5]:


# 聚合
a = df2.groupby(by=['Date','Type']).mean()
a


# In[6]:


df4 = pd.pivot_table(df2,
                     values='Value',
                     index='Date',
                     columns='Type',
                     aggfunc='count') # np.sum np.mean
df4


# In[11]:


customsum = lambda x:(x+x%2)
a = df4.groupby(level=0).transform(customsum)
a


# ### 缺失值

# In[13]:


# 去除缺失值NAN
df.dropna() 


# In[15]:


# 用预设值填充缺失值NaN
a = df4.fillna(df4.sum())
a


# In[19]:


# 用一个值替换另一个值
print("*"*25)
print(df2)
a= df2.replace("a","f")
print("*"*25)
print(a)


# In[ ]:


data2 = {'Date':['2016-03-01','2016-03-02','2016-03-01','2016-03-03','2016-03-02','2016-03-03'],
        'Type':['a','b','c','a','a','c'],
        'Value':[11.432,13.031,20.784,99.906,1.303,20.784]}
df2 = pd.DataFrame(data2,
                   columns=['Date','Type','Value'])
df2


# ### 合并数据

# In[22]:


data1 = {'X1':['a','b','c'],
        'X2':[11.432,1.303,99.906]}
data1 = pd.DataFrame(data1,columns=['X1','X2'])
data1


# In[23]:


data2 = {'X1':['a','b','d'],
        'X2':[20.784,'NaN',20.784]}
data2 = pd.DataFrame(data2,columns=['X1','X2'])
data2


# In[30]:


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


# ### 拼接-concatenate 

# In[ ]:


# 纵向
s.append(s2)
# 横向
pd.concat([s,s2],axis=1,keys=['One','Two'])
pd.concat([data1,data2],axis=1,join='inner')


# ### 日期

# In[32]:


df2.info()


# In[33]:


df2['Date']=pd.to_datetime(df2['Date'])
df2.info()


# # 四、Scikit-learn

# ### Scikit-learn是开源的Python库，通过统一的界面实现机器学习、预处理、交叉验证及可视化算法。
# 

# ## 加载数据

# ### Scikit-learn处理的数据是储存在Numpy数据或者SciPy稀疏矩阵的数字，还支持Pandas数据框等可转换为数据数组的其他数据类型。

# In[53]:


import numpy as np
X = np.random.random((10,5))
y = np.array(['M','M','F','F','M','F','M','M','F','F'])
X


# In[54]:


X[X<0.7]=0
X


# ### 划分训练验证

# In[55]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)


# ## 数据预处理

# ### 标准化

# In[57]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler().fit(X_train)
standardized_X = scaler.transform(X_train)
standardized_X_test = scaler.transform(X_test)


# In[58]:


scaler


# ### 归一化

# In[59]:


from sklearn.preprocessing import Normalizer
scaler = Normalizer().fit(X_train)
normalized_X = scaler.transform(X_train)
normalized_X_test = scaler.transform(X_test)


# In[60]:


scaler


# ### 二值化

# In[61]:


from sklearn.preprocessing import Binarizer
binarizer = Binarizer(threshold=0.0).fit(X)
binary_X = binarizer.transform(X)


# In[62]:


X


# In[63]:


binary_X


# ### 编码分类特征

# In[64]:


from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
y = enc.fit_transform(y)
y


# ### 输入缺失值

# In[65]:


X_train


# In[66]:


from sklearn.preprocessing import Imputer
imp = Imputer(missing_values=0,strategy='mean',axis=0)
imp.fit_transform(X_train)


# ### 生成多项式特征

# In[68]:


from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(5)
poly.fit_transform(X)


# In[69]:


X


# ## 创建模型

# ### 有监督学习评估器

# In[ ]:


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


# ### 无监督学习评估器

# In[ ]:


# 主成分分析（PCA）
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)

# K Means
from sklearn.cluster import KMeans
k_means = KMeans(n_clusters=3,random_state=0)


# ## 模型拟合

# In[ ]:


# 有监督学习
lr.fit(X,y)
knn.fit(X,y)
svc.fit(X,y)

# 无监督学习
k_means.fit(X_train)
pca_model = pca.fit_transform(X_train)


# ## 预测

# In[ ]:


# 有监督评估器
y_pred = svc.predict(np.random.random((2,5)))
y_pred = lr.predict(X_test)
y_pred = knn.predict_proba(X_test)

# 无监督评估器
y_pred = k_means.predict(X_test)


# ## 评估模型性能

# ### 分类指标

# In[ ]:


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


# ### 回归指标

# In[72]:


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


# ### 交叉验证

# In[ ]:


from sklearn.cross_validation import cross_val_score
print(cross_val_score(knn,X_train,y_train,cv=4))
print(cross_val_score(lr,X,y,cv=2))


# ### 模型调参

# In[ ]:


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

