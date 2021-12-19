"""
对数据进行预处理，按照8：2分为训练集与测试集
"""
import pandas as pd
import csv
import random
import os
import numpy as np

def itemConvert(items): #将itemID映射成1~n的连续值
    itemID = items[:, 1]
    setlist = list(set(itemID))
    itemNum = len(setlist)
    setlist=[int(x) for x in setlist] #将字符串类型转换成整型
    sortitemID = sorted(setlist)
    sortitemID=[str(x) for x in sortitemID] 
    tempMap = {}
    for i in range(0, itemNum):
        tempMap[sortitemID[i]] = i+1
    for i in range(0, len(itemID)):
        itemID[i] = tempMap[itemID[i]]
    items[:, 1] = itemID
    
    return items


#df = pd.read_csv('data/ratings.csv', encoding='utf-8')
#print((df.values[:, :2]))
df = pd.read_csv (r'data/ratings.csv', sep=',', engine='c', header=None).to_numpy ()[1:,:3]
#print(max((df[:, 1]).astype(np.int64)))
df[:, :2]=itemConvert(df[:, :2]) 
#转换成dataframe类型
df=pd.DataFrame(df) 
df_sample=df
#print(max((df[: , 1]).astype(np.int64)))

#将映射后的数据存储到csv文件中
print("sample shape:",df_sample.shape)
df_sample.to_csv('data/ratings_sample_tmp.csv',index=False)
#去掉第一行
origin_f = open('data/ratings_sample_tmp.csv','rt',encoding='utf-8',errors="ignore")
new_f = open('data/ratings_sample.csv','wt+',encoding='utf-8',errors="ignore",newline="")     #必须加上newline=""否则会多出空白行
reader = csv.reader(origin_f)
writer = csv.writer(new_f)
for i,row in enumerate(reader):
    if i>0:
        writer.writerow(row)
origin_f.close()
new_f.close()
os.remove('data/ratings_sample_tmp.csv')

#将数据按照8:2的比例进行划分得到训练数据集与测试数据集
df = pd.read_csv('data/ratings_sample.csv', encoding='utf-8')
df = df.sample(frac=1.0)  # 全部打乱
cut_idx = int(round(0.2 * df.shape[0]))
df_test, df_train = df.iloc[:cut_idx], df.iloc[cut_idx:]
# 打印数据集中的数据记录数
print("df shape:",df.shape,"test shape:",df_test.shape,"train shape:",df_train.shape)
#print(df_train)

# 将数据记录存储到csv文件中
# 存储训练数据集
df_train.to_csv('data/ratings_train_tmp.csv',index=False)
# 使用pandas读取得到的数据多了一行，在存储时也会将这一行存储起来，所以应该删除这一行
origin_f = open('data/ratings_train_tmp.csv','rt',encoding='utf-8',errors="ignore")
new_f = open('data/ratings_train.csv','wt+',encoding='utf-8',errors="ignore",newline="")
reader = csv.reader(origin_f)
writer = csv.writer(new_f)
for i,row in enumerate(reader):
    if i>0:
        writer.writerow(row)
origin_f.close()
new_f.close()
os.remove('data/ratings_train_tmp.csv')

# 存储测试数据集
df_test=pd.DataFrame(df_test)
df_test.to_csv('data/ratings_test_tmp.csv',index=False)
origin_f = open('data/ratings_test_tmp.csv','rt',encoding='utf-8',errors="ignore")
new_f = open('data/ratings_test.csv','wt+',encoding='utf-8',errors="ignore",newline="")
reader = csv.reader(origin_f)
writer = csv.writer(new_f)
for i,row in enumerate(reader):
    if i>0:
        writer.writerow(row)
origin_f.close()
new_f.close()
os.remove('data/ratings_test_tmp.csv')