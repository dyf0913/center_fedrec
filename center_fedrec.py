import pandas as pd
import torch as pt 
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt
import random
                            
BATCH_SIZE=100

# 读取测试以及训练数据
cols=['user','item','rating']
all_=pd.read_csv('data/ratings_sample.csv',encoding='utf-8',names=cols)
train=pd.read_csv('data/ratings_train.csv',encoding='utf-8',names=cols)
test=pd.read_csv('data/ratings_test.csv',encoding='utf-8',names=cols)
print("all_ shape:",all_.shape)
print("train shape:",train.shape)
print("test shape:",test.shape)

#userNo的最大值
userNo=max(train['user'].max(),test['user'].max())
print("userNo:",userNo)
#movieNo的最大值
itemNo=max(train['item'].max(),test['item'].max())
print("itemNo:",itemNo)


#构建user-item的rating矩阵
rating_all=pt.zeros((userNo,itemNo))
rating_train=pt.zeros((userNo,itemNo))
rating_test=pt.zeros((userNo,itemNo))
for index,row in all_.iterrows():
    rating_all[int(row['user']-1)][int(row['item']-1)]=row['rating']
for index,row in train.iterrows():
    rating_train[int(row['user']-1)][int(row['item']-1)]=row['rating']
for index,row in test.iterrows():
    rating_test[int(row['user']-1)][int(row['item']-1)] = row['rating']
print(rating_all)
print(rating_train.shape)
print(rating_train)

# 用户平均（UA）
# 计算用户对所有打过分电影的平均值
def UserAvRating(rating_train):
    m,n=rating_train.shape
    rating_mean=pt.zeros((m,1))
    for i in range(m):
        idx=(rating_train[i,:]!=0)
        rating_mean[i]=pt.mean(rating_train[i,idx])
    tmp=rating_mean.numpy()
    tmp=np.nan_to_num(tmp)        #对值为NaN进行处理，改成数值0
    rating_mean=pt.tensor(tmp)
    return rating_mean

rating_mean=UserAvRating(rating_train)
print("rating_mean:",rating_mean)
print(rating_mean.shape)

# 随机采样
#在各用户未评分项目中部分随机采样，并用该用户的平均分(UA值）对其赋值
p=3   #采样参数
rating_none=np.nonzero((rating_train==0)) #存放rating_train矩阵中值为零的位置
rating_none=rating_none.numpy()
print(rating_none)

for i in range(userNo):
    list_s=[r for r in rating_none if r[0]==i]
    #从用户未评分项目中随机采样p*Iu(Iu为用户评过分item数)个,并将其赋值为对应用户的打分平均值
    change=random.sample(list_s,p*len(rating_train[i].nonzero()))
    #print(change)
    for j in range(len(change)):
        rating_train[change[j][0]][change[j][1]] = rating_mean[i]
        #print(rating_train[change[j][0]][change[j][1]])
print(rating_train)        

#训练集分批处理
loader1 = Data.DataLoader(
    dataset=rating_train,      # torch TensorDataset format
    batch_size=BATCH_SIZE,     # 最新批数据
    shuffle=False              # 是否随机打乱数据
)

loader2 = Data.DataLoader(
    dataset=rating_test,      # torch TensorDataset format
    batch_size=BATCH_SIZE,    # 最新批数据
    shuffle=False             # 是否随机打乱数据
)

class MF(pt.nn.Module):
    def __init__(self,userNo,itemNo,num_feature=20):
        super(MF, self).__init__()
        self.num_feature=num_feature     #num of laten features
        self.userNo=userNo               #user num
        self.itemNo=itemNo               #item num
        #self.bi=pt.nn.Parameter(pt.rand(self.itemNo,1))    #parameter
        #self.bu=pt.nn.Parameter(pt.rand(self.userNo,1))    #parameter
        self.U=pt.nn.Parameter(pt.rand(self.userNo,self.num_feature))    #parameter
        self.V=pt.nn.Parameter(pt.rand(self.num_feature,self.itemNo))    #parameter
    def get_U(self):
        return self.U
    def get_V(self):
        return self.V

    def mf_layer(self):
        #predicts =self.bu + self.bi.t() + pt.mm(self.U, self.V)#矩阵相乘
        predicts =pt.mm(self.U, self.V)#矩阵相乘
        return predicts

    def forward(self, train_set):
        output=self.mf_layer()
        return output

num_feature=20    #k
mf=MF(userNo,itemNo,num_feature)
print("parameters len:",len(list(mf.parameters())))
param_name=[]
params=[]
for name,param in mf.named_parameters():
    param_name.append(name)
    #print(name)
    params.append(param)


lr=0.01
_lambda=0.001
epoch=1000
loss_list=[]
optimizer=pt.optim.SGD(mf.parameters(),lr,momentum=0.9)
# 对数据集进行训练
for i in range(epoch):
    optimizer.zero_grad()
    output=mf(train)
    #print(output)
    loss_func=pt.nn.MSELoss()
    loss=loss_func(output,rating_train)*(userNo*itemNo)+_lambda*(pt.sum(pt.pow(params[0],2))+pt.sum(pt.pow(params[1],2)))
    loss=loss/(userNo*itemNo)
    #loss = loss_func(output, rating_train)
    loss.backward()
    optimizer.step()
    loss_list.append(loss.detach().numpy())
    print("train loss:",loss)

#评价指标rmse
def rmse(pred_rate,real_rate):
    #使用均方根误差作为评价指标
    loss_func=pt.nn.MSELoss()
    mse_loss=loss_func(pred_rate,real_rate)
    rmse_loss=pt.sqrt(mse_loss)
    return rmse_loss


#测试时测试的是原来评分矩阵为0的元素，通过模型为其预测一个评分
prediction=output[np.where(rating_train==0)]
#评分矩阵中元素为0的位置对应测试集中的评分
rating_test=rating_test[np.where(rating_train==0)]
rmse_loss=rmse(prediction,rating_test)
print("test loss:",rmse_loss)


plt.clf()
plt.plot(range(epoch),loss_list,label='Training data')
plt.title("The MovieLens Dataset Learning Curve")
plt.xlabel('Number of Epochs')
plt.ylabel('RMSE')
plt.legend()
plt.grid()
plt.savefig("./loss.jpg")