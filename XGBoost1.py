#-*-coding:utf-8-*-
import numpy as np
import pandas as pd
import csv
import xgboost as xgb
from sklearn.cross_validation import train_test_split
# read in data
# dtrain = xgb.DMatrix('demo/data/agaricus.txt.train')
# dtest = xgb.DMatrix('demo/data/agaricus.txt.test')
# # specify parameters via map
# param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
# num_round = 2
# bst = xgb.train(param, dtrain, num_round)
# # make prediction
# preds = bst.predict(dtest)

import numpy as np
import scipy.sparse
import pickle
import xgboost as xgb
from sklearn.cross_validation import train_test_split

### simple example
# load file from text file, also binary buffer generated by xgboost
# dtrain = xgb.DMatrix('../data/agaricus.txt.train')
# dtest = xgb.DMatrix('../data/agaricus.txt.test')


# file1 = csv.reader(open("data/date/train.csv",'rb'))
# file1.next()
# train_data = []
# train_target = []
# for row in file1:
#     train_data.append(row[3:])
#     train_target.append(row[2])

# train = pd.read_csv("data/date/train2.csv")
# train_xy,val = train_test_split(train, test_size = 0.2,random_state=1)
# y = train_xy.label
# X = train_xy.drop(['label'],axis=1)
# val_y = val.label
# val_X = val.drop(['label'],axis=1)



tests = pd.read_csv("data/date/predict_xgboost.csv")
xgb_test = xgb.DMatrix(tests)


# X_train, X_test, y_train, y_test = train_test_split(train_data,train_target, test_size=0.2, random_state=0)





# X_train = np.array(X)
# X_test = np.array(X_test)
# y_train = np.array(y_train)
# y_test = np.array(y_test)
# print y_test
# for row in X_test:
#     for i in range(30):
#         row[i] = float(row[i])


# xgb_val = xgb.DMatrix(val_X,val_y)
# xgb_train = xgb.DMatrix(X,y)



# xgb_test = xgb.DMatrix(tests)

# params={
# 'booster':'gblinear',
# 'objective': 'multi:reg:linear', #多分类的问题
# 'gamma':0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
# 'max_depth':12, # 构建树的深度，越大越容易过拟合
# 'lambda':2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
# 'subsample':0.7, # 随机采样训练样本
# 'colsample_bytree':0.7, # 生成树时进行的列采样
# 'min_child_weight':3,
# # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
# #，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
# #这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
# 'silent':0 ,#设置成1则没有运行信息输出，最好是设置为0.
# 'eta': 0.007, # 如同学习率
# 'seed':1000,
# 'nthread':-1,# cpu 线程数
# #'eval_metric': 'auc'
# }



# # specify parameters via map, definition are same as c++ version
# param = {'max_depth':10, 'eta':0.01, 'silent':1, 'objective':'reg:linear','gamma':0.1,'min_child_weight':1}
#
# # specify validations set to watch performance
# watchlist  = [(xgb_val,'eval'), (xgb_train,'train')]
# num_round = 5000
# bst = xgb.train(param, xgb_train, num_round, watchlist)
#
# # this is prediction
# preds = bst.predict(xgb_val)
# labels = xgb_val.get_label()
# # print ('error=%f' % ( sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=labels[i]) /float(len(preds))))

# score = 0
# for j in range(len(preds)):
#     score += (float(preds[j])- float(labels[j])) * (float(preds[j]) - float(labels[j]))
# print score



# bst.save_model('model/xgb/0001.model')


model = xgb.Booster()
# dump model
model.load_model('model/xgb/0001.model')
preds = model.predict(xgb_test)

# # dump model with feature map
# bst.dump_model('dump.nice.txt','../data/featmap.txt')

f = open("new_data/result/airport_gz_passenger_predict6.csv", 'wb')
write = csv.writer(f)
write.writerow(['passengerCount', 'WIFIAPTag', 'slice10min'])

file1 = csv.reader(open("new_data/countWifiByEachAP/20160913.csv",'rb'))
text1 = []
for row in file1:
    if(int(row[1])>=150 and int(row[1]) <= 175):
        text1.append(row)

# print len(text1)
# print len(val_pred_label)
for i in range(len(text1)):
    passenger = preds[i]
    words = [passenger, text1[i][0], "2016-09-14-" + text1[i][1][0:2] + "-" + text1[i][1][2]]
    write.writerow(words)











# # save dmatrix into binary buffer
# dtest.save_binary('dtest.buffer')
# # save model
# bst.save_model('xgb.model')
# load model and data in
# bst2 = xgb.Booster(model_file='xgb.model')
# dtest2 = xgb.DMatrix('dtest.buffer')
# preds2 = bst2.predict(dtest2)
# # assert they are the same
# assert np.sum(np.abs(preds2-preds)) == 0

# # alternatively, you can pickle the booster
# pks = pickle.dumps(bst2)
# # load model and data in
# bst3 = pickle.loads(pks)
# preds3 = bst3.predict(dtest2)
# # assert they are the same
# assert np.sum(np.abs(preds3-preds)) == 0

###
# # build dmatrix from scipy.sparse
# print ('start running example of build DMatrix from scipy.sparse CSR Matrix')
# labels = []
# row = []; col = []; dat = []
# i = 0
# for l in open('../data/agaricus.txt.train'):
#     arr = l.split()
#     labels.append(int(arr[0]))
#     for it in arr[1:]:
#         k,v = it.split(':')
#         row.append(i); col.append(int(k)); dat.append(float(v))
#     i += 1
# csr = scipy.sparse.csr_matrix((dat, (row,col)))
# dtrain = xgb.DMatrix(csr, label = labels)
# watchlist  = [(dtest,'eval'), (dtrain,'train')]
# bst = xgb.train(param, dtrain, num_round, watchlist)

# print ('start running example of build DMatrix from scipy.sparse CSC Matrix')
# # we can also construct from csc matrix
# csc = scipy.sparse.csc_matrix((dat, (row,col)))
# dtrain = xgb.DMatrix(csc, label=labels)
# watchlist  = [(dtest,'eval'), (dtrain,'train')]
# bst = xgb.train(param, dtrain, num_round, watchlist)
#
# print ('start running example of build DMatrix from numpy array')
# # NOTE: npymat is numpy array, we will convert it into scipy.sparse.csr_matrix in internal implementation
# # then convert to DMatrix
# npymat = csr.todense()
# dtrain = xgb.DMatrix(npymat, label = labels)
# watchlist  = [(dtest,'eval'), (dtrain,'train')]
# bst = xgb.train(param, dtrain, num_round, watchlist)