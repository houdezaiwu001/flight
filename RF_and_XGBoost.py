import random
import os
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.externals import joblib
import xgboost as xgb
import pandas as pd
import csv
import time
import operator
import numpy as np
from sklearn.cross_validation import train_test_split
import time
print "======= start ======="
t0 = time.time()


file1 = csv.reader(open("data/date/standard_25_predict.csv",'rb'))
predict_data = []
tag = []
hour = []
for row in file1:
    predict_data.append(row[2:])
    tag.append(row[0])
    hour.append(row[3])

file1 = csv.reader(open("data/date/standard_25_train.csv",'rb'))
train_data = []
train_target = []
for row in file1:
    train_data.append(row[2:])
    train_target.append(row[1])


X_train, X_test, y_train, y_test = train_test_split(train_data,train_target, test_size=0.2, random_state=0)
print np.shape(train_data)
print "Start training Random forest..."
rfClf = ExtraTreesRegressor(n_estimators=500,n_jobs=-1,min_samples_split=1,criterion="mse")
rfClf.fit(X_train,y_train)
# val_pred_label = rfClf.predict(train_data)
# val_pred_label = rfClf.predict(predict_data)
val_pred_label = rfClf.predict(X_test)


# joblib.dump(rfClf, 'model/rf_14_12h_15h_add_checkInNumber.model')
# val_pred_label = rfClf.predict(val_data)
# val_pred_label = rfClf.predict(predict_data)


# rfClf = joblib.load('model/rf_14_12h_15h_add_checkInNumber.model')
# val_pred_label = rfClf.predict(train_data)

# import cPickle
# ppppppp = cPickle.load(open("new_data/train/"+"xgb_train"+".pkl",'rb'))


score = 0
for j in range(len(val_pred_label)):
    # score += (float(val_pred_label[j]) - float(train_target[j])) * (float(val_pred_label[j]) - float(train_target[j]))
    score += (float(val_pred_label[j]) - float(y_test[j])) * (float(val_pred_label[j]) - float(y_test[j]))
    # score += (float(val_pred_label[j]) - float(predict_test[j])) * (float(val_pred_label[j]) - float(predict_test[j]))
    # combine = 1 * float(val_pred_label[j]) + 0 * float(ppppppp[j])
    # score += (combine - float(train_target[j])) * (combine - float(train_target[j]))
print score

# print len(val_pred_label)
# f = open("new_data/result/standard_predict_rf_20_v2.csv", 'wb')
# write = csv.writer(f)
# write.writerow(['passengerCount', 'WIFIAPTag', 'slice10min'])
# for i in range(len(val_pred_label)):
#     passenger = val_pred_label[i]
#     if(float(passenger) < 0 ):
#         passenger = 0
#     words = [passenger, tag[i], "2016-09-25-" +str(int(hour[i])/6) + "-" + str(int(hour[i])%6)]
#     write.writerow(words)



dataset = pd.read_csv("data/date/standard_25_train.csv",header=None)
predict_dataset = pd.read_csv("data/date/standard_25_predict.csv",header=None)


print dataset.shape
trains = dataset.iloc[:, 2:].values
labels = dataset.iloc[:, 1].values
need_to_predict = predict_dataset.iloc[:, 2:].values
print np.shape(trains)
x_train, x_test, y_train, y_test = train_test_split(trains, labels, test_size=0.2)
params = {'booster': 'gblinear',
          'objective': 'reg:linear',
          #'num_class': 10,
          'gamma': 0.1,
          'max_depth': 9,
          #'lambda': 450,
          'subsample': 0.5,
          'colsample_bytree': 0.8,
          #'min_child_weight': 12,
          'silent': 1,
          'eta': 0.1,
          #'nthread': 4,
          'seed': 710
          }
# params = {'booster':'gblinear','max_depth':30, 'eta': 0.01, 'silent':1, 'objective':'reg:linear','seed':1000,'lambda':0.01,'subsample':0.8,'min_child_weight':5,'col_sample_bytree':0.2}

plst = list(params.items())
num_rounds = 1000
xgtrain = xgb.DMatrix(x_train, label=y_train)
xgval = xgb.DMatrix(x_test, label=y_test)
# xgpred = xgb.DMatrix(trains)
# xgpred = xgb.DMatrix(need_to_predict)
xgpred = xgb.DMatrix(x_test)


watchlist = [(xgtrain, 'train'), (xgval, 'val')]
print "Start training XGBoost..."
model = xgb.train(plst, xgtrain, num_rounds, watchlist)
preds = model.predict(xgpred)


score = 0
for i in range(0,len(preds),1):
    if(preds[i]<0):
        preds[i] = 0
    # score += (preds[i]-labels[i]) * (preds[i]-labels[i])
    score += (preds[i] - y_test[i]) * (preds[i] - y_test[i])
print score


# f = open("new_data/result/xgb_21_gblinear.csv", 'wb')
# write = csv.writer(f)
# write.writerow(['passengerCount', 'WIFIAPTag', 'slice10min'])
# for i in range(len(preds)):
#     passenger = preds[i]
#     if(float(passenger) < 0 ):
#         passenger = 0
#     words = [passenger, tag[i], "2016-09-25-" +str(int(hour[i])/6) + "-" + str(int(hour[i])%6)]
#     write.writerow(words)

# print len(val_pred_label),len(preds),len(train_target),len(labels)
#
f = open("new_data/result/test001.csv", 'wb')
write = csv.writer(f)
score = 0
for j in range(len(val_pred_label)):
    predict_value = 0.5*val_pred_label[j] + 0.5*preds[j]
    score += (predict_value - float(train_target[j])) * (predict_value - float(train_target[j]))
    words = [predict_value,float(train_target[j]),val_pred_label[j],preds[j]]
    write.writerow(words)
print score


f = open("new_data/result/test002.csv", 'wb')
write = csv.writer(f)
score = 0
for i in range(0,len(preds),1):
    if(preds[i]<0):
        preds[i] = 0
    predict = 0.5*val_pred_label[i] + 0.5*preds[i]
    score += (predict-labels[i]) * (predict-labels[i])
    words = [predict, float(train_target[i]), val_pred_label[i], labels[i]]
    write.writerow(words)
print score

t1 = time.time()
print "======= end  ======="
print "It take %f s to process" % (t1 - t0)