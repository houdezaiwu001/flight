import csv
import operator
import numpy as np
import random
from sklearn.ensemble import ExtraTreesRegressor
# from xgboost.sklearn import XGBClassifier
# import xgboost as xgb
table = {}
i = 0
file1 = csv.reader(open('new_data/countWifiByEachAP/20160912.csv','rb'))
for row in file1:
    if not(table.has_key(row[0])):
        table[row[0]] = i
        i += 1

# for key,value in table.items():
#     print key,value
# print len(table)


file55 = csv.reader(open("new_data/airport_gz_gates.csv",'rb'))
header = file55.next()
ap_flight = []
text = []
e1 = []
e2 = []
e3 = []
w1 = []
w2 = []
w3 = []
e1_flight = []
e2_flight = []
e3_flight = []
w1_flight = []
w2_flight = []
w3_flight = []


for row in file55:
    if(row[1] == "E1"):
        e1.append(row[0])
    elif(row[1] == "E2"):
        e2.append(row[0])
    elif (row[1] == "E3"):
        e3.append(row[0])
    elif (row[1] == "W1"):
        w1.append(row[0])
    elif (row[1] == "W2"):
        w2.append(row[0])
    elif (row[1] == "W3"):
        w3.append(row[0])


file55 = csv.reader(open("new_data/airport_gz_flights_chusai_1stround.csv",'rb'))
header = file55.next()
for row in file55:
    if((row[-1]) in e1):
        e1_flight.append(row[0])
    elif(row[-1] in e2):
        e2_flight.append(row[0])
    elif (row[-1] in e3):
        e3_flight.append(row[0])
    elif (row[-1] in w1):
        w1_flight.append(row[0])
    elif (row[-1] in w2):
        w2_flight.append(row[0])
    elif (row[-1] in w3):
        w3_flight.append(row[0])



def reWriteData(name):
    file55 = csv.reader(open("new_data/fetchTicket/"+name+".csv",'rb'))
    text = []
    for row in file55:
        if(row[0] in e1_flight):
            row[0] = "E1"
        elif(row[0] in e2_flight):
            row[0] = "E2"
        elif (row[0] in e3_flight):
            row[0] = "E3"
        elif (row[0] in w1_flight):
            row[0] = "W1"
        elif (row[0] in w2_flight):
            row[0] = "W2"
        elif (row[0] in w3_flight):
            row[0] = "W3"
        text.append(row)
    with open('new_data/fetchTicket/new/' + name+'.csv', 'wb') as myFile:
        myWriter = csv.writer(myFile, dialect='excel')
        dictionary = {}
        data = []
        for row in text:
            sample = (row[0], row[1])
            if (dictionary.has_key(sample)):
                dictionary[sample] += 1
            else:
                dictionary[sample] = 1
        for k, v in dictionary.iteritems():
            text = []
            text.append(k[0])
            text.append(k[1])
            text.append(v)
            data.append(text)
        data = sorted(data, key=operator.itemgetter(1, 0))
        myWriter.writerows(data)

def reWriteData2(name):
    file55 = csv.reader(open("new_data/passSecurity/"+name+".csv",'rb'))
    text = []
    for row in file55:
        if(row[0] in e1_flight):
            row[0] = "E1"
        elif(row[0] in e2_flight):
            row[0] = "E2"
        elif (row[0] in e3_flight):
            row[0] = "E3"
        elif (row[0] in w1_flight):
            row[0] = "W1"
        elif (row[0] in w2_flight):
            row[0] = "W2"
        elif (row[0] in w3_flight):
            row[0] = "W3"
        text.append(row)
    with open('new_data/passSecurity/new/' + name+'.csv', 'wb') as myFile:
        myWriter = csv.writer(myFile, dialect='excel')
        dictionary = {}
        data = []
        for row in text:
            sample = (row[0], row[1])
            if (dictionary.has_key(sample)):
                dictionary[sample] += 1
            else:
                dictionary[sample] = 1
        for k, v in dictionary.iteritems():
            text = []
            text.append(k[0])
            text.append(k[1])
            text.append(v)
            data.append(text)
        data = sorted(data, key=operator.itemgetter(1, 0))
        myWriter.writerows(data)

# reWriteData("20160910")
# reWriteData("20160911")
# reWriteData("20160912")
# reWriteData("20160913")
# reWriteData("20160914")
# reWriteData2("20160910")
# reWriteData2("20160911")
# reWriteData2("20160912")
# reWriteData2("20160913")
# reWriteData2("20160914")

def addFeature(name):
    file55 = csv.reader(open("new_data/countWifiByEachAP/"+name+".csv",'rb'))
    text = []
    for row in file55:
        file66 = csv.reader(open("new_data/passSecurity/new/"+name+".csv", 'rb'))
        flag = 0
        for row2 in file66:
            if(row[0][0:2] == row2[0] and row[1] == row2[1]):
                row.append(row2[2])
                flag = 1
                break
        if(flag == 0):
            row.append(0)
        text.append(row)
    with open("new_data/countWifiByEachAP/new/" + name + '.csv', 'wb') as myFile:
        myWriter = csv.writer(myFile, dialect='excel')
        myWriter.writerows(text)

def addFeature2(name):
    file55 = csv.reader(open("new_data/countWifiByEachAP/new/" + name + ".csv", 'rb'))
    dictionary = {}
    text = []
    i = 0
    for row in file55:
        sample = (row[0][0:5],row[1])
        if(dictionary.has_key(sample)):
            row.append(dictionary[sample])
        else:
            dictionary[sample] = i
            i += 1
            row.append(dictionary[sample])
        text.append(row)
    with open("new_data/countWifiByEachAP/new/" + name + '.csv', 'wb') as myFile:
        myWriter = csv.writer(myFile, dialect='excel')
        myWriter.writerows(text)

def addFeature3(name):
    file55 = csv.reader(open("new_data/countWifiByEachAP/new/" + name + ".csv", 'rb'))
    dictionary1 = {}
    dictionary2 = {}
    text = []
    i = 0
    j = 0
    for row in file55:
        sample1 = (row[0][0:2],row[1])
        sample2 = row[0][3:5]
        if not(dictionary1.has_key(sample1)):
            dictionary2.clear()
            j = 0
            dictionary1[sample1] = i
            i += 1
        if(dictionary2.has_key(sample2)):
            row[-1] = dictionary2[sample2]
        else:
            dictionary2[sample2] = j
            j += 1
            row[-1] = dictionary2[sample2]

        text.append(row)
    with open("new_data/countWifiByEachAP/new/" + name + '.csv', 'wb') as myFile:
        myWriter = csv.writer(myFile, dialect='excel')
        myWriter.writerows(text)


def writedata(name):
    file55 = csv.reader(open("new_data/countWifiByEachAP/new/" + name + ".csv", 'rb'))
    text = []
    for row in file55:
        text.append(row[0:5])
    with open("new_data/countWifiByEachAP/new/" + name + '.csv', 'wb') as myFile:
        myWriter = csv.writer(myFile, dialect='excel')
        myWriter.writerows(text)
# writedata("20160910")
# writedata("20160911")
# writedata("20160912")
# writedata("20160913")
# writedata("20160914")
# addFeature("20160910")
# addFeature("20160911")
# addFeature("20160912")
# addFeature("20160913")
# addFeature("20160914")
# addFeature2("20160910")
# addFeature2("20160911")
# addFeature2("20160912")
# addFeature2("20160913")
# addFeature2("20160914")
# addFeature3("20160910")
# addFeature3("20160911")
# addFeature3("20160912")
# addFeature3("20160913")
# addFeature3("20160914")
# file77 = csv.reader(open("new_data/countWifiByEachAP/new/20160911.csv",'rb'))
# for row in file77:
#     dictionary = {}
#     if(dictionary.has_key(row[0][0:2])):
#         flag =1
#     else:
#         dictionary[row[0][0:2]]


# m = [];
# m.append([]);
# m[0].append([]);
# m[0].append([]);
# m[0][0].append(2);
# m[0][0].append(1);
# m[0][1].append('m');
# m[0][1].append('n');
# m[0][1].append('s');
# for i in range(len(m)):
#    for j in range(len(m[i])):
#       for k in range(len(m[i][j])):
#          print(m[i][j][k]);
# print m
file1 = csv.reader(open("new_data/countWifiByEachAP/new/20160912.csv",'rb'))
file2 = csv.reader(open("new_data/countWifiByEachAP/new/20160913.csv",'rb'))
file3 = csv.reader(open("new_data/countWifiByEachAP/new/20160911.csv",'rb'))
file4 = csv.reader(open("new_data/countWifiByEachAP/new/20160914.csv",'rb'))
file5 = csv.reader(open("new_data/countWifiByEachAP/new/20160910.csv",'rb'))

def loadTrainSet():
    traindata = []
    trainlabel = []

    for row in file1:
        text = []
        text.append(int(row[1]))
        text.append(table.get(row[0]))
        text.append(int(row[-2]))
        text.append(float(row[-1]))
        traindata.append(text)
        trainlabel.append(float(row[-3])/10)

    for row in file2:
        text = []
        text.append(int(row[1]))
        text.append(table.get(row[0]))
        text.append(int(row[-2]))
        text.append(float(row[-1]))
        traindata.append(text)
        trainlabel.append(float(row[-3]) / 10)

    for row in file3:
        text = []
        text.append(int(row[1]))
        text.append(table.get(row[0]))
        text.append(int(row[-2]))
        text.append(float(row[-1]))
        traindata.append(text)
        trainlabel.append(float(row[-3]) / 10)


    for row in file4:
        if (int(row[1]) <= 115):
            text = []
            text.append(int(row[1]))
            text.append(table.get(row[0]))
            text.append(int(row[-2]))
            text.append(float(row[-1]))
            traindata.append(text)
            trainlabel.append(float(row[-3]) / 10)

    for row in file5:
        text = []
        text.append(int(row[1]))
        text.append(table.get(row[0]))
        text.append(int(row[-2]))
        text.append(float(row[-1]))
        traindata.append(text)
        trainlabel.append(float(row[-3]) / 10)

    # traindata = np.array(traindata, dtype="float")
    # trainlabel = np.array(trainlabel, dtype="float")


    # Standardize(zero-mean,nomalization)
    # mean = traindata.mean(axis=0)
    # std = traindata.std(axis=0)
    # traindata = (traindata - mean) / std
    #
    # # shuffle the data
    # randomIndex = [i for i in xrange(len(trainlabel))]
    # random.shuffle(randomIndex)
    # traindata = traindata[randomIndex]
    # trainlabel = trainlabel[randomIndex]
    return traindata, trainlabel

def loadTestSet():
    file14 = csv.reader(open("new_data/countWifiByEachAP/new/20160914.csv", 'rb'))
    testdata = []
    for row in file14:
        # print row
        if (int(row[1]) >= 120):
            text = []
            text.append(row[1])
            text.append(table.get(row[0]))
            text.append(int(row[-2]))
            text.append(float(row[-1]))
            testdata.append(text)

    # testdata = np.array(testdata,dtype="float")
    # mean = testdata.mean(axis=0)
    # std = testdata.std(axis=0)
    # testdata = (testdata - mean)/std
    return testdata



# import pandas as pd
# import numpy as np
# import xgboost as xgb
# from xgboost.sklearn import XGBClassifier
# from sklearn import cross_validation, metrics   #Additional     scklearn functions
# from sklearn.grid_search import GridSearchCV   #Perforing grid search
#
# import matplotlib.pylab as plt
# from matplotlib.pylab import rcParams
# rcParams['figure.figsize'] = 12, 4
#
# train = pd.read_csv('train_modified.csv')
# target = 'Disbursed'
# IDcol = 'ID'
# def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
#     if useTrainCV:
#         xgb_param = alg.get_xgb_params()
#         xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
#         cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
#             metrics='auc', early_stopping_rounds=early_stopping_rounds, show_progress=False)
#         alg.set_params(n_estimators=cvresult.shape[0])
#
#     #Fit the algorithm on the data
#     alg.fit(dtrain[predictors], dtrain['Disbursed'],eval_metric='auc')
#
#     #Predict training set:
#     dtrain_predictions = alg.predict(dtrain[predictors])
#     dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
#
#     #Print model report:
#     print "\nModel Report"
#     print "Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions)
#     print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob)
#
#     feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
#     feat_imp.plot(kind='bar', title='Feature Importances')
#     plt.ylabel('Feature Importance Score')
#
#
# predictors = [x for x in train.columns if x not in [target,IDcol]]
# xgb1 = XGBClassifier(
#  learning_rate =0.1,
#  n_estimators=1000,
#  max_depth=5,
#  min_child_weight=1,
#  gamma=0,
#  subsample=0.8,
#  colsample_bytree=0.8,
#  objective= 'binary:logistic',
#  nthread=4,
#  scale_pos_weight=1,
#  seed=27)
# modelfit(xgb1, train, predictors)





print "Start training Random forest..."
rfClf = ExtraTreesRegressor(n_estimators=200,n_jobs=-1)
train_data,train_label = loadTrainSet()
val_data = loadTestSet()


rfClf.fit(train_data,train_label)
#evaluate on validation set
val_pred_label = rfClf.predict(val_data)


# dtrain = xgb.DMatrix(train_data, val_data)
# dval = xgb.DMatrix(train_label, y_val)
# num_round = 60
#
# params = {"bst:max_depth": 7,
#                   "bst:eta": 0.01,
#                   "subsample": 0.8,
#                   "colsample_bytree": 1,
#                   "silent": 1,
#                   "objective": "reg:linear",
#                   "nthread": 6,
#                   "seed": 42}
#
# evallist = [(dtrain, "train"), (dval, "val")]
# bst = xgb.train(params, dtrain, num_round, evallist,verbose_eval=10, early_stopping_rounds=10)







# for row in val_data:
#     print row





f = open("new_data/train/airport_gz_passenger_predict.csv", 'wb')
write = csv.writer(f)
write.writerow(['passengerCount', 'WIFIAPTag', 'slice10min'])

file1 = csv.reader(open("new_data/countWifiByEachAP/20160914.csv",'rb'))
text1 = []
for row in file1:
    if(int(row[1])>=120 and int(row[1]) <= 145):
        text1.append(row)

for i in range(len(val_pred_label)):
    passenger = val_pred_label[i]
    words = [passenger, text1[i][0], "2016-09-14-" + text1[i][1][0:2] + "-" + text1[i][1][2]]
    write.writerow(words)
