import csv
import os
import numpy as np
import operator
csv_file_object = csv.reader(open('new_data/WIFI_AP_Passenger_Records_chusai_1stround.csv', 'rb'))
header = csv_file_object.next()  # The next() command just skips the

data0 = []
data1 = []
data2 = []
data3 = []
data4 = []

def saveResult(result,name):
    with open("new_data/countWifiByEachAP/"+name+'.csv', 'wb') as myFile:
        myWriter = csv.writer(myFile,dialect='excel')
        dictionary = {}
        data = []
        for row in result:
            sample = (row[0], row[2])
            if (dictionary.has_key(sample)):
                dictionary[sample] += int(row[1])
            else:
                dictionary[sample] = int(row[1])
        for k, v in dictionary.iteritems():
            text = []
            text.append(k[0])
            text.append(k[1])
            text.append(v)
            data.append(text)
        data = sorted(data, key=operator.itemgetter(1, 0))
        myWriter.writerows(data)



for row in csv_file_object:
    if (row[2][8:10] == "10"):
        row[2] = row[2][11:13]+row[2][14]
        data0.append(row)
    elif (row[2][8:10] == "11"):
        row[2] = row[2][11:13]+row[2][14]
        data1.append(row)
    elif (row[2][8:10] == "12"):
        row[2] = row[2][11:13]+row[2][14]
        data2.append(row)
    elif (row[2][8:10] == "13"):
        row[2] = row[2][11:13]+row[2][14]
        data3.append(row)
    elif (row[2][8:10] == "14"):
        row[2] = row[2][11:13]+row[2][14]
        data4.append(row)

saveResult(data0,"20160910")
saveResult(data1,"20160911")
saveResult(data2,"20160912")
saveResult(data3,"20160913")
saveResult(data4,"20160914")

