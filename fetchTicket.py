import csv
import os
import numpy as np
csv_file_object = csv.reader(open('new_data/airport_gz_departure_chusai_1stround.csv', 'rb'))
header = csv_file_object.next()  # The next() command just skips the

data0 = []
data1 = []
data2 = []
data3 = []
data4 = []

def saveResult(result,name):
    with open("new_data/fetchTicket/"+name+'.csv', 'wb') as myFile:
        myWriter = csv.writer(myFile,dialect='excel')
        dictionary = {}
        for i in result:
            if(dictionary.has_key(i[0])):
                dictionary[i[0]] += 1
            else:
                dictionary[i[0]] = 1
        for i in result:
            words = i
            words.append(dictionary[i[0]])
            myWriter.writerow(words)

for row in csv_file_object:
    date = row[3].split(" ")[0]
    hour = row[3].split(" ")[1]
    if (date == "2016/9/10"):
        words = row[1:2]
        words.append(hour[0:4])
        data0.append(words)
    elif (date == "2016/9/11"):
        words = row[1:2]
        words.append(hour[0:4])
        data1.append(words)
    elif (date == "2016/9/12"):
        words = row[1:2]
        words.append(hour[0:4])
        data2.append(words)
    elif (date == "2016/9/13"):
        words = row[1:2]
        words.append(hour[0:4])
        data3.append(words)
    elif (date == "2016/9/14"):
        words = row[1:2]
        words.append(hour[0:4])
        data4.append(words)


saveResult(data0,"20160910")
saveResult(data1,"20160911")
saveResult(data2,"20160912")
saveResult(data3,"20160913")
saveResult(data4,"20160914")

