import csv
import os
import numpy as np
csv_file_object = csv.reader(open('new_data/airport_gz_security_check_chusai_1stround.csv', 'rb'))
header = csv_file_object.next()  # The next() command just skips the

data0 = []
data1 = []
data2 = []
data3 = []
data4 = []

def saveResult(result,name):
    with open("new_data/passSecurity/"+name+'.csv', 'wb') as myFile:
        myWriter = csv.writer(myFile,dialect='excel')
        dictionary = {}
        for i in result:
            if(dictionary.has_key(i[0])):
                dictionary[i[0]] += 1
            else:
                dictionary[i[0]] = 1
        for j in result:
            text = j
            text.append(dictionary[j[0]])
            myWriter.writerow(text)

for row in csv_file_object:
    date = row[1].split(" ")[0]
    hour = row[1].split(" ")[1]
    if (date == "2016-09-10"):
        words = []
        words.append(row[2])
        if(len(hour)==8):
            words.append(hour[0:4])
            print hour[0:4]
        elif(len(hour)==7):
            words.append(hour[0:3])
        data0.append(words)
    elif (date == "2016-09-11"):
        words = []
        words.append(row[2])
        if (len(hour) == 8):
            words.append(hour[0:4])
        elif (len(hour) == 7):
            words.append(hour[0:3])
        data1.append(words)
    elif (date == "2016-09-12"):
        words = []
        words.append(row[2])
        if (len(hour) == 8):
            words.append(hour[0:4])
        elif (len(hour) == 7):
            words.append(hour[0:3])
        data2.append(words)
    elif (date == "2016-09-13"):
        words = []
        words.append(row[2])
        if (len(hour) == 8):
            words.append(hour[0:4])
        elif (len(hour) == 7):
            words.append(hour[0:3])
        data3.append(words)
    elif (date == "2016-09-14"):
        words = []
        words.append(row[2])
        if (len(hour) == 8):
            words.append(hour[0:4])
        elif (len(hour) == 7):
            words.append(hour[0:3])
        data4.append(words)


saveResult(data0,"20160910")
saveResult(data1,"20160911")
saveResult(data2,"20160912")
saveResult(data3,"20160913")
saveResult(data4,"20160914")

