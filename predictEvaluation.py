import numpy as np
import csv
import time

print "======= star ======="
t0 = time.time()

predict_value_dict = {}
standard_value_dict = {}


predict_value = csv.reader(open("new_data/airport_gz_passenger_predict.csv",'rb'))
standard_value = csv.reader(open("airport_gz_passenger_predict_standard.csv",'rb'))
header1 = predict_value.next()
header2 = standard_value.next()

for row in predict_value:
    row[1] = row[1].upper()
    sample = (row[1],row[2])
    predict_value_dict[sample] = float(row[0])

for row in standard_value:
    sample = (row[1],row[2])
    standard_value_dict[sample] = float(row[0])

score = 0
for key in predict_value_dict:
    score += (float(predict_value_dict[key]) - float(standard_value_dict[key])) * (float(predict_value_dict[key]) - float(standard_value_dict[key]))
print score

t1 = time.time()
print "======= end  ======="
print "It take %f s to process" % (t1 - t0)