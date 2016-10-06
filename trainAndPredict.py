import csv
file1 = csv.reader(open("new_data/countWifiByEachAP/20160912.csv",'rb'))
file2 = csv.reader(open("new_data/countWifiByEachAP/20160913.csv",'rb'))
file3 = csv.reader(open("new_data/countWifiByEachAP/20160911.csv",'rb'))
text1 = []
text2 = []
text3 = []
for row in file1:
    if(int(row[1])>=120 and int(row[1]) <= 145):
        text1.append(row)

for row in file2:
    if(int(row[1])>=120 and int(row[1]) <= 145):
        text2.append(row)

# for row in file3:
#     if(int(row[1])>=113 and int(row[1]) <= 115):
#         text3.append(row)
for row in file3:
    if(int(row[1])>=120 and int(row[1]) <= 145):
        text3.append(row)




x = 0.025
y = 0.015
z = 0.005
# while x < 0.1 and y < 0.1 :
#     x += 0.005
while x < 0.050:
    while y < 0.050:
        while z < 0.030:
            f = open("new_data/airport_gz_passenger_predict.csv", 'wb')
            write = csv.writer(f)
            write.writerow(['passengerCount', 'WIFIAPTag', 'slice10min'])

            for i in range(len(text1)):
                passenger = x * float(text1[i][2]) + y * float(text2[i][2]) + z * float(text3[i][2])
                words = [passenger, text1[i][0], "2016-09-14-" + text1[i][1][0:2] + "-" + text1[i][1][2]]
                write.writerow(words)
            # for i in range(len(wifi_ap_tag_passenger_ten)):
            #     for j in range(len(date_list_before)):
            #         passenger = 0.4 * passenger_cou + 0.5 * passenger_ago + 0.1 * passenger_ago_one + 0.1 * passenger_ago_two
            #         words = [passenger, rank_to_wifi_ap[i], slice_ten_min[j]]
            #         write.writerow(words)

            f.close()

            import time

            # print "======= star ======="
            t0 = time.time()

            predict_value_dict = {}
            standard_value_dict = {}

            predict_value = csv.reader(open("new_data/airport_gz_passenger_predict.csv", 'rb'))
            standard_value = csv.reader(open("airport_gz_passenger_predict_standard.csv", 'rb'))
            header1 = predict_value.next()
            header2 = standard_value.next()

            for row in predict_value:
                row[1] = row[1].upper()
                sample = (row[1], row[2])
                predict_value_dict[sample] = float(row[0])

            for row in standard_value:
                sample = (row[1], row[2])
                standard_value_dict[sample] = float(row[0])

            score = 0
            for key in predict_value_dict:
                score += (float(predict_value_dict[key]) - float(standard_value_dict[key])) * (float(predict_value_dict[key]) - float(standard_value_dict[key]))

            if(score <= 156782.0 ):
                print score
                print "x = %f "%x
                print " and y =%f "%y
                print " and z =%f " %z
            t1 = time.time()
            # print "======= end  ======="
            # print "It take %f s to process" % (t1 - t0)
            z += 0.002
            if(z >= 0.028 ):
                z = 0.006
                break
        y += 0.002
        if(y>=0.048):
            y = 0.014
            break
    x += 0.002
