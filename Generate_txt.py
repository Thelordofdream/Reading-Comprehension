import pymysql.cursors
import requests
from bs4 import BeautifulSoup


def store_set(input_set, file_name):
    fw = open(file_name, 'w')
    fw.writelines(input_set)
    fw.close()

# Connect to the database
connection = pymysql.connect(user='root', password='root',
                             database='RC')
txt = []
num_list = ["A", "B", "C", "D", "E"]
for index in range(1, 947):
    print "-----Essay NO." + str(index) + "-----"
    txt.append("-----Essay NO." + str(index) + "-----" + "\n\n")
    cursor = connection.cursor()
    commit = "Select Context from Context where No = %d;" % index
    cursor.execute(commit)
    for context in cursor.fetchall():
        # print context[0]
        txt.append(context[0] + "\n\n")
    commit = "Select * from Question where No = %d;" % index
    cursor.execute(commit)
    num = 0
    for context in cursor.fetchall():
        # print str(num) + ". " + context[1]
        txt.append(str(num + 1) + ". " + context[1] + "\n")
        cursor0 = connection.cursor()
        commit = "Select A,B,C,D,E from Answer where No = %d;" % index
        cursor0.execute(commit)
        no = 0
        # print cursor0.fetchall()
        # print num
        # print cursor0.fetchall()
        for option in cursor0.fetchall()[-1]:
            # print num_list[no] + ". " + option
            txt.append(num_list[no] + ". " + option + "\n")
            no += 1
        # print "Right Answer is:" + context[2]
        txt.append("Right Answer is:" + context[2] + "\n\n")
        num += 1
    txt.append("\n")
# print txt
store_set(txt, "GRE.txt")

