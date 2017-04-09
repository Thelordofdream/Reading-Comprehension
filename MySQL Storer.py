# coding=utf-8
import pymysql.cursors
import string

connection = pymysql.connect(user='root', password='root',
                             database='QA')

cursor = connection.cursor()

commit = "CREATE TABLE IF NOT EXISTS CQA2 (No int, Context VARCHAR(100), Question VARCHAR(100), Answer VARCHAR(100));"
cursor.execute(commit)
connection.commit()

filename = "./data/en/qa1_single-supporting-fact_train.txt"
fr = open(filename)
lines = fr.readlines()
number = len(lines)/3
for i in range(number):
    context_list = lines[3 * i].replace(".\n", "").split(" ")[1:]
    context_list += lines[3 * i + 1].replace(".\n", "").split(" ")[1:]
    context = ""
    for word in context_list[:-1]:
        context += word + " "
    context += context_list[-1]
    QA = lines[3 * i + 2].split("? 	")
    question_list = QA[0].split(" ")[1:]
    question = ""
    for word in question_list[:-1]:
        question += word + " "
    question += question_list[-1]
    answer = QA[1].replace("\n", "").replace("\t", " ").split(" ")[0]
    print answer
    with connection.cursor() as cursor:
        # Create a new record
        sql = "INSERT INTO CQA2 "
        sql += "(No, Context, Question, Answer) VALUES (%s, %s, %s, %s)"
        cursor.execute(sql, (i+1, context, question, answer))
    connection.commit()


