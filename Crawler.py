# coding=utf-8
import urllib2
import re
import time
import pymysql.cursors
import requests
from bs4 import BeautifulSoup

# 阅读
# Connect to the database
connection = pymysql.connect(user='root', password='root',
                             database='RC')

cursor = connection.cursor()
commit = "CREATE TABLE IF NOT EXISTS Context (No int, Context VARCHAR(6000));"
cursor.execute(commit)
connection.commit()
commit = "CREATE TABLE IF NOT EXISTS Question (No int, Question VARCHAR(1000), Answer VARCHAR(10));"
cursor.execute(commit)
connection.commit()
commit = "CREATE TABLE IF NOT EXISTS Answer (No int, A VARCHAR(1000), B VARCHAR(1000), C VARCHAR(1000), D VARCHAR(1000), E VARCHAR(1000));"
cursor.execute(commit)
connection.commit()
No = 0
index = 0


def deal_unicode(string):
    string = string.replace("\\n", "").replace("\\u2019", "'").replace("<i>", "").replace("</i>", "").replace("\\u2013",
                                                                                                              "-").replace(
        "<b>", "").replace("</b>", "").replace("\\u201c", "\"").replace("\\u201d", "\"").replace("\\u2026",
                                                                                                 "...").replace("\xe9",
                                                                                                                "e").replace(
        "\\u2014", "-").replace("     <br/>", "")
    return string


for i in range(100):
    print "---------------------page:%d---------------------" % (i + 1)
    url = "http://gre.kmf.com/subject/lib?&t=12239101,12210104,12210202,12221202,12231101,12210105,12239103,12210111,12210204,12221204,12231102,12210112&p=2&p=%d" % (
    1 + i)
    content = urllib2.urlopen(url).read()
    # print content
    bases = re.findall(r'<b>.*?<a href="(.*?)">.*?</a></b>', content)
    # print bases
    essay = 0
    for base in bases:
        print base
        url0 = "http://gre.kmf.com" + base
        response = requests.get(url0)
        soup = BeautifulSoup(response.text, "lxml")
        try:
            context = soup.find_all("div", class_="content")[0].find_all("div")
            essay += 1
            print "---------------------essay:%d---------------------" % essay
            context = re.findall(r'<div>(.*?)</div>', str(context))[0].replace("                     ", "")
            context = context.replace("<br/><br/>", " ")
            context = deal_unicode(context)
            num = soup.find_all("ul", class_="clearfix")
            num = re.findall(r'<a enid=.*? href="(.*?)">', str(num))
            get = False
            for url1 in num:
                index += 1
                url2 = "http://gre.kmf.com" + url1
                response = requests.get(url2)
                soup0 = BeautifulSoup(response.text, "lxml")
                question = soup0.find_all("div", class_="options")[0].find_all("div", class_="mb20")
                question = re.findall(r'<div class="mb20">(.*?)</div>', str(question))[0]
                question = deal_unicode(question).replace(u'\u2014', "-").replace("         ", "").replace("       ",
                                                                                                           "")
                print question
                answer = soup0.find_all("div", class_="que-anser-myanswer", id="ShowAnswer")[0].find("b").string
                print answer
                if not answer is None and len(answer) <= 4:
                    if not get:
                        No += 1
                        with connection.cursor() as cursor:
                            # Create a new record
                            sql = "INSERT INTO Context "
                            sql += "(No, Context) VALUES (%s, %s)"
                            cursor.execute(sql, (No, context))
                        connection.commit()
                        # print context
                    get = True
                    with connection.cursor() as cursor:
                        # Create a new record
                        sql = "INSERT INTO Question "
                        sql += "(No, Question, Answer) VALUES (%s, %s, %s)"
                        cursor.execute(sql, (No, question, answer))
                    connection.commit()
                    options = soup0.find_all("li", class_="clearfix")
                    A = re.findall(r'<span><strong>.*?</strong>(.*?)</span>', str(options[0]))[0]
                    A = deal_unicode(A)
                    B = re.findall(r'<span><strong>.*?</strong>(.*?)</span>', str(options[1]))[0]
                    B = deal_unicode(B)
                    C = re.findall(r'<span><strong>.*?</strong>(.*?)</span>', str(options[2]))[0]
                    C = deal_unicode(C)
                    try:
                        D = re.findall(r'<span><strong>.*?</strong>(.*?)</span>', str(options[3]))[0]
                        D = deal_unicode(D)
                    except IndexError:
                        D = ""
                    try:
                        E = re.findall(r'<span><strong>.*?</strong>(.*?)</span>', str(options[4]))[0]
                        E = deal_unicode(E)
                    except IndexError:
                        E = ""
                    # print A, B, C, D, E
                    with connection.cursor() as cursor:
                        # Create a new record
                        sql = "INSERT INTO Answer "
                        sql += "(No, A, B, C, D, E) VALUES (%s, %s, %s, %s, %s, %s)"
                        cursor.execute(sql, (No, A, B, C, D, E))
                    connection.commit()
        except IndexError:
            pass
connection.close()
print "total:%d" % index
