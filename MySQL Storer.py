import pymysql.cursors

connection = pymysql.connect(user='root', password='root',
                             database='QA')

cursor = connection.cursor()

cursor.execute("CREATE TABLE IF NOT EXISTS CQA (No int, Context VARCHAR(100), Question VARCHAR(100), Answer VARCHAR(100);")
connection.commit()
