# coding=utf-8
import pymysql.cursors
import numpy as np


def store_set(input_set, file_name):
    fw = open(file_name, 'w')
    fw.writelines(input_set)
    fw.close()


def storeVecs(input, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(input, fw)
    fw.close()


def grabVecs(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)


def pre_processing(table):
    connection = pymysql.connect(user='root', password='root',
                                 database='QA')
    cursor = connection.cursor()
    commit = "select * from %s;" % table
    cursor.execute(commit)
    vocab_Set = set([])
    length_c = []
    length_q = []
    length_a = []
    for each in cursor.fetchall():
        context = each[1].split(" ")
        length_c.append(len(context))
        question = each[2].split(" ")
        length_q.append(len(question))
        answer = each[3].split(" ")
        length_a.append(len(answer))
        vocab_Set = vocab_Set | set(context + question + answer)
    length = len(vocab_Set)
    len_c = max(length_c)
    len_q = max(length_q)
    len_a = max(length_a)

    vocab_dict = {}
    no = 0
    for i in vocab_Set:
        vocab_dict[i] = no + 1
        no += 1
    storeVecs(vocab_dict, "./Data for input/vocabTree.pkl")
    answer_dict = {'hallway': 1, 'garden': 2, 'office': 3, 'kitchen': 4, 'bathroom': 5, 'bedroom': 6}
    vocab = []
    for i in vocab_Set:
        i += "\n"
        vocab += i
    store_set(vocab, "./Data for input/vocabSet.txt")

    c = []
    q = []
    a = []
    commit = "select * from %s;" % table
    cursor.execute(commit)
    for each in cursor.fetchall():
        context = each[1].split(" ")
        # sentence = np.zeros((len_c, length), dtype="int32")
        sentence = [0 for n in range(len_c)]
        m = len(context)
        start = int((len_c - m) / 2)
        i = 0
        for word in context:
            # vector = [0 for n in range(length)]
            # vector[vocab_dict[word] - 1] = 1
            # sentence[start + i] = vector
            sentence[start + i] = vocab_dict[word]
            i += 1
        c.append(sentence)

        question = each[2].split(" ")
        # sentence = np.zeros((len_q, length), dtype="int32")
        sentence = [0 for n in range(len_q)]
        m = len(question)
        start = int((len_q - m) / 2)
        i = 0
        for word in question:
            # vector = [0 for n in range(length)]
            # vector[vocab_dict[word] - 1] = 1
            # sentence[start + i] = vector
            sentence[start + i] = vocab_dict[word]
            i += 1
        q.append(sentence)

        answer = each[3].split(" ")
        # sentence = np.zeros((len_a, length), dtype="int32")
        # m = len(answer)
        # start = int((len_a - m) / 2)
        # i = 0
        for word in answer:
            vector = [0 for n in range(6)]
            vector[answer_dict[word] - 1] = 1
            # sentence[start + i] = vector
            # i += 1
            a.append(vector)
    storeVecs(c, "./Data for input/context.pkl")
    storeVecs(q, "./Data for input/question.pkl")
    storeVecs(a, "./Data for input/answer.pkl")


pre_processing("CQA2")
