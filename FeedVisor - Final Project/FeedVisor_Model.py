import tensorflow as tf
import pandas as pd
from sqlalchemy import create_engine
from random import random
fileName = 'listingp1.csv'
import csv

data = pd.read_csv(fileName, nrows = 20)
print(data)

# Converter
def conv(val):
    if not val:
        return 0
    try:
        return np.float64(val)
    except:
        return np.float64(0)

file = 'listingp1.csv'
# create an sqllite mini database
csv_database = create_engine('sqlite:///csv_database.db')
chunksize = 100000
i = 0
j = 1
# read the data from csv in chunks and write to the database
for df in pd.read_csv(file, chunksize=chunksize, iterator=True):
    df = df.rename(columns={c: c.replace(' ', '') for c in df.columns}) #add converter
    df.index += j
    i+=1
    df.to_sql('table', csv_database, if_exists='append')
    j = df.index[-1] + 1

# unique without sqllite
# # print(data)
# s = data.take([0], axis=1)
# s=s.rename(columns = {'2016-03-20':'id'})
# s=s.id.unique()


# from here - solutions of stocks
import numpy as np

# def _load_data(data, n_prev = 10):
#     """
#     data should be pd.DataFrame()
#     """
#
#     docX, docY = [], []
#     for i in range(len(data)-n_prev):
#         docX.append(data.iloc[i:i+n_prev].as_matrix())
#         docY.append(data.iloc[i+n_prev].as_matrix())
#     alsX = np.array(docX)
#     alsY = np.array(docY)
#
#     return alsX, alsY
#
#
# def train_test_split(df, test_size = 0.1):
#     """
#     This just splits data to training and testing parts
#     """
#     ntrn = round(len(df) * (1 - test_size))
#
#     X_train, y_train = _load_data(df.iloc[0:ntrn])
#     X_test, y_test = _load_data(df.iloc[ntrn:])
#
#     return (X_train, y_train), (X_test, y_test)
#
# (X_train, y_train), (X_test, y_test) = train_test_split(data)
#
# print(X_train)
# print(X_test)


# flow = (list(range(1,10,1)) + list(range(10,1,-1)))*1000
# pdata = pd.DataFrame({"a":flow, "b":flow})
# pdata.b = pdata.b.shift(9)
# data = pdata.iloc[10:] * random()  # some noise
# print (data)



# keren tries
# node1 = tf.constant(3.0, tf.float32)
# node2 = tf.constant(4.0) # also tf.float32 implicitly
# print(node1, node2)
# filename_queue = tf.train.string_input_producer(["test.csv"])
#
# flow = (list(range(1,10,1)) + list(range(10,1,-1)))*2
# print (flow)
# x = list(range(1,1,2))
# print(x)

