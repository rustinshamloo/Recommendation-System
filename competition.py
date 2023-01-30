
# Description of the method: My approach consists of the following components combined linearly:

# 1. I used a similar method to hwk3 to determine the item-based CF prediction.
# 2. XGBregressor: I used the same procedure as in hwk3 to generate XGBregressor prediction, but I made some adjustments for better results.
# 3. Took the average rate for each business from business (column name: bus mean). 
# 4. Took  the review count for each business from the business (column name: bus review count).
# 5. From user.json, took average stars value to get the user's average rating (column name: user mean).
# 6. From user.json-> the number of "useful"s sent by the user (column name: user useful).
# 7. The weighted rate of a business (column name: useful bus rate)-> A business review will be given more weight if more users have rated it "useful".
# To save execution time, I kept track of the coefficients and intersections for each part and applied them to the data.

# Error Distribution:

# >=0 and <1: 41286 
# >=1 and <2: 15244 
# >=2 and <3: 5260
# >=3 and <4: 795
# >=4: 1


# RMSE: 0.9796709211544142
# Time: 660 sec



import numpy as np
import xgboost as xgb
from itertools import combinations
import math
import random
from operator import itemgetter
from operator import add
from pyspark import SparkContext
import sys
import csv
from pyspark import SparkConf
import json


def geting_total_sum(x):
    return sum(x.values())


def get_length_of_str(x):
    if x != "None":
        list_x = x.split(", ")
        return len(list_x)
    else:
        return 0


def get_num_of_comp(x):
    sum_ = 0
    for x_ in x:
        sum_ += x_
    return sum_


def total_star(x, y):
    return x * y


folder_path = sys.argv[1]
test_file_name = sys.argv[2]
output_file_name = sys.argv[3]


sc = SparkContext.getOrCreate()
sc.setLogLevel("WARN")


all_train_rdd = sc.textFile(folder_path + 'yelp_train.csv')
header_train = all_train_rdd.first()
train_rdd = all_train_rdd.filter(lambda x: x != header_train).map(
    lambda x: (str(x.split(',')[0]), (str(x.split(',')[1]), float(x.split(',')[2]))))

all_val_rdd = sc.textFile(test_file_name)
header_val = all_val_rdd.first()
val_rdd = all_val_rdd.filter(lambda x: x != header_train).map(
    lambda x: (str(x.split(',')[0]), str(x.split(',')[1])))

all_user_rdd = sc.textFile(folder_path + 'user.json').map(lambda line: json.loads(line))

user_rdd = all_user_rdd.map(lambda x: (x['user_id'], (float(x['useful']), float(x['funny']), float(x['cool']),
                                                      float(x['review_count']), float(get_length_of_str(x['friends'])),
                                                      float(x['fans']), get_length_of_str(x['elite']),
                                                      float(x['average_stars']),
                                                      float(get_num_of_comp(
                                                          [x['compliment_hot'], x['compliment_more'],
                                                            x['compliment_profile'], x['compliment_cute'],
                                                            x['compliment_list'], x['compliment_note'],
                                                            x['compliment_plain'], x['compliment_cool'],
                                                            x['compliment_funny'], x['compliment_writer'],
                                                            x['compliment_photos']])))))



all_business_rdd = sc.textFile(folder_path + 'business.json').map(lambda line: json.loads(line))
business_rdd = all_business_rdd.map(
    lambda k: (k['business_id'], (float(k['stars']), float(total_star(k['stars'], k['review_count'])))))

all_checkin_rdd = sc.textFile(folder_path + 'checkin.json').map(lambda line: json.loads(line))
checkin_rdd = all_checkin_rdd.map(lambda k: (k['business_id'], geting_total_sum(k['time'])))
train_data_rdd = train_rdd.leftOuterJoin(user_rdd).map(
    lambda x: (x[1][0][0], (x[0], x[1][0][1], x[1][1]))).leftOuterJoin(business_rdd).leftOuterJoin(
    checkin_rdd).map(
    lambda x: ((x[1][0][0][0], x[0]), x[1][0][0][1], x[1][0][0][2], x[1][0][1], x[1][1])).map(
    lambda s: (s[0], (s[1],) + s[2] + s[3] + (s[4],)))


val_data_rdd = val_rdd.leftOuterJoin(user_rdd).map(lambda x: (x[1][0], (x[0], x[1][1]))).leftOuterJoin(
    business_rdd).leftOuterJoin(checkin_rdd).map(
    lambda x: ((x[1][0][0][0], x[0]), x[1][0][0][1], x[1][0][1], x[1][1])).map(
    lambda s: (s[0], s[1] + s[2] + (s[3],)))


X_train = train_data_rdd.map(lambda x: x[1][1:]).collect()
X_test = val_data_rdd.map(lambda x: x[1]).collect()
y_train = train_data_rdd.map(lambda x: x[1][0]).collect()

params = {
    'booster': 'gbtree', 'objective': 'reg:linear','gamma': 0.1, 'max_depth': 10,
    'lambda': 10, 'subsample': 0.8, 'colsample_bytree': 0.7,
    'min_child_weight': 15, 'eta': 0.01, 'seed': 321, 'nthread': 4,}

xgtrain = xgb.DMatrix(np.array(X_train), np.array(y_train))
num_rounds = 1600
model = xgb.train(params, xgtrain, num_rounds)
xgtest = xgb.DMatrix(np.array(X_test))
ans = model.predict(xgtest)

result = val_data_rdd.map(lambda x: x[0]).collect()
for i in range(len(ans)):
    result[i] =result[i]  + (ans[i],) 

with open(output_file_name, "w+") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["user_id", " business_id", " prediction"])
    writer.writerows(result)

