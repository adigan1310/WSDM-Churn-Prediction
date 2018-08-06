# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 17:26:10 2017

@authors: Adithya Ganapathy(axg172330)
          Sri Hari Murali(sxm179330)
          Nisshantni Divakaran(nxd171330)
"""
import gc; gc.enable()
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sys
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt

#To reduce the file size by changing datatype of integer and float columns
def reducesize(df):
    int_cols = list(df.select_dtypes(include=['int']).columns)
    for col in int_cols:
        if ((np.max(df[col]) <= 127) and(np.min(df[col] >= -128))):
            df[col] = df[col].astype(np.int8)
        elif ((np.max(df[col]) <= 32767) and(np.min(df[col] >= -32768))):
            df[col] = df[col].astype(np.int16)
        elif ((np.max(df[col]) <= 2147483647) and(np.min(df[col] >= -2147483648))):
            df[col] = df[col].astype(np.int32)
        else:
            df[col] = df[col].astype(np.int64)         
    float_cols = list(df.select_dtypes(include=['float']).columns)
    for col in float_cols:
        df[col] = df[col].astype(np.float32)
        
#Get folder location of files
filepath = sys.argv[1]

#Read train and test file
train = pd.read_csv(filepath + "/train.csv")
test = pd.read_csv(filepath + "/test.csv")

#Getting transactions count and merging with test and train file
transactions = pd.read_csv(filepath + "/transactions.csv", usecols=['msno'])
transactions = pd.DataFrame(transactions['msno'].value_counts().reset_index())
transactions.columns = ['msno','trans_count']
train = pd.merge(train, transactions, how='left', on='msno')
test = pd.merge(test, transactions, how='left', on='msno')
transactions = []
print("transaction count merged")

#Getting user logs count and merging with test and train file
user_logs = pd.read_csv(filepath + "/user_logs.csv", usecols=['msno'])
user_logs = pd.DataFrame(user_logs['msno'].value_counts().reset_index())
user_logs.columns = ['msno','days_listened']
train = pd.merge(train, user_logs, how='left', on='msno')
test = pd.merge(test, user_logs, how='left', on='msno')
user_logs = []
print("user logs count merged")

#Merging members file with test and train file
members = pd.read_csv(filepath + "/members.csv")
train = pd.merge(train, members, how='left', on='msno')
test = pd.merge(test, members, how='left', on='msno')
members = [];
print("members merged")

#Merging transactions file with train and test file
transactions = pd.read_csv(filepath + "/transactions.csv")
transactions = transactions.sort_values(by=['transaction_date'], ascending=[False]).reset_index(drop=True)
transactions = transactions.drop_duplicates(subset=['msno'], keep='first')
train = pd.merge(train, transactions, how='left', on='msno')
test = pd.merge(test, transactions, how='left', on='msno')
transactions=[]
print("transactions merged")

#Merging user_logs file with test and train file
user_logs = pd.read_csv(filepath + "/user_logs.csv")
del user_logs['date']
user_logs = user_logs.groupby(["msno"], as_index=False)["num_25", "num_50", "num_75", "num_985", "num_100", "num_unq", "total_secs"].sum()  
train = pd.merge(train,user_logs,how='left',on='msno')
test = pd.merge(test,user_logs,how='left',on='msno') 
user_logs=[] 
print("userlogs merged")

#Changing gender value to numerical, filling NAN's with 0
#Adding two new features discount and is_discount
gender = {'male':1, 'female':2}
train['gender'] = train['gender'].map(gender)
test['gender'] = test['gender'].map(gender)
train['discount'] = train['plan_list_price'] - train['actual_amount_paid']
test['discount'] = test['plan_list_price'] - test['actual_amount_paid']
train['is_discount'] = train.discount.apply(lambda x: 1 if x > 0 else 0)
test['is_discount'] = test.discount.apply(lambda x: 1 if x > 0 else 0)

#Filling NAN's with 0 and reducing data size
train = train.fillna(0)
test = test.fillna(0)
reducesize(train)
reducesize(test)

#Removing unnamed columns from the dataframe
for col in train.columns:
    if 'Unnamed' in col:
        del train[col]
for col in test.columns:
    if 'Unnamed' in col:
        del test[col]
        
train.to_csv(path_or_buf= filepath + "/trainfinal.csv", index=False)
test.to_csv(path_or_buf= filepath + "/testfinal.csv", index=False)
print("Exported")
train = []
test = []

#Obtaining the columns required for training the model
train = pd.read_csv(filepath + "/trainfinal.csv")
test = pd.read_csv(filepath + "/testfinal.csv")
cols = [c for c in train.columns if c not in ['is_churn','msno']]

#Pre-processing the file with Robust Scaler
scaler = RobustScaler()
scaler.fit(train[cols])
train_x = scaler.transform(train[cols])
test_x = scaler.transform(test[cols])
train_y = train['is_churn']
print("Pre-processing completed")

#Training Random Forest Classifier
model = RandomForestClassifier(n_estimators = 50)
model.fit(train_x,train_y)
print("Training Completed")

#Predicting the test data with the trained model
predictions = model.predict(test_x)

#Exporting the msno and predicted values to a csv file
submission = pd.DataFrame()
submission['msno'] = test['msno']
submission['is_churn'] = predictions
submission.set_index('msno' )
submission.to_csv(path_or_buf="C:/Users/adith/Desktop/Submission.csv", index=False)
print("Prediction of WSDM KKBOX Completed")

#Printing feature imporatance in a graph
importance = model.feature_importances_
importance = pd.DataFrame(importance,index=train[cols].columns)
x = range(importance.shape[0])
y = importance.iloc[:, 0]
plt.figure(figsize=(10,10))
plt.bar(x, y, align="center")
plt.xticks(range(len(importance)), list(importance.index.values),rotation='vertical')
plt.show()
