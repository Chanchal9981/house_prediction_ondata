# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''x=pd.read_csv("C:\\Users\\patid\\Downloads\\archive (1)\\Bengaluru_House_Data.csv")
x'''
x1=r"https://drive.google.com/uc?export=download&id=1nz2hqpRLXQtW2qSGSyClctKa1QP7iL42"
x=pd.read_csv(x1)
x

pd.set_option('display.max_column',None)
pd.set_option('display.max_row',None)

x.isnull().sum()

x_new=x.dropna()
x_new.isnull().sum()

x_new['bhk']=x_new['size'].apply(lambda x: int(x.split(' ')[0]))
x_new.head()

x_new["bhk"].unique()

x_new[x_new.bhk>10]

x_new1=x_new.drop(columns=["area_type","availability","size","society"],axis=1)
x_new1

#to check values which have contain range value format
def is_float(x):
    try:
        float (x)
    except:
        return False
    return True

x_new1[~x_new1["total_sqft"].apply(is_float)].head()

#convert this range value to exact value
def convert_value(x):
    tokens=x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None

df=x_new1.copy()
df["total_sqft"]=df["total_sqft"].apply(convert_value)

df

df["sqrt_per_feet"]=df["price"]*100000/df["total_sqft"]
df

len(df["location"].unique())

df.location=df.location.apply(lambda x:x.strip())
locations_state=df.groupby('location')['location'].agg('count').sort_values(ascending=False)
locations_state

location_state_lesathen_10=locations_state[locations_state<=10]
location_state_lesathen_10

df.location=df.location.apply(lambda x: 'other' if x in location_state_lesathen_10 else x)
len(df.location.unique())
df.head(20)

df[df.total_sqft/df.bhk<300].head()

df1=df[~(df.total_sqft/df.bhk<300)]
df2=df1.fillna(df1.mean())
df2.isnull().sum()

X=df2.drop("price",axis=1)
y=df2["price"]
print("shape of X is",X.shape)
print("shape of y is",y.shape)
x_dummy=pd.get_dummies(X,drop_first=True)

from sklearn.preprocessing import MinMaxScaler

mn=MinMaxScaler()
x_dummy=mn.fit_transform(x_dummy)
x_dummy

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(x_dummy,y,test_size=0.3,random_state=51)
print("shape of X_train is",X_train.shape)
print("shape of X_test is",X_test.shape)
print("shape of y_train is",y_train.shape)
print("shape of y_test is",y_test.shape)

from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(X_train,y_train)

lr.score(X_test,y_test)

from sklearn.linear_model import Ridge,Lasso

ls=Lasso()
ls.fit(X_train,y_train)
ls.score(X_test,y_test)

rd=Ridge()
rd.fit(X_train,y_train)
rd.score(X_test,y_test)

import joblib

data_files="model_house_data"
joblib.dump(lr,data_files)

