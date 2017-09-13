# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 17:37:32 2017

@author: Nishant
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing
import xgboost as xgb

color = sns.color_palette
pd.options.mode.chained_assignment =None
pd.set_option('display.max_columns', 500)

train_df = pd.read_csv('data/train.csv')

#print(train_df.shape)
#print(train_df.head())

Y = train_df.price_doc
X = train_df.drop('price_doc', axis = 1)

plt.figure(figsize=(10,8))
plt.scatter(range(train_df.shape[0]),np.sort(Y.values))
plt.xlabel('Index', fontsize = 12)
plt.ylabel('Price', fontsize = 12)
plt.show()

#We can now bin the 'price_doc' and plot it.
#seaborn.distplot function combines the matplotlib hist function 
#(with automatic calculation of a good default bin size) 
#with the seaborn kdeplot() and rugplot() functions. 
#It can also fit scipy.stats distributions and plot the estimated PDF over the data.

plt.figure(figsize=(12,8))
sns.distplot(Y.values, bins = 50, kde = True)
plt.xlabel('Price', fontsize = 12)
plt.show()

#Certainly a very long right tail. Since our metric is Root Mean Square Logarithmic error, 
#let us plot the log of price_doc variable.

plt.figure(figsize=(10,8))
sns.distplot(np.log(Y.values), bins = 50, kde = True)
plt.xlabel('Price', fontsize = 12)
plt.show()

#Now lets see how median house price change with time.

train_df['yearmonth'] = train_df['timestamp'].apply(lambda x: x[:4]+x[5:7])
grouped_df = train_df.groupby('yearmonth')['price_doc'].aggregate(np.median).reset_index()


plt.figure(figsize=(10,8))
sns.barplot(grouped_df.yearmonth.values, grouped_df.price_doc.values, alpha = 1.0)
plt.xlabel('Year Month', fontsize = 12)
plt.ylabel('Median Price', fontsize = 12)
plt.xticks(rotation='vertical')
plt.show()

#Now let us dive into other variables and see. 
#Let us first start with getting the count of different data types.

train_df = pd.read_csv('data/train.csv', parse_dates = ['timestamp'])
dtype_df = train_df.dtypes.reset_index()
dtype_df.columns = ['Count', 'Column_type']
print(dtype_df.groupby('Column_type').aggregate('count').reset_index())


#So majority of data is numerical values, 15 object type, and 1 date type
#now lets check for the missing values

missing_df = train_df.isnull().sum(axis = 0).reset_index()
missing_df.columns= ['Column_Name', 'Missing_Count']
missing_df = missing_df.loc[missing_df['Missing_Count']>0]
ind = np.arange(missing_df.shape[0])
print(ind)
width = 0.9
fig, pl = plt.subplots(figsize=(12,20))
rects = pl.barh(ind, missing_df.Missing_Count.values, color='y')
pl.set_yticks(ind)
pl.set_yticklabels(missing_df.Column_Name.values, rotation='horizontal')
pl.set_xlabel("Count of missing values")
pl.set_title("Number of missing values in each column")
plt.show()

#Seems variables are found to missing as groups.
#Since there are 292 variables, 
#let us build a basic xgboost model and then explore only the important variables.

for f in train_df.columns:
    if train_df[f].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[f].values))
        train_df[f] = lbl.transform(list(train_df[f].values))
    
train_y = train_df.price_doc.values
train_x = train_df.drop(["id", "timestamp", "price_doc"], axis=1)

xgb_params = {
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}
dtrain = xgb.DMatrix(train_x, train_y, feature_names=train_x.columns.values)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)

# plot the important features #
fig, ax = plt.subplots(figsize=(12,60))
xgb.plot_importance(model, height=0.8, ax=ax) #max_num_features=50(eleminated due to error/xgboost need to be updated)
plt.show()

#So the top 5 important features and their description from the data dictionary are:
#full_sq - total area in square meters, including loggias, balconies and other non-residential areas
#life_sq - living area in square meters, excluding loggias, balconies and other non-residential areas
#floor - for apartments, floor of the building
#max_floor - number of floors in the building
#build_year - year built
#Now let us see how these important variables are distributed with respect to target variable.
#Total area in square meters:

    #Using full_sq
    
ulimit = np.percentile(train_df.price_doc.values, 99.5)
llimit = np.percentile(train_df.price_doc.values, 0.5)
train_df['price_doc'].loc[train_df['price_doc']>ulimit] = ulimit
train_df['price_doc'].loc[train_df['price_doc']<llimit] = llimit

    #Using full_sq

col = 'full_sq'
ulimit = np.percentile(train_df[col].values, 99.5)
llimit = np.percentile(train_df[col].values, 0.5)
train_df[col].loc[train_df[col]>ulimit] = ulimit
train_df[col].loc[train_df[col]<llimit] = llimit

plt.figure(figsize=(12,12))
sns.jointplot(x = np.log1p(train_df.full_sq.values), y = np.log1p(train_df.price_doc.values), size = 10)
plt.xlabel('Log of price', fontsize = 12)
plt.ylabel('log of total area in square meter', fontsize = 12)
plt.show()


#Living area in square meters:
    
    #Using life_sq
    
col = 'life_sq'
train_df[col].fillna(0, inplace = True)
ulimit = np.percentile(train_df[col].values, 95)
llimit = np.percentile(train_df[col].values, 5)
train_df[col].loc[train_df[col]>ulimit] = ulimit
train_df[col].loc[train_df[col]<llimit] = llimit

plt.figure(figsize = (12,12))
sns.jointplot(x = np.log1p(train_df.life_sq.values), y=np.log1p(train_df.price_doc.values), kind='kde', size=10)
plt.ylabel('Log of Price', fontsize=12)
plt.xlabel('Log of living area in square metre', fontsize=12)
plt.show()


#Floor:
#We will see the count plot of floor variable.

    #Using Floors for apartment
    
plt.figure(figsize=(12,8))
sns.countplot(x = 'floor', data = train_df)
plt.xlabel('floor number', fontsize = 12)
plt.ylabel('Floor count', fontsize = 12)
plt.xticks(rotation = 'vertical')
plt.show()

#The distribution is right skewed. 
#There are some good drops in between (5 to 6, 9 to 10, 12 to 13, 17 to 18). 

    #Now lets see how price change with respect to floor
    
grouped_df = train_df.groupby('floor')['price_doc'].aggregate(np.median).reset_index()
plt.figure(figsize=(12,8))
sns.pointplot(grouped_df.floor.values, grouped_df.price_doc.values, alpha=0.8)
plt.ylabel('Median Price', fontsize=12)
plt.xlabel('Floor number', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()

#This shows an overall increasing trend 
#(individual houses seems to be costlier as well - check price of 0 floor houses). 
#A sudden increase in the house price is also observed at floor 18.


    #using Max Floors
    
plt.figure(figsize=(12,8))
sns.countplot(x="max_floor", data=train_df)
plt.ylabel('Count', fontsize=12)
plt.xlabel('Max floor number', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()

#We could see that there are few tall bars in between 
#(at 5,9,12,17 - similar to drop in floors in the previous graph). 
#May be there are some norms / restrictions on the number of maximum floors present(?).
#Now let us see how the median prices vary with the max floors.

plt.figure(figsize=(12,8))
sns.boxplot(x="max_floor", y="price_doc", data=train_df)
plt.ylabel('Median Price', fontsize=12)
plt.xlabel('Max Floor number', fontsize=12)
plt.xticks(rotation='vertical')
plt.show()

print(train_df)





