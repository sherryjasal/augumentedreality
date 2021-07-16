
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from pandas import DataFrame
from pandas import Series
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
from datetime import date, datetime, timedelta


# In[10]:


campaign = pd.read_csv("C:/Users/hp/Desktop/Media Project/Campaign.csv",parse_dates=['Start Date', 'End Date'])


# In[11]:


campaign.describe()
campaign.dtypes


# In[13]:


campaign=campaign.sort_values(by=['Start Date'])


# In[56]:


from datetime import date, datetime, timedelta

def perdelta(start, end, delta):
    curr = start
    while curr < end:
        yield curr
        curr += delta
mydatelist=[result for result in perdelta(date(2018, 1, 1), date(2019, 1, 1), timedelta(days=1))]


# In[57]:


df=pd.DataFrame(mydatelist,columns =['myDate'])
df.describe()


# In[58]:


df['myDate'] = pd.to_datetime(df['myDate'])


# In[59]:


df.describe()


# In[60]:


df.sort_values


# In[61]:


df.head()


# In[79]:


df["Num_C"] = 0


# In[80]:


df.describe()


# In[83]:


print(campaign.iloc[0 , 2])


# In[90]:


df["Num_C"] = 0
for i in range(0,365):
    for j in range(0,74) :
        if campaign.iloc[j , 1] <= df.iloc[i , 0] <= campaign.iloc[j , 2]:
            df.iloc[i ,1] = df.iloc[i , 1]+1
print(df)            


# In[253]:


iterate_month = pd.read_csv("C:/Users/hp/Desktop/Media Project/Campaign.csv",parse_dates=['Start Date', 'End Date'])

def iterate_months(start_ym, end_ym):
    for ym in range(int(start_ym), int(end_ym) + 1):
        if ym % 100 > 12 or ym % 100 == 0:
            continue
        yield str(ym)

list(iterate_months('201801', '201812'))


# In[254]:


iterate_month.describe()
iterate_month.dtypes


# In[258]:


iterate_month=iterate_month.sort_values(by=['Start Date'])
df=pd.DataFrame(iterate_month,columns =['Date'])

df.describe()


# In[259]:


iterate_month=iterate_month.sort_values(by=['Start Date'])


# In[260]:


df['Date'] = pd.to_datetime(df['Date'])


# In[252]:


df["Num_C"] = 0
for j in range(0,13):
    for i in range(0,74) :
        if iterate_month.iloc[j , 1] <= df.iloc[i , 0] <= iterate_month.iloc[j , 2]:
            df.iloc[i ,1] = df.iloc[i , 1]+1
print(df)


# In[191]:


#from datetime import *

#def iterate_months(start_ym, end_ym):
    #for ym in range(int(start_ym), int(end_ym) + 1):
        #iterate_month=[i[0] for i in iterate_month]
        #interval=[b-a for a,b in zip(start_ym[:-1], end_ym[1:])]
        #sorted_interval=[item for item in interval if item!=timedelta(0)]
  # It doesn't function and just give wrong information
        #interval_concatenated=[ko[i]+ko[i+1] for i in range(len(sorted_interval)-1)]
    #print(interval_concatenated)#


# In[263]:


df["Num_C"] = 0
for i in range(0,12):
    for j in range(0,74) :
        if iterate_month.iloc[i , 1] <= df.iloc[j , 0] <= iterate_month.iloc[i , 2]:
            df.iloc[i ,1] = df.iloc[i , 1]+1
print(df)

