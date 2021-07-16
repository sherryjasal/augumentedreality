
# coding: utf-8

# In[137]:


import pandas as pd
import numpy as np
import warnings
import itertools
import math
import statsmodels.api as sm
import matplotlib.pyplot as plt   #importing libraries
import warnings
import itertools
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from pandas import DataFrame
from pandas import Series
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
plt.style.use('fivethirtyeight')


# In[138]:


def parser(x):
    return datetime.strptime('2017'+x, '%d/%m/%Y') #parsing date to convert to datetime variable

views_show = pd.read_csv("C:/Users/hp/Desktop/Media Project/mediacompany.csv",parse_dates=['Date'])

views_show.to_csv("C:/Users/hp/Desktop/Media Project/mediacompany_py.csv")


# In[139]:


views_show.head()
views_show.dtypes       

#converted using a copy of original file
#to know the dtypes of each variable
                         


# In[140]:


print(views_show.head())
series = pd.Series(views_show['Views_show'], index=views_show.index)
series.plot()
temp = views_show[['Date','Views_show']]  #checking views_shows against dates
temp.plot('Date','Views_show')
pyplot.show()


# In[141]:


views_show.describe()


# In[142]:


p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 7) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


# In[143]:


views_show[['Views_platform','Ad_impression']].describe()


# In[144]:


views_show.describe()
model1=sm.OLS(endog=views_show['Views_show'],exog=views_show[['Views_platform','Ad_impression']])
results1=model1.fit()
print(results1.summary())  #regression analysis(extra step)


# In[145]:


for param in pdq:
    for param_seasonal in seasonal_pdq:
        mod = sm.tsa.statespace.SARIMAX(endog=views_show['Views_show'],exog=views_show[['Visitors','Views_platform','Ad_impression','Cricket_match_india','Character_A']],
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

        results = mod.fit()
        #print(results.summary())

        residuals = DataFrame(results.resid)
#residuals.plot()
#pyplot.show()
#residuals.plot(kind='kde')
#pyplot.show()
        print("mean")
        print(math.sqrt(((residuals)**2).mean()))
        print("std")
        print((residuals).std())
        print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        print("")


# In[146]:


#warnings.filterwarnings("ignore") # specify to ignore warning messages

for param in pdq:
    for param_seasonal in seasonal_pdq:
        mod = sm.tsa.statespace.SARIMAX(endog=views_show['Views_show'],exog=views_show[['Views_platform','Ad_impression']],
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

        results = mod.fit()

        print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))


# In[163]:


for param in pdq:
    for param_seasonal in seasonal_pdq:
        mod = sm.tsa.statespace.SARIMAX(endog=views_show['Views_show'],exog=views_show['Character_A'],
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

        results = mod.fit()

        print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        print("")


# In[147]:


#residual mean   -2204.935758
#ARIMA(1, 1, 1)x(0, 1, 1, 7)12 - AIC:1524.7272830148054

mod = sm.tsa.statespace.SARIMAX(endog=views_show['Views_show'],exog=views_show[['Visitors','Views_platform','Ad_impression','Cricket_match_india','Character_A']],
                                            order=(0,1,1),
                                            seasonal_order=(0,1,1,7),
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
results = mod.fit()
print(results.summary())
residuals = DataFrame(results.resid)
residuals.plot()
print(math.sqrt(((residuals)**2).mean()))
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print((residuals).describe())


# In[158]:


series.shift(1) 

