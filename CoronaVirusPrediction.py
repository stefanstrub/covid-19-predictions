# %%
"""
# Prediction of the corona virus

This notebook contains some basic predictions of the corona virus.
"""

# %%
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
import datetime 
figsize = (12, 9)

# %%
import pandas as pd
import glob

# %%
def sigmoid(x, x0, k, L):
     y = L / (1 + np.exp(-k*(x-x0)))
     return y

def exp(x):
    return np.exp(x)

# %%
"""
Save confirmed cases and deaths into data frames
"""

# %%
dataframes = []
for f in glob.glob("time_series_covid19_confirmed_global.csv"):
    dataframes.append(pd.read_csv(f))
df = pd.concat(dataframes)

df = df.drop(columns=['Lat', 'Long', 'Province/State'])
df = df.T
df = df.rename(columns=df.iloc[0])
df = df.drop('Country/Region')
df['date'] = df.index

dataframes = []
for f in glob.glob("time_series_covid19_deaths_global.csv"):
    dataframes.append(pd.read_csv(f))
df_deaths = pd.concat(dataframes)

df_deaths = df_deaths.drop(columns=['Lat', 'Long', 'Province/State'])
df_deaths = df_deaths.T
df_deaths = df_deaths.rename(columns=df_deaths.iloc[0])
df_deaths = df_deaths.drop('Country/Region')
df_deaths['date'] = df_deaths.index

# %%
"""
# Prediction Growth
"""

# %%
def predict(country, prediction_length = 30):
    
    # sum provinces into one country
    try:
        sum_doubles = df[country].sum(1)
    except:
        sum_doubles = df[country]

    confirmed = sum_doubles.to_numpy()
    
    growth = confirmed[1:]-confirmed[:-1]
    average_length = 7 # in days
    growth_factor = np.zeros(len(growth))
    for i in range(len(growth)-1):
        if growth[i] == 0:
            grwoth_factor = 0
        else:
            growth_factor[i+1] = growth[i+1]/growth[i]
    growth_factor_averaged = np.zeros(len(growth_factor))
    for i in range(len(growth_factor)-average_length):
        growth_factor_averaged[i+average_length] = np.mean(growth_factor[i:i+average_length])
    growth_averaged = np.zeros(len(growth))
    for i in range(len(growth)-average_length):
        growth_averaged[i+average_length] = np.mean(growth[i:i+average_length])

    #Prediction
    fit_length = 7

    #compare different fits - curve fit with exp seams to work well
    #growth_averaged_p, residuals, rank, singular_values, rcond = np.polyfit(np.arange(0,fit_length),growth_averaged[-fit_length:],1, full=True)
    #growth_averaged_p_exp, residuals_exp, rank, singular_values, rcond = np.polyfit(np.arange(0,fit_length),np.log(growth_averaged[-fit_length:]),1, full=True)
    #residuals_exp = np.sum((np.exp(np.polyval(growth_averaged_p_exp, np.arange(0,fit_length))) - growth_averaged[-fit_length:])**2)
    #print('residuals',residuals,'residuals exp',residuals_exp, 'difference', (residuals-residuals_exp)/residuals)
    exp_function = lambda t,a,b: a*np.exp(b*t)
    growth_averaged_p_exp = curve_fit(exp_function, np.arange(0,fit_length), growth_averaged[-fit_length:],bounds=([-10**10,-1], [10**10, 0.2]))[0]

    
    #growth_averaged_fit = np.poly1d(growth_averaged_p)
    #growth_averaged_prediction = growth_averaged_fit(np.arange(0,prediction_length+fit_length))
    #growth_averaged_fit = np.poly1d(growth_averaged_p_exp)
    #growth_averaged_prediction = np.exp(growth_averaged_fit(np.arange(0,prediction_length+fit_length)))
    growth_averaged_prediction = exp_function(np.arange(0,prediction_length+fit_length),growth_averaged_p_exp[0],growth_averaged_p_exp[1])
    for i in range(len(growth_averaged_prediction)-fit_length):
        if growth_averaged_prediction[i+fit_length] < growth_averaged_prediction[0+fit_length]/10:
            growth_averaged_prediction[i+fit_length] = growth_averaged_prediction[0+fit_length]/10
    confirmed_predicted = np.zeros(len(growth_averaged_prediction)-fit_length)
    for i in range(prediction_length):
        if i == 0:
            confirmed_predicted[i] = confirmed[-1] + growth_averaged_prediction[i+fit_length]
        else:
            confirmed_predicted[i] = confirmed_predicted[i-1] + growth_averaged_prediction[i+fit_length]
            
    confirmed_predicted = confirmed_predicted.astype(int)
    
    return confirmed_predicted, confirmed

# %%

countries = df.columns
countries = countries.drop_duplicates() 
countries = countries[:-1] #drop date column
#countries = countries[:2]
date = df['date'].to_numpy()
today = datetime.date.today()
x_date = []
length = 150
for i in range(length):
    x_date.append(datetime.datetime(2020, 1, 22)+ datetime.timedelta(days=i) )

prediction_length = 30
confirmed_and_predicted = np.zeros((len(countries),len(date)+prediction_length))

for i in range(len(countries)): 
    country = countries[i]
    confirmed_predicted, confirmed = predict(country, prediction_length)
    for j in range(len(date)):
        confirmed_and_predicted[i,j] = confirmed[j]
    for j in range(prediction_length):
        confirmed_and_predicted[i,j+len(date)] = confirmed_predicted[j]
        
    #print(country,'t+2',confirmed_predicted[1],'t+7',confirmed_predicted[6],'t+30',confirmed_predicted[29])

    """"
    fig = plt.figure()
    plt.plot(x_date[:len(date)],confirmed,'.')
    #plt.plot(x_date,y, label='fit')
    plt.plot(x_date[len(date):len(date)+prediction_length],confirmed_predicted,'.', label='prediction')
    plt.xlabel('date')
    plt.xticks( rotation='vertical')
    plt.ylabel('confirmed')
    plt.title(country)   
    plt.show()"""
    
#create df of confirmed and predicted cases
confirmed_and_predicted = pd.DataFrame(data=confirmed_and_predicted.T,columns=countries.to_numpy())  

"""
fig = plt.figure()
plt.plot(x_date[:len(date)],confirmed,'.')
#plt.plot(x_date,y, label='fit')
plt.plot(x_date[len(date):len(date)+prediction_length],confirmed_predicted,'.', label='prediction')
plt.xlabel('date')
plt.xticks( rotation='vertical')
plt.ylabel('confirmed')
plt.title(country)   
plt.show()

fig = plt.figure()
plt.plot(date[1:],growth,'.')
plt.xticks(locs[::every_nth])
plt.xlabel('date')
plt.ylabel('growth')
plt.title(country)

fig = plt.figure()
plt.plot(confirmed[1:],growth_averaged,'.')
plt.xscale("log")
plt.yscale("log")
plt.xlabel('confirmed')
plt.ylabel('growth averaged past '+ str(average_length)+' days')
plt.title(country)

fig, ax = plt.subplots()
plt.plot(x_date[1:len(date)],growth_averaged,'o')
plt.plot(x_date[len(date)-fit_length:len(date)+prediction_length],growth_averaged_prediction,'.')
#plt.plot(x_date[len(date)-fit_length:len(date)],np.exp(np.polyval(growth_averaged_p_exp, np.arange(0,fit_length))),'.')
every_nth = 10
locs, labels = plt.xticks()
plt.xticks(locs[::every_nth])
plt.xlabel('date')
plt.ylabel('growth averaged past '+ str(average_length)+' days')
plt.title(country)

fig, ax = plt.subplots()
plt.plot(date[1:],growth_factor,'.')
every_nth = 10
locs, labels = plt.xticks()
plt.axhline(1, color='r')
plt.xticks(locs[::every_nth])
plt.xlabel('date')
plt.ylabel('growth factor')
plt.title(country)

fig, ax = plt.subplots()
plt.plot(date[1:],growth_factor_averaged,'.')
every_nth = 10
locs, labels = plt.xticks()
plt.axhline(1, color='r')
plt.xticks(locs[::every_nth])
plt.xlabel('date')
plt.ylabel('growth factor averaged past '+ str(average_length)+' days')
plt.title(country)
"""

# %%
"""
# Predict Deaths
"""

# %%
def predict_deaths(country, prediction_length = 30):
    
    # sum provinces into one country
    try:
        sum_doubles = df_deaths[country].sum(1)
    except:
        sum_doubles = df_deaths[country]

    deaths = sum_doubles.to_numpy()
    confirmed_and_predicted_array = confirmed_and_predicted[country].to_numpy()
    growth = confirmed_and_predicted_array[1:]-confirmed_and_predicted_array[:-1]
    index_of_today = len(df['date'])
    
    death_rate = 0.09
    death_latency = 12 # days
    
    new_deaths_predicted = growth[index_of_today-death_latency:]*death_rate
    
    deaths_predicted = np.zeros(prediction_length)
    for i in range(prediction_length):
        if i == 0:
            deaths_predicted[i] = deaths[-1] + new_deaths_predicted[i]
        else:
            deaths_predicted[i] = deaths_predicted[i-1] + new_deaths_predicted[i]

    deaths_predicted = deaths_predicted.astype(int)
    return deaths_predicted, deaths

# %%
deaths_and_predicted = np.zeros((len(countries),len(date)+prediction_length))
for i in range(len(countries)): 
    country = countries[i]
    deaths_predicted, deaths = predict_deaths(country, prediction_length)
    for j in range(len(date)):
        deaths_and_predicted[i,j] = deaths[j]
    for j in range(prediction_length):
        deaths_and_predicted[i,j+len(date)] = deaths_predicted[j]

deaths_and_predicted = pd.DataFrame(data=deaths_and_predicted.T,columns=countries.to_numpy()) 
print(deaths_and_predicted)

# %%
"""
# Save to desired csv format
"""

# %%
confirmed_and_predicted = confirmed_and_predicted.T
confirmed_and_predicted['Country'] = confirmed_and_predicted.index
confirmed_and_predicted = confirmed_and_predicted.reset_index()
confirmed_and_predicted = confirmed_and_predicted.drop('index',1)
# get country as the first column
cols = confirmed_and_predicted.columns.tolist()
cols = cols[-1:] + cols[:-1]
confirmed_and_predicted = confirmed_and_predicted[cols]

deaths_and_predicted = deaths_and_predicted.T
deaths_and_predicted['Country'] = deaths_and_predicted.index
deaths_and_predicted = deaths_and_predicted.reset_index()
deaths_and_predicted = deaths_and_predicted.drop('index',1)
# get country as the first column
cols = deaths_and_predicted.columns.tolist()
cols = cols[-1:] + cols[:-1]
deaths_and_predicted = deaths_and_predicted[cols]

# %%
"""
Change Taiwan* to Taiwan
"""

# %%
as_list = countries.tolist()
idx = as_list.index('Taiwan*')
as_list[idx] = 'Taiwan'
countries = as_list

# %%
"""
Change x_date from datetime to string
"""

# %%
x_date = []
for i in range(length):
    x_date.append(datetime.datetime(2020, 1, 22)+ datetime.timedelta(days=i) )
for i in range(len(x_date)):    
    x_date[i] = x_date[i].strftime('%Y-%m-%d')
x_date = np.asarray(x_date)

# t2
n = 2
path2 = r'C:\Users\stefa\OneDrive\Documents\Machine Learning\Covid-19 predictions\predictions'+r'\2day_prediction_'+x_date[len(df['date'])+n]+'.csv'
t2_prediction = pd.DataFrame(data={'Country': countries,'Target/Date': x_date[len(df['date'])+n],'N': confirmed_and_predicted.iloc[:,len(date)+n], 'D': deaths_and_predicted.iloc[:,len(date)+n]})
print(t2_prediction)
t2_prediction.to_csv(path2, index=False)
# t7
n = 7
path7 = r'C:\Users\stefa\OneDrive\Documents\Machine Learning\Covid-19 predictions\predictions'+r'\7day_prediction_'+x_date[len(df['date'])+n]+'.csv'
t7_prediction = pd.DataFrame(data={'Country': countries,'Target/Date': x_date[len(df['date'])+n],'N': confirmed_and_predicted.iloc[:,len(date)+n], 'D': deaths_and_predicted.iloc[:,len(date)+n]})
print(t7_prediction)
t7_prediction.to_csv(path7, index=False)
# t30
n = 30
path30 = r'C:\Users\stefa\OneDrive\Documents\Machine Learning\Covid-19 predictions\predictions'+r'\30day_prediction_'+x_date[len(df['date'])+n]+'.csv'
t30_prediction = pd.DataFrame(data={'Country': countries,'Target/Date': x_date[len(df['date'])+n],'N': confirmed_and_predicted.iloc[:,len(date)+n], 'D': deaths_and_predicted.iloc[:,len(date)+n]})
print(t30_prediction)
t30_prediction.to_csv(path30, index=False)

# %%
