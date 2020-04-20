#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 10:01:21 2020

@author: loebn
"""


import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
import datetime 
figsize = (12, 9)

import pandas as pd
import glob

def sigmoid(x, x0, k, L):
     y = L / (1 + np.exp(-k*(x-x0)))
     return y

def exp(x):
    return np.exp(x)


class Data:
    def __init__(self, confirmedFile = None, deathsFile = None,recoveredFile = None):
        self.confirmedFile = str(confirmedFile)
        self.deathsFile = str(deathsFile)
        self.recoveredFile = str(recoveredFile)
        #confirmed cases.
        self.df = self.initialize(self.confirmedFile)
        #death cases.
        self.df_deaths = self.initialize(self.deathsFile)
        #recovered cases.
        self.df_recovered = self.initialize(self.recoveredFile)
        
        #set the countries.
        self.countries = self.setCountries()
        
        #set the dates.
        self.length = 150
        self.date, self.today, self.x_date = self.setDates()
        self.x_date_str = self.setDatesToString()
    
    def setDatesToString(self):
        self.x_date_str = []
        for i in range(self.length):
            self.x_date_str.append(datetime.datetime(2020, 1, 22)+ datetime.timedelta(days=i) )
        for i in range(len(self.x_date)):    
            self.x_date_str[i] = self.x_date[i].strftime('%Y-%m-%d')
        self.x_date_str = np.asarray(self.x_date_str)
        return self.x_date_str
    
    def setDates(self): 
        self.date = self.df['date'].to_numpy()
        self.today = datetime.date.today()
        self.x_date = []
        
        for i in range(self.length):
            self.x_date.append(datetime.datetime(2020, 1, 22)+ datetime.timedelta(days=i) )
        return self.date, self.today, self.x_date
        
    def setCountries(self):
        countries = self.df.columns
        countries = countries.drop_duplicates() 
        self.countries = countries[:-1] #drop date column
        return self.countries
    
    def initialize(self,File = None):
        dataframes = []
        for f in glob.glob(str(File)):
            dataframes.append(pd.read_csv(f))
        df = pd.concat(dataframes)
        df = df.drop(columns=['Lat', 'Long', 'Province/State'])
        df = df.T
        df = df.rename(columns=df.iloc[0])
        df = df.drop('Country/Region')
        df['date'] = df.index
        return df
    
    def saveInDesiredFormat(self, confirmed_and_predicted, deaths_and_predicted, recovered_and_predicted, path2, path7, path30):
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
        
        recovered_and_predicted = recovered_and_predicted.T
        recovered_and_predicted['Country'] = recovered_and_predicted.index
        recovered_and_predicted = recovered_and_predicted.reset_index()
        recovered_and_predicted = recovered_and_predicted.drop('index',1)
        # get country as the first column
        cols = recovered_and_predicted.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        recovered_and_predicted = recovered_and_predicted[cols]
        
        
        as_list = self.countries.tolist()
        idx = as_list.index('Taiwan*')
        as_list[idx] = 'Taiwan'
        self.countries = as_list
        
        
        
        print(self.df['date'])
        self.x_date = []
        for i in range(self.length):
            self.x_date.append(datetime.datetime(2020, 1, 22)+ datetime.timedelta(days=i) )
        for i in range(len(self.x_date)):    
            self.x_date[i] = self.x_date[i].strftime('%Y-%m-%d')
        self.x_date = np.asarray(self.x_date)
        # t2
        n = 2
        ###################################path2 = r'C:\Users\stefa\OneDrive\Documents\Machine Learning\Covid-19 predictions\predictions'+r'\2day_prediction_'+x_date[len(df['date'])+n]+'.csv'
        t2_prediction = pd.DataFrame(data={'Province/State': np.nan, 'Country': self.countries,'Target/Date': self.x_date[len(self.df['date'])+n],'N': confirmed_and_predicted.iloc[:,len(self.date)+n+1], 'D': deaths_and_predicted.iloc[:,len(self.date)+n+1], 'R': recovered_and_predicted.iloc[:,len(self.date)+n+1]})
        print(self.x_date[len(self.df['date'])])
        print(len(self.date))
        print(t2_prediction)
        t2_prediction.to_csv(path2, index=False)
        # t7
        n = 7
        ###################################path7 = r'C:\Users\stefa\OneDrive\Documents\Machine Learning\Covid-19 predictions\predictions'+r'\7day_prediction_'+x_date[len(df['date'])+n]+'.csv'
        t7_prediction = pd.DataFrame(data={'Province/State': np.nan, 'Country': self.countries,'Target/Date': self.x_date[len(self.df['date'])+n],'N': confirmed_and_predicted.iloc[:,len(self.date)+n+1], 'D': deaths_and_predicted.iloc[:,len(self.date)+n+1], 'R': recovered_and_predicted.iloc[:,len(self.date)+n+1]})
        print(t7_prediction)
        t7_prediction.to_csv(path7, index=False)
        # t30
        n = 30
        ###################################path30 = r'C:\Users\stefa\OneDrive\Documents\Machine Learning\Covid-19 predictions\predictions'+r'\30day_prediction_'+x_date[len(df['date'])+n]+'.csv'
        t30_prediction = pd.DataFrame(data={'Province/State': np.nan,'Country': self.countries,'Target/Date': self.x_date[len(self.df['date'])+n],'N': confirmed_and_predicted.iloc[:,len(self.date)+n+1], 'D': deaths_and_predicted.iloc[:,len(self.date)+n+1], 'R': recovered_and_predicted.iloc[:,len(self.date)+n+1]})
        print(t30_prediction)
        t30_prediction.to_csv(path30, index=False)
        
        
        
        
        
        
        
        
    
    
class Prediction(Data):
    def __init__(self,confirmedFile, deathsFile, recoveredFile):
        super(Prediction, self).__init__(confirmedFile, deathsFile, recoveredFile)
        
        
        
        
        #hardcoded stuff.
        self.fit_length = 5
        self.center_average_length = 7
        self.average_length = 7 # in days
        # Set the noise level.
        self.noise_level=2
        # Test time shifts in days.
        self.delta=np.arange(1.0,20.0,1)
        # Test mortailities.
        self.mortalities=np.arange(0.001,0.5,0.001)
        
        
        
        
        #variable stuff.
        self.prediction_length = None
        
        self.confirmed = None
        self.confirmed_predicted = None
        self.confirmed_and_predicted = None
        self.new_cases_manipulated_summed = None
        self.new_cases_averaged_p_exp = None
        
        self.deaths = None
        self.new_deaths = None
        self.new_deaths_averaged = None
        self.deaths_predicted = None
        self.deaths_and_predicted = None
        self.death_latency = None
        self.mortality = None
        
        self.recovered = None
        self.new_recovered = None
        self.new_recovered_averaged = None
        self.recovered_predicted = None
        
        self.new_cases = None
        self.new_cases_averaged = None
        
        self.growth_factor = None
        self.growth_factor_averaged = None
        
        self.growth_factor_recovered = None
        self.growth_factor_recovered_averaged = None
        
    def setUpEverything(self, country, prediction_length = 31):
        self.prediction_length = prediction_length+int((self.center_average_length+1)/2)

        
        # sum provinces into one country
        try:
            sum_doubles = self.df[country].sum(1)
        except:
            sum_doubles = self.df[country]
    
        self.confirmed = sum_doubles.to_numpy()

        self.setGrowthFactor()
        self.setNewCasesAveraged()
    
    def setUpRecovered(self, country, prediction_length = 31):
        self.prediction_length = prediction_length+int((self.center_average_length+1)/2)

        
        # sum provinces into one country
        try:
            sum_doubles = self.df_recovered[country].sum(1)
        except:
            sum_doubles = self.df_recovered[country]
    
        self.recovered = sum_doubles.to_numpy()
        
        self.new_recovered = self.recovered[1:]-self.recovered[:-1]
        self.growth_factor_recovered = np.zeros(len(self.new_recovered))
        for i in range(len(self.new_recovered)-1):
            if self.new_recovered[i] == 0:
                self.growth_factor_recovered[i+1] = 0
            else:
                self.growth_factor_recovered[i+1] = self.new_recovered[i+1]/self.new_recovered[i]
        self.growth_factor_recovered_averaged = np.zeros(len(self.growth_factor_recovered))
        for i in range(len(self.growth_factor)-self.average_length):
            self.growth_factor_recovered_averaged[i+self.average_length] = np.mean(self.growth_factor_recovered[i:i+self.average_length])
        
        self.new_recovered_averaged = np.zeros(len(self.new_recovered))
        for i in range(len(self.new_recovered)-self.average_length):
            self.new_recovered_averaged[i+self.average_length] = np.mean(self.new_recovered[i:i+self.average_length])
    
        
    def setUpDeaths(self, country):
         # sum provinces into one country
        try:
            sum_doubles = self.df_deaths[country].sum(1)
        except:
            sum_doubles = self.df_deaths[country]
            
        self.deaths = sum_doubles.to_numpy()
        
    def setGrowthFactor(self):
        self.new_cases = self.confirmed[1:]-self.confirmed[:-1]
        self.growth_factor = np.zeros(len(self.new_cases))
        for i in range(len(self.new_cases)-1):
            if self.new_cases[i] == 0:
                self.growth_factor[i+1] = 0
            else:
                self.growth_factor[i+1] = self.new_cases[i+1]/self.new_cases[i]
        self.growth_factor_averaged = np.zeros(len(self.growth_factor))
        for i in range(len(self.growth_factor)-self.average_length):
            self.growth_factor_averaged[i+self.average_length] = np.mean(self.growth_factor[i:i+self.average_length])
    
    def setNewCasesAveraged(self):
        self.new_cases_averaged = np.zeros(len(self.new_cases))
        for i in range(len(self.new_cases)-self.average_length):
            self.new_cases_averaged[i+self.average_length] = np.mean(self.new_cases[i:i+self.average_length])
    
# =============================================================================
#     def exp_function(self):
#         return lambda t,a,b: a*np.exp(b*t)
# =============================================================================
    
    def curveFit(self):
        exp_function = lambda t,a,b: a*np.exp(b*t)
        self.new_cases_averaged_p_exp = curve_fit(exp_function, np.arange(0,self.fit_length), self.new_cases_averaged[-self.fit_length:],bounds=([-10**10,-1], [10**10, 0.2]))[0]
    
    def predict(self, country, prediction_length = 30):
        self.setUpEverything(country, prediction_length)
        self.curveFit()
        exp_function = lambda t,a,b: a*np.exp(b*t)
        new_cases_averaged_prediction = exp_function(np.arange(0,self.prediction_length+self.fit_length),self.new_cases_averaged_p_exp[0],self.new_cases_averaged_p_exp[1])
        
        
        
        for i in range(len(new_cases_averaged_prediction)-self.fit_length):
            if new_cases_averaged_prediction[i+self.fit_length] < new_cases_averaged_prediction[0+self.fit_length]/10:
                new_cases_averaged_prediction[i+self.fit_length] = new_cases_averaged_prediction[0+self.fit_length]/10
              
        new_cases_and_averaged_prediction = np.zeros(len(self.new_cases_averaged)+self.prediction_length)
        new_cases_and_averaged_prediction[:len(self.new_cases_averaged)] = self.new_cases_averaged
        new_cases_and_averaged_prediction[len(self.new_cases_averaged):] = new_cases_averaged_prediction[-self.prediction_length:]
        
        #calculate backwards from the average to daily numbers
        new_cases_and_prediction = np.zeros(len(self.new_cases)+self.prediction_length)
        
        for i in range(len(new_cases_and_prediction)-self.average_length):
            new_cases_and_prediction[i+self.average_length] = new_cases_and_averaged_prediction[i+self.average_length]*self.average_length-np.sum(new_cases_and_prediction[i+1:i+self.average_length])
    
        #comupte a center average of the new cases and the predicted new cases
        new_cases_and_prediction_center_averaged = np.zeros(len(new_cases_and_prediction)-int((self.center_average_length-1)/2)-1)
        for i in range(len(new_cases_and_prediction)-self.center_average_length):
            new_cases_and_prediction_center_averaged[i+int((self.center_average_length-1)/2)] = np.mean(new_cases_and_prediction[i:i+self.center_average_length])
             
        # sum the predicted new cases center averaged numbers to the confirmed predicted numbers      
        confirmed_predicted = np.zeros(len(new_cases_and_prediction_center_averaged)-self.fit_length)
        for i in range(self.prediction_length-int((self.center_average_length-1)/2)-1):
            if i == 0:
                confirmed_predicted[i] = self.confirmed[-1] + new_cases_and_prediction_center_averaged[i+len(self.date)-1]
            else:
                confirmed_predicted[i] = confirmed_predicted[i-1] + new_cases_and_prediction_center_averaged[i+len(self.date)-1]
                
        self.confirmed_predicted = confirmed_predicted.astype(int)
    
        """
        fig, ax = plt.subplots()
        plt.plot(x_date[1:len(date)],new_cases,'o')
        plt.plot(x_date[1:len(date)],new_cases_averaged,'o')
        plt.plot(x_date[len(date)-fit_length:len(date)+prediction_length],new_cases_averaged_prediction,'r.')
        plt.plot(x_date[0:len(new_cases_and_prediction)],new_cases_and_prediction,'.')
        #plt.plot(x_date[len(date)-fit_length:len(date)],np.exp(np.polyval(new_cases_averaged_p_exp, np.arange(0,fit_length))),'.')
        plt.xlabel('date')
        plt.ylabel('new cases averaged past '+ str(average_length)+' days')
        plt.title(country)
        plt.xticks(rotation='vertical')
        plt.show()"""
        
        return self.confirmed_predicted, self.confirmed
    
    def makePrediction(self, prediction_length = 31):
        self.prediction_length = int(prediction_length)
        confirmed_and_predicted = np.zeros((len(self.countries),len(self.date)+self.prediction_length))

        for i in range(len(self.countries)): 
            country = self.countries[i]
            #print("xxxxxxxx"+ str(self.prediction_length))
            self.confirmed_predicted, self.confirmed = self.predict(country, prediction_length)
            for j in range(len(self.date)):
                confirmed_and_predicted[i,j] = self.confirmed[j]
            for j in range(prediction_length):
                confirmed_and_predicted[i,j+len(self.date)] = self.confirmed_predicted[j]
        #create df of confirmed and predicted cases
        self.confirmed_and_predicted = pd.DataFrame(data=confirmed_and_predicted.T,columns=self.countries.to_numpy())
        return self.confirmed_and_predicted
        
    # Define the prior in data space. ----------------------------------
    def prior_data(self, s1, s2, noise_level):
        p=np.sum((s1-s2)**2)/(len(s1)*noise_level**2)
        return np.exp(-p/2.0)  
        
    def findDeathLatencyAndMortality(self, country):
        self.setUpDeaths(country)
   
        confirmed_and_predicted_array = self.confirmed_and_predicted[country].to_numpy()
        self.new_cases = confirmed_and_predicted_array[1:]-confirmed_and_predicted_array[:-1]
        self.new_cases_averaged = np.zeros(len(self.new_cases)-int((self.average_length-1)/2)-1) # here take center average instead of forward averaging
        for i in range(len(self.new_cases)-self.average_length):
            self.new_cases_averaged[i+int((self.average_length-1)/2)] = np.mean(self.new_cases[i:i+self.average_length])
            
        self.new_deaths = self.deaths[1:]-self.deaths[:-1]
        self.new_deaths_averaged = np.zeros(len(self.new_deaths)-int((self.average_length-1)/2)-1) # here take center average instead of forward averaging
        for i in range(len(self.new_deaths)-self.average_length):
            self.new_deaths_averaged[i+int((self.average_length-1)/2)] = np.mean(self.new_deaths[i:i+self.average_length])
                
        # Use grid search to find death latency and mortality
        # March through all possible time shifts. --------------------------
        
        
        
    
        # Initialise posterior distribution.
        p=np.zeros((len(self.delta),len(self.mortalities)))
        
    
        # March through all possible time shifts.
        for n in range(len(self.delta)):
            death_latency = int(self.delta[n])
            # March through all possible time shifts.
            for m in range(len(self.mortalities)):
                mortality = self.mortalities[m]
                # Make test time series by shifting new_cases_averaged.
                s = np.zeros(len(self.new_deaths_averaged))
                s[death_latency:] = self.new_cases_averaged[:len(self.new_deaths_averaged)-death_latency]*mortality
                """
                fig = plt.figure()
                plt.plot(x_date[1:len(s)+1],s,'.',label='s')
                plt.plot(x_date[1:len(new_deaths_averaged)+1],new_deaths_averaged,'.',label='new deaths averaged'+str(n)+str(mortality))
                plt.xlabel('date')
                plt.ylabel('new deaths')
                plt.xticks(rotation='vertical')
                plt.grid(True)
                plt.legend()
                plt.title(country)
                plt.show() """
    
                # Evaluate posterior.
                p[n,m]=self.prior_data(s,self.new_deaths_averaged,self.noise_level)
    
    
        # Normalise distribution.
        #p=p/(0.01*np.sum(p))
    
        # Plot posterior. --------------------------------------------------
        
        plt.pcolor(self.mortalities,self.delta,p, cmap=plt.cm.get_cmap('binary'))
        plt.xlabel(r'mortalities')
        plt.ylabel(r'death latency')
        plt.colorbar()
        plt.title('posterior probability density')
        plt.grid()
        plt.show()
        
        death_latency_index, mortality_index = np.where(p == np.amax(p))
        self.death_latency, self.mortality = int(self.delta[death_latency_index]), self.mortalities[mortality_index]
        print(self.death_latency, self.mortality)
        print(self.deaths[-1])
        print(self.death_latency)
        print(self.prediction_length)
        print(confirmed_and_predicted_array[-self.death_latency-self.prediction_length])
        print(self.deaths[-1]/confirmed_and_predicted_array[-self.death_latency-self.prediction_length])
        
        return self.death_latency, self.mortality    
    
    def predict_deaths(self, country, prediction_length = 31, death_latency=9, mortality=0.054):
        self.setUpDeaths(country)
        
        self.prediction_length = prediction_length
        self.death_latency = death_latency
        self.mortality = mortality
        
        
        confirmed_and_predicted_array = self.confirmed_and_predicted[country].to_numpy()
        self.new_cases = confirmed_and_predicted_array[1:]-confirmed_and_predicted_array[:-1]
        self.new_cases_averaged = np.zeros(len(self.new_cases)-int((self.average_length-1)/2)-1) # here take center average instead of forward averaging
        for i in range(len(self.new_cases)-self.average_length):
            self.new_cases_averaged[i+int((self.average_length-1)/2)] = np.mean(self.new_cases[i:i+self.average_length])
            
        self.new_deaths = self.deaths[1:]-self.deaths[:-1]
        self.new_deaths_averaged = np.zeros(len(self.new_deaths)-int((self.average_length-1)/2)-1) # here take center average instead of forward averaging
        for i in range(len(self.new_deaths)-self.average_length):
            self.new_deaths_averaged[i+int((self.average_length-1)/2)] = np.mean(self.new_deaths[i:i+self.average_length])
            
        index_of_today = len(self.df['date'])
        
        new_deaths_predicted = self.new_cases[index_of_today-self.death_latency:]*self.mortality
        
        deaths_predicted = np.zeros(self.prediction_length)
        for i in range(self.prediction_length):
            if i == 0:
                deaths_predicted[i] = self.deaths[-1] + new_deaths_predicted[i]
            else:
                deaths_predicted[i] = deaths_predicted[i-1] + new_deaths_predicted[i]
                
        new_cases_manipulated = np.zeros(len(self.date)+self.prediction_length)
        new_cases_manipulated[self.death_latency:] = self.new_cases_averaged[:len(self.date)+self.prediction_length-self.death_latency]*self.mortality
        self.new_cases_manipulated_summed = np.zeros(len(self.date)+self.prediction_length)
        for i in range(len(new_cases_manipulated)-1):
            self.new_cases_manipulated_summed[i+1] = self.new_cases_manipulated_summed[i] + new_cases_manipulated[i+1]
    
        self.deaths_predicted = deaths_predicted.astype(int)
        """
        fig = plt.figure()
        plt.plot(x_date[1:len(new_deaths)+1],new_deaths,'.',label='new deaths')
        plt.plot(x_date[1:len(new_deaths_averaged)+1],new_deaths_averaged,'.',label='new deaths averaged')
        #plt.plot(x_date[1:len(new_cases_manipulated)+1],new_cases_manipulated,'.',label='new cases manipulated')
        plt.xlabel('date')
        plt.ylabel('new deaths')
        plt.xticks(rotation='vertical')
        plt.grid(True)
        plt.legend()
        plt.title(country)
        plt.show() 
        
        fig = plt.figure()
        plt.plot(x_date[1:len(new_cases)+1],new_cases,'.',label='new cases')
        plt.plot(x_date[1:len(new_cases_averaged)+1],new_cases_averaged,'.',label='new cases averaged')
        plt.xlabel('date')
        plt.ylabel('new cases')
        plt.xticks(rotation='vertical')
        plt.legend()
        plt.grid(True)
        plt.title(country)
        plt.show()
        plt.savefig('new_cases.pdf')"""
        
        return self.deaths_predicted, self.deaths, self.new_cases_manipulated_summed
    
    
    def makeDeathPrediction(self,prediction_length):
        deaths_and_predicted = np.zeros((len(self.countries),len(self.date)+self.prediction_length))
        self.death_latency, self.mortality = self.findDeathLatencyAndMortality('Switzerland')
        self.prediction_length = prediction_length
        for i in range(len(self.countries)): 
            country = self.countries[i]
            self.deaths_predicted, self.deaths, self.new_cases_manipulated_summed = self.predict_deaths(country, self.prediction_length, self.death_latency, self.mortality)
            for j in range(len(self.date)):
                deaths_and_predicted[i,j] = self.deaths[j]
            for j in range(self.prediction_length):
                deaths_and_predicted[i,j+len(self.date)] = self.deaths_predicted[j]
            """   
            fig = plt.figure()
            plt.plot(x_date[:len(date)],deaths_and_predicted[i][:len(date)],'.')
            plt.plot(x_date[len(date):len(deaths_and_predicted[i])],deaths_and_predicted[i][len(date):],'.')
            plt.plot(x_date[:len(new_cases_manipulated_summed)],new_cases_manipulated_summed,'.')
            plt.xlabel('date')
            plt.ylabel('deaths')
            plt.title(country)
            plt.show()"""
        
        self.deaths_and_predicted = pd.DataFrame(data=deaths_and_predicted.T,columns=self.countries.to_numpy())
        return self.deaths_and_predicted
    
    
    
    
    
    
    
    
    ########################
    def curveFitRecovered(self):
        exp_function = lambda t,a,b: a*np.exp(b*t)
        self.new_recovered_averaged_p_exp = curve_fit(exp_function, np.arange(0,self.fit_length), self.new_recovered_averaged[-self.fit_length:],bounds=([-10**10,-1], [10**10, 0.2]))[0]
    
    
    def predictRecovered(self, country, prediction_length = 31):
        self.setUpRecovered(country, prediction_length)
        self.curveFitRecovered()
        exp_function = lambda t,a,b: a*np.exp(b*t)
        new_recovered_averaged_prediction = exp_function(np.arange(0,self.prediction_length+self.fit_length),self.new_recovered_averaged_p_exp[0],self.new_recovered_averaged_p_exp[1])
        
        
        
        for i in range(len(new_recovered_averaged_prediction)-self.fit_length):
            if new_recovered_averaged_prediction[i+self.fit_length] < new_recovered_averaged_prediction[0+self.fit_length]/10:
                new_recovered_averaged_prediction[i+self.fit_length] = new_recovered_averaged_prediction[0+self.fit_length]/10
              
        new_recovered_and_averaged_prediction = np.zeros(len(self.new_recovered_averaged)+self.prediction_length)
        new_recovered_and_averaged_prediction[:len(self.new_recovered_averaged)] = self.new_recovered_averaged
        new_recovered_and_averaged_prediction[len(self.new_recovered_averaged):] = new_recovered_averaged_prediction[-self.prediction_length:]
        
        #calculate backwards from the average to daily numbers
        new_recovered_and_prediction = np.zeros(len(self.new_recovered)+self.prediction_length)
        
        for i in range(len(new_recovered_and_prediction)-self.average_length):
            new_recovered_and_prediction[i+self.average_length] = new_recovered_and_averaged_prediction[i+self.average_length]*self.average_length-np.sum(new_recovered_and_prediction[i+1:i+self.average_length])
    
        #comupte a center average of the new cases and the predicted new cases
        new_recovered_and_prediction_center_averaged = np.zeros(len(new_recovered_and_prediction)-int((self.center_average_length-1)/2)-1)
        for i in range(len(new_recovered_and_prediction)-self.center_average_length):
            new_recovered_and_prediction_center_averaged[i+int((self.center_average_length-1)/2)] = np.mean(new_recovered_and_prediction[i:i+self.center_average_length])
             
        # sum the predicted new cases center averaged numbers to the confirmed predicted numbers      
        recovered_predicted = np.zeros(len(new_recovered_and_prediction_center_averaged)-self.fit_length)
        for i in range(self.prediction_length-int((self.center_average_length-1)/2)-1):
            if i == 0:
                recovered_predicted[i] = self.recovered[-1] + new_recovered_and_prediction_center_averaged[i+len(self.date)-1]
            else:
                recovered_predicted[i] = recovered_predicted[i-1] + new_recovered_and_prediction_center_averaged[i+len(self.date)-1]
                
        self.recovered_predicted =recovered_predicted.astype(int)
    
        """
        fig, ax = plt.subplots()
        plt.plot(x_date[1:len(date)],new_cases,'o')
        plt.plot(x_date[1:len(date)],new_cases_averaged,'o')
        plt.plot(x_date[len(date)-fit_length:len(date)+prediction_length],new_cases_averaged_prediction,'r.')
        plt.plot(x_date[0:len(new_cases_and_prediction)],new_cases_and_prediction,'.')
        #plt.plot(x_date[len(date)-fit_length:len(date)],np.exp(np.polyval(new_cases_averaged_p_exp, np.arange(0,fit_length))),'.')
        plt.xlabel('date')
        plt.ylabel('new cases averaged past '+ str(average_length)+' days')
        plt.title(country)
        plt.xticks(rotation='vertical')
        plt.show()"""
        
        return self.recovered_predicted, self.recovered
    
    def makeRecoveredPrediction(self, prediction_length = 31):
        self.prediction_length = int(prediction_length)
        recovered_and_predicted = np.zeros((len(self.countries),len(self.date)+self.prediction_length))

        for i in range(len(self.countries)): 
            country = self.countries[i]
            #print("xxxxxxxx"+ str(self.prediction_length))
            self.recovered_predicted, self.recovered = self.predictRecovered(country, prediction_length)
            for j in range(len(self.date)):
                recovered_and_predicted[i,j] = self.recovered[j]
            for j in range(prediction_length):
                recovered_and_predicted[i,j+len(self.date)] = self.recovered_predicted[j]
        #create df of confirmed and predicted cases
        self.recovered_and_predicted = pd.DataFrame(data=recovered_and_predicted.T,columns=self.countries.to_numpy())
        return self.recovered_and_predicted
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
if __name__ == '__main__':
    print("========================")
    print("========================")
    print("Start the CodeWithClass.")
    
    
    confirmedFile = "time_series_covid19_confirmed_global.csv"
    deathsFile = "time_series_covid19_deaths_global.csv"
    recoveredFile = "time_series_covid19_recovered_global.csv"
    PredictionLength = 31
    
    newPrediction = Prediction(confirmedFile,deathsFile,recoveredFile)
    confirmed_and_predicted = newPrediction.makePrediction(PredictionLength)
    deaths_and_predicted = newPrediction.makeDeathPrediction(PredictionLength)
    recovered_and_predicted = newPrediction.makeRecoveredPrediction(PredictionLength)
    
    
    path2 = 'predictions'+r'\2day_prediction_'+newPrediction.x_date_str[len(newPrediction.df['date'])+2]+'.csv'
    path7 = 'predictions'+r'\7day_prediction_'+newPrediction.x_date_str[len(newPrediction.df['date'])+7]+'.csv'
    path30 = 'predictions'+r'\30day_prediction_'+newPrediction.x_date_str[len(newPrediction.df['date'])+30]+'.csv'
    
    newPrediction.saveInDesiredFormat(confirmed_and_predicted, deaths_and_predicted, recovered_and_predicted,path2, path7, path30)
    
    
    
    
    
    print("End the CodeWithClass.")
    print("========================")
    print("========================")