# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 11:24:28 2020

@author: loebn
"""

import numpy as np
import matplotlib.pyplot as plt
from datathon import data,country,day
import math
from scipy.optimize import curve_fit


class prediction:
    def __init__(self, newAnalysis, yAxis, coun, province = None):
        self.analysis = newAnalysis
        index = self.initialize(coun, province)
        self.data_x = []
        self.data_y = []
        self.data_y_growth = []
        for i,day in enumerate(self.analysis.countries[index].days):
            self.data_x.append(day.date)
            if yAxis == "recovered":
                self.data_y.append(day.recovered)
                if i == 0:
                    self.data_y_growth.append(day.recovered)
                if i != 0:
                    self.data_y_growth.append(self.analysis.countries[index].days[i].recovered - self.analysis.countries[index].days[i-1].recovered)
            if yAxis == "deaths":
                self.data_y.append(day.deaths)
                if i == 0:
                    self.data_y_growth.append(day.deaths)
                if i != 0:
                    self.data_y_growth.append(self.analysis.countries[index].days[i].deaths - self.analysis.countries[index].days[i-1].deaths)
            if yAxis == "confirmed":
                self.data_y.append(day.confirmed)
                if i == 0:
                    self.data_y_growth.append(day.confirmed)
                if i != 0:
                    self.data_y_growth.append(self.analysis.countries[index].days[i].confirmed - self.analysis.countries[index].days[i-1].confirmed)
            
        self.predict_x = []
        self.predict_y = []


    def initialize(self, Country, province):
        if province == None:
            for i,coun in enumerate(self.analysis.countries):
                if str(Country) == coun.name:
                    index = i
                    break
        if province != None:
            for i,coun in enumerate(self.analysis.countries):
                if str(Country) == coun.name:
                    if coun.province == province:
                        index = i
                        break
        return index
        
        
    def polyFit(self, degree, growth = None):
        polynom = np.polyfit(self.data_x, self.data_y, int(degree))
        if growth != None:
            polynom = np.polyfit(self.data_x, self.data_y_growth, int(degree))
        function = np.poly1d(polynom)
        return function
    
    
    def logPolyFit(self, degree, growth = "no"):
        polynom = np.polyfit(self.data_x, np.log(self.data_y), int(degree))
        if growth == "yes":
            polynom = np.polyfit(self.data_x, np.log(self.data_y_growth), int(degree))
        function = np.poly1d(polynom)
        return function
    
    
    def expFit(self,growth = None):
        expFunc = lambda t,a,b: a*np.exp(b*t)
        popt = curve_fit(expFunc, self.data_x, self.data_y)[0]
        if growth == "yes":
            popt = curve_fit(expFunc, self.data_x, self.data_y_growth)[0]
        return expFunc, popt
    
    
    def selfMadePolyFit(self,growth = None):
        selfMade = lambda t,a,b,c,d,e,f,g,x: a + b*(t+x) + c*(t+x)**2 + d * (t+x)**3 + e * (t+x)**4 + f * (t+x)**5 + g * (t+x)**6
        popt = curve_fit(selfMade, self.data_x, self.data_y)[0]
        if growth == "yes":
            popt = curve_fit(selfMade, self.data_x, self.data_y_growth)[0]
        return selfMade, popt
    
    
    def plotPrediction(self, function, endDay, popt=np.array([None]), growth = None):
        self.predict_x = np.arange(0, int(endDay), 1)
        if popt.any() == None:
            for x in self.predict_x:
                self.predict_y.append(float(function(x)))
        if popt.any() != None:
            self.predict_y = function(self.predict_x,*popt)
        
        plt.plot(self.predict_x,self.predict_y)
        if growth == "no":
            plt.scatter(self.data_x,self.data_y)
        if growth == "yes":
            plt.scatter(self.data_x,self.data_y_growth)
        
        plt.yscale('log')
        plt.ylim(1,200000)
        plt.show()
        

if __name__ == '__main__':
    print("========================")
    print("========================")
    print("Start the prediction.")

    newAnalysis = data("C:/Users/loebn/OneDrive/Desktop/Code/Datathon/data/")#just make here the path to the data folder
    newAnalysis.initializeIt()
    
    countryToInvestigate = "Switzerland"
    province = ""
    yAxis = "confirmed"
    growth = "no" #yes or no
    daysToPredict = 100#start from first case world wide
    
    print("Number of Countries/provinces: " + str(len(newAnalysis.countries)))
    print("Country to investigate: " + str(countryToInvestigate))
   
    #Exponential Fit:
    newPrediction = prediction(newAnalysis, yAxis, countryToInvestigate, province)
    function, popt = newPrediction.expFit(growth)
    newPrediction.plotPrediction(function, daysToPredict, popt, growth)
    
    #selfMade PolyFit
    newPrediction = prediction(newAnalysis, yAxis, countryToInvestigate, province)
    function, popt = newPrediction.selfMadePolyFit(growth)
    newPrediction.plotPrediction(function, daysToPredict, popt, growth)
      
    
    #polyfit:
    for degree in range(14):
        newPrediction = prediction(newAnalysis, yAxis, countryToInvestigate, province)
        function = newPrediction.polyFit(degree, growth)
        newPrediction.plotPrediction(function, daysToPredict, np.array([None]) ,growth)
    
    
    print("Finished the prediction.")
    print("========================")
    print("========================")