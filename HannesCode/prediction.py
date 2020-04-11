# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 11:24:28 2020

@author: loebn
"""

import numpy as np
import matplotlib.pyplot as plt
from datathon import data,country,day
import math

class prediction:
    def __init__(self, newAnalysis, yAxis, coun, province = None):
        self.analysis = newAnalysis
        index = self.initialize(coun, province)
        self.data_x = []
        self.data_y = []
        for day in self.analysis.countries[index].days:
            self.data_x.append(day.date)
            if yAxis == "recovered":
                self.data_y.append(day.recovered)
            if yAxis == "deaths":
                self.data_y.append(day.deaths)
            if yAxis == "confirmed":
                self.data_y.append(day.confirmed)
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
        
        
    def polyFit(self,degree):
        polynom = np.polyfit(self.data_x, self.data_y, int(degree))
        function = np.poly1d(polynom)
        return function
    
    def plotPrediction(self, function, endDay):
        self.predict_x = np.arange(0, int(endDay), 1)
        for x in self.predict_x:
            self.predict_y.append(float(function(x)))
        plt.plot(self.predict_x,self.predict_y)
        plt.scatter(self.data_x,self.data_y)
        
        plt.yscale('log')
        plt.ylim(1,200000)
        plt.show()
        

if __name__ == '__main__':
    print("========================")
    print("========================")
    print("Start the prediction.")

    newAnalysis = data("C:/Users/loebn/OneDrive/Desktop/Code/Datathon/data/")
    newAnalysis.initializeIt()
    
    countryToInvestigate = "Switzerland"
    province = ""
    yAxis = "confirmed"
    daysToPredict = 100#start from first case world wide
    
    print("Number of Countries/provinces: " + str(len(newAnalysis.countries)))
    print("Country to investigate: " + str(countryToInvestigate))
    
    
    #polyfit:
    for degree in range(14):
        newPrediction = prediction(newAnalysis, yAxis, countryToInvestigate, province)
        function = newPrediction.polyFit(degree)
        newPrediction.plotPrediction(function, daysToPredict)
    
    
    print("Finished the prediction.")
    print("========================")
    print("========================")