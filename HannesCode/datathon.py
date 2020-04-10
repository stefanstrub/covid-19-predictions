# -*- coding: utf-8 -*-

"""
Created on Fri Apr 10 10:10:07 2020

@author: loebn
"""

import os
import numpy as np
import matplotlib.pyplot as plt
class data:
    def __init__(self, datafolder):
        self.datafolder = str(datafolder)
        root, dirs, files = next(os.walk(datafolder))
        self.evaluationFiles = [os.path.join(root, file) for file in files if "-" in file]
        self.countries = []
        
    def initializeIt(self):
        countryList = []
        for i,file in enumerate(self.evaluationFiles):
            #print("Initialize File: " + str(file.split("/")[-1]))
            with open(file,"r") as f:
                content = f.readlines()
            
            for l,line in enumerate(content):
                arguments = line.split(",")
                if l == 0:
                    length = len(arguments)
                    for a,argument in enumerate(arguments):
                        if "country" in str(argument) or "Country" in str(argument):
                            countryIndex = int(a)
                        if "Update" in str(argument) or "update" in str(argument):
                            dateIndex = int(a)
                        if "confirmed" in str(argument) or "Confirmed" in str(argument):
                            confirmedIndex = int(a)
                        if "deaths" in str(argument) or "Deaths" in str(argument):
                            deathsIndex = int(a)
                        if "recovered" in str(argument) or "Recovered" in str(argument):
                            recoveredIndex = int(a)
                        if "province" in str(argument) or "Province" in str(argument):
                            provinceIndex = int(a)
                    continue
                if len(arguments) != length:#because of names with "," in them
                    continue#just ignoring all the names with ","
# =============================================================================
#                     diff = len(arguments)-length
#                     if str(arguments[countryIndex]) in countryList:
#                         index = countryList.index(str(arguments[countryIndex]))
#                         self.countries[index].setDay(arguments[dateIndex+diff], arguments[confirmedIndex+diff], arguments[deathsIndex+diff], arguments[recoveredIndex+diff])
#                     
#                     if str(arguments[countryIndex]) not in countryList:
#                         countryList.append(str(arguments[countryIndex]))
#                         newCountry = country(arguments[countryIndex])
#                         newCountry.setProvince(arguments[provinceIndex])
#                         newCountry.setDay(arguments[dateIndex+diff], arguments[confirmedIndex+diff], arguments[deathsIndex+diff], arguments[recoveredIndex+diff])
#                         self.countries.append(newCountry)
# =============================================================================
                if str(arguments[countryIndex]) in countryList:
                    index = countryList.index(str(arguments[countryIndex]))
                    self.countries[index].setDay(arguments[dateIndex], arguments[confirmedIndex], arguments[deathsIndex], arguments[recoveredIndex])
                    
                if str(arguments[countryIndex]) not in countryList:
                    countryList.append(str(arguments[countryIndex]))
                    newCountry = country(arguments[countryIndex])
                    newCountry.setProvince(arguments[provinceIndex])
                    newCountry.setDay(arguments[dateIndex], arguments[confirmedIndex], arguments[deathsIndex], arguments[recoveredIndex])
                    self.countries.append(newCountry)
                    #print(arguments[dateIndex])
    
    def plotIt(self, Country, yaxis):
        for i,coun in enumerate(self.countries):
            if str(Country) == coun.name:
                index = i
                break
        x=[]
        y=[]
        for day in self.countries[index].days:
            if str(yaxis) == "recovered":
                yAxis = day.recovered
            if str(yaxis) == "deaths":
                yAxis = day.deaths
            if str(yaxis) == "confirmed":
                yAxis = day.confirmed
            
            x.append(day.date)
            y.append(yAxis)
        if str(yaxis) != "all":
            plt.scatter(x , y,  c='b')
        
            
        #plt.yticks(np.arange(min(y), max(y)+1, 1.0))
        plt.show()
             
      
class country:
    def __init__(self, name):
        self.name = str(name)
        self.province = ""
        self.days = []

    def setProvince(self, province=None):
        self.province = province
        
    def setDay(self, date, confirmed, deaths, recovered):
        newDay = day(date, confirmed, deaths, recovered)
        self.days.append(newDay)
        
class day:
    def __init__(self, date, confirmed, deaths, recovered):
        self.date = self.adjustDate(str(date))
        self.confirmed = confirmed
        self.deaths = deaths
        self.recovered = recovered
    
    def adjustDate(self, date):
        if "T" in date:
            date = date.split("T")[0]
        if " " in date:
            date = date.split(" ")[0]
        if "-" in date:
            date = date.replace("-","")
        if "/" in date:
            string = date.split("/")
            if len(string[0])==1:
                string[0] = "0"+string[0]
            if len(string[1])==1:
                string[1] = "0"+string[1]
            if len(string[2])==2:
                string[2] = "20"+string[2]
            date = string[2]+string[0]+string[1]
        date = date.split("2020")[1]    
        return date
    

if __name__ == '__main__':
    print("========================")
    print("========================")
    print("Start the datathon.")

    newAnalysis = data("C:/Users/loebn/OneDrive/Desktop/Code/Datathon/data/")
    newAnalysis.initializeIt()
    print(len(newAnalysis.countries))
# =============================================================================
#     for coun in newAnalysis.countries:
#         if "Switzerland" == coun.name:
#             print(coun.name)
#     for day in newAnalysis.countries[0].days:
#         print(day.date)
# =============================================================================
    newAnalysis.plotIt("Switzerland", "confirmed")    
        
    
    
    print("Finished the datathon.")
    print("========================")
    print("========================")