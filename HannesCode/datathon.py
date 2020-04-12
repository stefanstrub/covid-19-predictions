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
        self.earliestDate = 99999999999999999999999999
       
        
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
                    
                if str(arguments[countryIndex]) in countryList:
                    provinceList = []
                    for coun in self.countries:
                        if coun.name == arguments[countryIndex]:
                            provinceList.append(coun.province)

                    if arguments[provinceIndex] not in provinceList:
                        countryList.append(str(arguments[countryIndex]))
                        newCountry = country(arguments[countryIndex])
                        newCountry.setProvince(arguments[provinceIndex])
                        newCountry.setDay(arguments[dateIndex], arguments[confirmedIndex], arguments[deathsIndex], arguments[recoveredIndex])
                        self.countries.append(newCountry)
                        #print(arguments[dateIndex])
                    if arguments[provinceIndex] in provinceList:
                        for coun in self.countries:
                            if coun.province == arguments[provinceIndex] and coun.name == arguments[countryIndex]:
                                coun.setDay(arguments[dateIndex], arguments[confirmedIndex], arguments[deathsIndex], arguments[recoveredIndex])

                if str(arguments[countryIndex]) not in countryList:   
                    countryList.append(str(arguments[countryIndex]))
                    newCountry = country(arguments[countryIndex])
                    newCountry.setProvince(arguments[provinceIndex])
                    newCountry.setDay(arguments[dateIndex], arguments[confirmedIndex], arguments[deathsIndex], arguments[recoveredIndex])
                    self.countries.append(newCountry)
                    #print(arguments[dateIndex])
                
                    
        self.findEarliestDate()
        for coun in self.countries:
            coun.earliestDate = self.earliestDate
        for coun in self.countries:
            coun.sortDays()
        
    
    def findEarliestDate(self):
        earliestDate = 9999999999999999
        for coun in self.countries:
            for day in coun.days:
                if int(day.date) < earliestDate:
                    earliestDate = int(day.date)  
        self.earliestDate = earliestDate


    def plotIt(self, yaxis, Country, province=None):
        if province == None:
            for i,coun in enumerate(self.countries):
                if str(Country) == coun.name:
                    index = i
                    break
        if province != None:
            for i,coun in enumerate(self.countries):
                if str(Country) == coun.name:
                    if coun.province == province:
                        index = i
                        break
            
        x = []
        y = []
        y1 = []
        y2 = []
        y3 = []
        for day in self.countries[index].days:
            if str(yaxis) == "recovered" or str(yaxis) == "all":
                y.append(day.recovered)
                y1.append(day.recovered)
                ylabel = "Number of recovered cases"
            if str(yaxis) == "deaths" or str(yaxis) == "all":
                y.append(day.deaths)
                y2.append(day.deaths)
                ylabel = "Number of deaths cases"
            if str(yaxis) == "confirmed" or str(yaxis) == "all":
                y.append(day.confirmed)
                y3.append(day.confirmed)
                ylabel = "Number of confirmed cases"
            x.append(day.date)

        if str(yaxis) != "all":
            plt.scatter(x , y,  c='b', label = str(yaxis))
            
        if str(yaxis) == "all":
            plt.scatter(x , y1,  c='b', label = "recovered")
            plt.scatter(x , y2,  c='r', label = "deaths")
            plt.scatter(x , y3,  c='g', label = "confirmed")
            ylabel = "Total Number"
            plt.legend(loc="lower right")
            
        plt.yscale('log')
        plt.ylim(1,200000)
        plt.ylabel(ylabel)
        plt.legend(loc="lower right")
        plt.xlabel("Days after the the first case reported world wide.")   
        plt.show()
       
      
class country:
    def __init__(self, name):
        self.name = str(name)
        self.province = ""
        self.days = []
        self.earliestDate = 0
        self.firstDay = 0


    def setProvince(self, province=None):
        self.province = province
        
        
    def setDay(self, date, confirmed, deaths, recovered):
        if len(confirmed) == 0 or confirmed == "\n":
            confirmed = 0
        if len(deaths) == 0 or deaths == "\n":
            deaths = 0 
        if len(recovered) == 0 or recovered == "\n":
            recovered = 0  
        newDay = day(date, confirmed, deaths, recovered)
        self.days.append(newDay)
        
        
    def sortDays(self):
        helpList1 = []
        helpList2 = []
        helpList3 = []
        helpList4 = []
        for day in self.days:
            helpList1.append(int(day.date))
            helpList2.append(int(day.confirmed))
            helpList3.append(int(day.deaths))
            helpList4.append(int(day.recovered))
        indices = np.argsort(helpList1)
        self.date2int(helpList1[0])
        
        i = 0
        for ind in indices:
            self.days[ind].date = self.firstDay + i
            i+=1
            self.days[ind].confirmed = helpList2[ind]
            self.days[ind].deaths = helpList3[ind]
            self.days[ind].recovered = helpList4[ind]
   
    
    def date2int(self, day1):
        if str(day1)[-3] == "1":
            sol = int(str(day1)[-2:]) - int(str(self.earliestDate)[-2:])
        if str(day1)[-3] == "2":#februar
            sol = int(str(day1)[-2:]) + 31 - int(str(self.earliestDate)[-2:])
        if str(day1)[-3] == "3":#march
            sol = int(str(day1)[-2:]) + 31 + 29 - int(str(self.earliestDate)[-2:])
        if str(day1)[-3] == "4":#april
            sol = int(str(day1)[-2:]) + 31 + 29 + 31 - int(str(self.earliestDate)[-2:])
        self.firstDay = sol   
    
        
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
        return int(date)
    

if __name__ == '__main__':
    print("========================")
    print("========================")
    print("Start the datathon.")

    newAnalysis = data("C:/Users/loebn/OneDrive/Desktop/Code/Datathon/data/") #just make here the path to the data folder
    newAnalysis.initializeIt()
    
    countryToInvestigate = "Switzerland"
    province = ""
    
    print("Number of Countries/provinces: " + str(len(newAnalysis.countries)))
    print("Country to investigate: " + str(countryToInvestigate))
    for coun in (newAnalysis.countries):
        if "Switzerland" == coun.name:
            print(coun.province)
    
    
    newAnalysis.plotIt("confirmed", countryToInvestigate, province) 
    newAnalysis.plotIt("deaths", countryToInvestigate, province) 
    newAnalysis.plotIt("recovered", countryToInvestigate, province) 
    newAnalysis.plotIt( "all", countryToInvestigate,province)    

    print("Finished the datathon.")
    print("========================")
    print("========================")