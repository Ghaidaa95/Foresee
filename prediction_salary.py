import pandas as pd
import numpy as np
import csv
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from flask import Flask
from flask import render_template
from flask import request

class salary:
    
    
    def __init__(self):
        self.data = list()
        self.yearsToPredict=list()
        self.x = list()
        self.x_value= list()
        self.y_value= list()
        self.x_valuenew= list()
        self.y_valuenew = list()

#         
            
        
    
    def name (self, spe):
        df = pd.read_csv('data/Wuzzuf_Job_Posts_SampleZ.csv')    
        df.head()    
        df['job_category1'].value_counts()        
        df=df[df['job_category1']==spe]       
        data = df[['job_title','salary_minimum','salary_maximum','post_date']]
        
        # create a list to calculate values of salary in one variable                
        salary= list()
        for num in range(len(df)):
            salary = np.array(df['salary_minimum'], df['salary_maximum'])
            np.mean(salary)        
        # create new data frame for salary and date of it            
        rf = pd.DataFrame({'salary':salary, 'post_date':data.post_date})
        rf.head()
        
        # training and testing the machine for prediction actual value of the growth rate for salary         
        x_train, x_test, y_train, y_test = train_test_split(rf[['post_date']] , rf[['salary']], test_size=0.20) 
        x_value = rf[['post_date']].values
        y_value = rf[['salary']].values        
        # draw line of prediction for actual values of salaries in years ago        
        salary_reg = linear_model.LinearRegression()
        salary_reg.fit(x_value, y_value)        
        # draw line of prediction for actual values of salaries for future     
        #yearsToPredict.append(2015)
        self.yearsToPredict.append(2016)
        self.yearsToPredict.append(2017)
        self.yearsToPredict.append(2018)
        #yearsToPredict.append(2019.0)
        #yearsToPredict.append(2020.0)       
        da=pd.DataFrame({"date":self.yearsToPredict})
        self.x= salary_reg.predict(da.values)
        self.x_valuenew=da.values
        self.y_valuenew=self.x
           

              
          
              
#p=salary()
#p.name(spe="IT/Software Development")
        newlist = []
        #print(p.x_valuenew)
        #print(p.y_valuenew)
        for i in range (len(self.x_valuenew)):
            newlist.append(str("{x:"+str(self.x_valuenew[i].item())+" , y:"+str(self.y_valuenew[i].item())+"}"))
        self.data = newlist
        print(self.data[0])
      
